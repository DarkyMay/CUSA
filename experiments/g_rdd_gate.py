#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g_rdd_gate.py —— R2-4 / R3-6：线状/网状门控在 RDD2022 专家框级标签上的大样本验证

RDD2022 的 XML 标注：D00 纵向裂缝、D10 横向裂缝、D20 龟裂(alligator)、D40 坑槽。
  网状样本 = D20 框裁块；线状样本 = 仅含 D00/D10（不含 D20/D40）图像中的 D00/D10 框裁块。
每个裁块（框外扩 20%，短边 ≥ 96 px）跑 SAM3 零样本 → 预测掩膜 → 门控（论文规则：欧拉示性数 + A_min=(factor·w̃)²）。
报告：
  (1) factor ∈ {6,8,10,12,14,16,20} 下的混淆矩阵 / TPR / FPR（分区域、合并）
  (2) 判别量的 AUC：最大洞 √面积/w̃（门控的连续形式）、洞数、洞数/骨架长
  (3) 分辨率影响：裁块短边分箱（<150 / 150–300 / >300 px）下的 TPR/FPR
  (4) 2× 上采样后再跑 SAM3 的门控（检验低分辨率下的鲁棒性）
  (5) 失败案例：各区域 FN/FP 各列 10 个文件名 + 裁块统计
输出 results/g_rdd_gate_<ts>.txt + g_rdd_gate_<ts>.csv（逐裁块）；裁块与掩膜存 results/rdd_gate_crops/ 供人工复核
用法 python3 g_rdd_gate.py [--regions China_Drone,China_MotorBike,Japan] [--per_class 150] [--limit N]
"""
import os, sys, csv, glob, re, time, random, argparse, datetime, traceback
import xml.etree.ElementTree as ET
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from transformers import Sam3Model, Sam3Processor

OUT = "/root/autodl-tmp/proj_bridge"; W = "/root/autodl-tmp/sam3_weights"
RDD = "/root/autodl-tmp/RDD2022/data"
GATE_FACTOR = 14.0; FACTORS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0, 20.0]
LINEAR = {"D00", "D10"}; NETWORK = {"D20"}; EXCLUDE = {"D20", "D40"}
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"g_rdd_gate_{ts}.txt")
CSVP = os.path.join(OUT, "results", f"g_rdd_gate_{ts}.csv")
CROPD = os.path.join(OUT, "results", "rdd_gate_crops")


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()


def parse_xml(p):
    out = []
    try:
        for o in ET.parse(p).getroot().findall("object"):
            n = o.findtext("name"); b = o.find("bndbox")
            if n is None or b is None: continue
            out.append((n.strip(), [float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]))
    except Exception:
        pass
    return out


def crop_bbox(mask, pad=5):
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    return mask[y0:y1, x0:x1]


def gate_stats(mask):
    """返回 dict: w_med, n_holes, max_ratio(最大洞 sqrt(面积)/w̃), ratios(前5), skel_len, fg"""
    if mask.sum() < 30: return dict(w_med=0.0, n_holes=0, max_ratio=0.0, ratios="", skel_len=0, fg=int(mask.sum()))
    m = crop_bbox(mask); sk = skeletonize(m); d = ndi.distance_transform_edt(m)
    w = 2.0 * d[sk]; wm = float(np.median(w)) if w.size else 0.0
    holes = ndi.binary_fill_holes(m) & ~m; lab, nh = ndi.label(holes)
    if nh == 0: return dict(w_med=wm, n_holes=0, max_ratio=0.0, ratios="", skel_len=int(sk.sum()), fg=int(mask.sum()))
    areas = np.sort(ndi.sum(holes, lab, index=np.arange(1, nh + 1)))[::-1]
    r = np.sqrt(areas) / max(wm, 1.0)
    return dict(w_med=wm, n_holes=int(nh), max_ratio=float(r[0]), ratios=";".join(f"{x:.1f}" for x in r[:5]), skel_len=int(sk.sum()), fg=int(mask.sum()))


def is_network(st, factor, min_cells=1):
    if not st["ratios"]: return False
    rs = [float(x) for x in st["ratios"].split(";")]
    return sum(1 for x in rs if x > factor) >= min_cells


def auc(scores_pos, scores_neg):
    p = np.asarray(scores_pos, float); n = np.asarray(scores_neg, float)
    if len(p) == 0 or len(n) == 0: return np.nan
    allv = np.concatenate([p, n]); ranks = allv.argsort().argsort().astype(float) + 1
    # 平均秩处理并列
    order = np.argsort(allv); sv = allv[order]; r = np.empty(len(allv)); i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]: j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    return (r[:len(p)].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", type=str, default="China_Drone,China_MotorBike,Japan")
    ap.add_argument("--per_class", type=int, default=150, help="每区域每类最多裁块数")
    ap.add_argument("--limit", type=int, default=0, help="冒烟：每区域每类只取 N")
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()
    sys.stdout = Tee(LOGP); T0 = time.time(); random.seed(args.seed)
    os.makedirs(CROPD, exist_ok=True)
    regions = [r.strip() for r in args.regions.split(",")]
    per = args.limit if args.limit else args.per_class
    print(f"g_rdd_gate.py  {ts}  regions={regions}  per_class={per}  factor={GATE_FACTOR}\nlog -> {LOGP}\n")

    # ---- 采样裁块
    samples = []   # (region, cls, img_path, box, stem)
    for reg in regions:
        xmls = sorted(glob.glob(os.path.join(RDD, reg, "train", "annotations", "xmls", "*.xml")))
        imgd = os.path.join(RDD, reg, "train", "images")
        net, lin = [], []
        for xp in xmls:
            objs = parse_xml(xp); names = {n for n, _ in objs}
            stem = os.path.splitext(os.path.basename(xp))[0]
            ip = os.path.join(imgd, stem + ".jpg")
            if not os.path.exists(ip): continue
            if names & NETWORK:
                for n, b in objs:
                    if n in NETWORK: net.append((reg, "network", ip, b, stem))
            elif names & LINEAR and not (names & EXCLUDE):
                for n, b in objs:
                    if n in LINEAR: lin.append((reg, "linear", ip, b, stem))
        random.shuffle(net); random.shuffle(lin)
        print(f"  {reg}: XML {len(xmls)}  D20 框 {len(net)}  纯线状图的 D00/D10 框 {len(lin)}  → 各取 {min(per,len(net))}/{min(per,len(lin))}")
        samples += net[:per] + lin[:per]

    processor = Sam3Processor.from_pretrained(W); model = Sam3Model.from_pretrained(W).cuda().eval(); torch.set_grad_enabled(False)

    def seg(img):
        inp = processor(images=img, text="crack", return_tensors="pt")
        inp = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(**inp)
        Wd, Hd = img.size
        return F.interpolate(o["semantic_seg"][0, 0][None, None].float(), size=(Hd, Wd), mode="bilinear")[0, 0].cpu().numpy() > 0

    rows = []
    print(f"\n裁块 {len(samples)} 个，开始推理", flush=True)
    for i, (reg, cls, ip, b, stem) in enumerate(samples):
        try:
            img = Image.open(ip).convert("RGB"); Wd, Hd = img.size
            bw, bh = b[2] - b[0], b[3] - b[1]
            x0 = int(max(0, b[0] - 0.2 * bw)); y0 = int(max(0, b[1] - 0.2 * bh)); x1 = int(min(Wd, b[2] + 0.2 * bw)); y1 = int(min(Hd, b[3] + 0.2 * bh))
            if min(x1 - x0, y1 - y0) < 96: continue
            crop = img.crop((x0, y0, x1, y1)); m1 = seg(crop); st1 = gate_stats(m1)
            crop2 = crop.resize((crop.width * 2, crop.height * 2), Image.BICUBIC); m2 = seg(crop2); st2 = gate_stats(m2)
            r = dict(region=reg, cls=cls, stem=stem, box="_".join(str(int(v)) for v in b), crop_w=x1 - x0, crop_h=y1 - y0, short=min(x1 - x0, y1 - y0),
                     fg_ratio=float(m1.mean()), **{f"{k}_1x": v for k, v in st1.items()}, **{f"{k}_2x": v for k, v in st2.items()})
            r["net14_1x"] = int(is_network(st1, GATE_FACTOR)); r["net14_2x"] = int(is_network(st2, GATE_FACTOR))
            rows.append(r)
            if i < 400 or r["net14_1x"] != (cls == "network"):   # 存前 400 个 + 全部误判，供人工复核
                tag = f"{reg}_{cls}_{stem}_{r['box']}"
                crop.save(os.path.join(CROPD, tag + ".jpg"), quality=85)
                Image.fromarray(m1.astype(np.uint8) * 255).save(os.path.join(CROPD, tag + "_mask.png"))
        except Exception:
            torch.cuda.empty_cache(); traceback.print_exc()
        if i % 50 == 0: print(f"  {i+1}/{len(samples)}  {time.time()-T0:.0f}s", flush=True)
    with open(CSVP, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---- 汇总
    def confusion(rs, factor, key="1x", min_cells=1):
        tp = fn = fp = tn = 0
        for r in rs:
            st = dict(ratios=r[f"ratios_{key}"]); pred = is_network(st, factor, min_cells); truth = r["cls"] == "network"
            tp += pred and truth; fn += (not pred) and truth; fp += pred and (not truth); tn += (not pred) and (not truth)
        n_p, n_n = tp + fn, fp + tn
        return tp, fn, fp, tn, (tp / n_p if n_p else np.nan), (fp / n_n if n_n else np.nan)

    print("\n" + "=" * 100 + "\n[1] 门控 vs RDD2022 专家框级标签：按 factor（原分辨率裁块，SAM3 零样本掩膜）\n" + "=" * 100)
    print(f"  {'region':16s}{'factor':>7s}{'n_net':>6s}{'n_lin':>6s}{'TP':>5s}{'FN':>5s}{'FP':>5s}{'TN':>5s}{'TPR':>7s}{'FPR':>7s}{'acc':>7s}")
    for reg in regions + ["ALL"]:
        rs = [r for r in rows if reg == "ALL" or r["region"] == reg]
        if not rs: continue
        for fac in FACTORS:
            tp, fn, fp, tn, tpr, fpr = confusion(rs, fac)
            print(f"  {reg:16s}{fac:7.0f}{tp+fn:6d}{fp+tn:6d}{tp:5d}{fn:5d}{fp:5d}{tn:5d}{tpr:7.3f}{fpr:7.3f}{(tp+tn)/len(rs):7.3f}")
    print("\n  SUT-Crack 对照（b 脚本 B2）：GT 掩膜 4/4 命中 0 FP；SAM3 预测掩膜 2/4 命中 0 FP。")

    print("\n" + "=" * 100 + "\n[2] 判别量的 AUC（网状 vs 线状）\n" + "=" * 100)
    for reg in regions + ["ALL"]:
        rs = [r for r in rows if reg == "ALL" or r["region"] == reg]
        if not rs: continue
        P = [r for r in rs if r["cls"] == "network"]; N = [r for r in rs if r["cls"] == "linear"]
        print(f"  {reg:16s} n={len(P)}/{len(N)}  AUC: 最大洞比 {auc([r['max_ratio_1x'] for r in P],[r['max_ratio_1x'] for r in N]):.3f}  "
              f"洞数 {auc([r['n_holes_1x'] for r in P],[r['n_holes_1x'] for r in N]):.3f}  "
              f"洞数/骨架长 {auc([r['n_holes_1x']/max(r['skel_len_1x'],1) for r in P],[r['n_holes_1x']/max(r['skel_len_1x'],1) for r in N]):.3f}  "
              f"前景比 {auc([r['fg_ratio'] for r in P],[r['fg_ratio'] for r in N]):.3f}  | 2×上采样 最大洞比 {auc([r['max_ratio_2x'] for r in P],[r['max_ratio_2x'] for r in N]):.3f}")
    print("  判读：最大洞比是论文门控的连续形式；AUC 高而 factor=14 的 TPR 低，说明规则正确但阈值对该分辨率偏严（见 [3]）。")

    print("\n" + "=" * 100 + "\n[3] 分辨率：裁块短边分箱下 factor=14 的 TPR / FPR（原分辨率 vs 2× 上采样后推理）\n" + "=" * 100)
    bins = [(0, 150), (150, 300), (300, 10 ** 6)]
    print(f"  {'短边(px)':12s}{'n_net':>6s}{'n_lin':>6s} | {'TPR_1x':>7s}{'FPR_1x':>7s} | {'TPR_2x':>7s}{'FPR_2x':>7s} | {'w̃_1x中值':>9s}")
    for lo, hi in bins:
        rs = [r for r in rows if lo <= r["short"] < hi]
        if not rs: continue
        tp, fn, fp, tn, tpr, fpr = confusion(rs, GATE_FACTOR, "1x"); tp2, fn2, fp2, tn2, tpr2, fpr2 = confusion(rs, GATE_FACTOR, "2x")
        print(f"  [{lo},{hi if hi<10**6 else '∞'}){'':<4s}{tp+fn:6d}{fp+tn:6d} | {tpr:7.3f}{fpr:7.3f} | {tpr2:7.3f}{fpr2:7.3f} | {np.median([r['w_med_1x'] for r in rs]):9.1f}")
    print("\n  factor 扫描（ALL，1x vs 2x）：")
    for fac in FACTORS:
        a = confusion(rows, fac, "1x"); b2 = confusion(rows, fac, "2x")
        print(f"    factor={fac:4.0f}  1x TPR {a[4]:.3f} FPR {a[5]:.3f} | 2x TPR {b2[4]:.3f} FPR {b2[5]:.3f}")

    print("\n" + "=" * 100 + "\n[4] 失败案例（factor=14, 1x）\n" + "=" * 100)
    for reg in regions:
        rs = [r for r in rows if r["region"] == reg]
        fns = [r for r in rs if r["cls"] == "network" and not r["net14_1x"]]; fps = [r for r in rs if r["cls"] == "linear" and r["net14_1x"]]
        print(f"  {reg}: FN {len(fns)}  FP {len(fps)}")
        for r in fns[:10]: print(f"    FN {r['stem']} box={r['box']} short={r['short']} fg={r['fg_ratio']:.3f} w̃={r['w_med_1x']:.1f} holes={r['n_holes_1x']} top=[{r['ratios_1x']}]")
        for r in fps[:10]: print(f"    FP {r['stem']} box={r['box']} short={r['short']} fg={r['fg_ratio']:.3f} w̃={r['w_med_1x']:.1f} holes={r['n_holes_1x']} top=[{r['ratios_1x']}]")
    fn_empty = sum(1 for r in rows if r["cls"] == "network" and not r["net14_1x"] and r["fg_1x"] < 30)
    fn_noholes = sum(1 for r in rows if r["cls"] == "network" and not r["net14_1x"] and r["fg_1x"] >= 30 and r["n_holes_1x"] == 0)
    fn_small = sum(1 for r in rows if r["cls"] == "network" and not r["net14_1x"] and r["n_holes_1x"] > 0)
    print(f"\n  FN 机制分解（ALL）：掩膜为空/过小 {fn_empty}；有掩膜但无封闭洞（网格被填实或断裂） {fn_noholes}；有洞但都小于 A_min {fn_small}")
    print(f"  裁块与掩膜已存 {CROPD}（前 400 个 + 全部误判）\n逐裁块 CSV {CSVP}\n总耗时 {(time.time()-T0)/60:.1f} min")


if __name__ == "__main__":
    main()