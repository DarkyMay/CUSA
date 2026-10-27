#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
closeloop_sutcrack_v2.py —— SUT-Crack 严重度闭环（重写版）

【相对旧版的改动】
 1. 框条件重设计。旧版五条件(oracle/jit10/20/30/wholeimg)建立在"框有效"的假设上，
    而 probe_box_effect 已证实归一化cxcywh框被processor丢弃，那组实验无意义。
    新三条件：
      text_full   整图纯文本            —— 主协议，与分割部分结论一致
      roi_crop    按检测框裁剪后纯文本    —— 检验"检测框该当ROI而非prompt"
      box_prompt  整图+框(正确像素xyxy)  —— 记录框当prompt的危害
 2. 变体：零训练 与 v3微调（零训练是主线）。
 3. ★命题调整★ 不再声称"准确测量绝对mm"——SUT-Crack 只有真值长度、无真值宽度，
    且掩码存在过/欠分割与标注惯例双重偏差，绝对精度无法验证。
    改为量化【分级结果对通常被隐含的选择有多敏感】：掩码来源 × 缝宽口径 × 规范体系。
 4. 可靠性曲线按【图】聚合，不再按(图×条件)计数（旧版伪重复严重）。
 5. GT几何量每图只算一次并缓存；预测掩码几何量裁剪到bbox加速。

标定 0.2219 mm/px（EXIF光学 + 骨架经验 双重独立佐证，差2%）
用法: 冒烟 python3 closeloop_sutcrack_v2.py --limit 10
      全量 python3 closeloop_sutcrack_v2.py
"""
import os, sys, glob, re, csv, time, argparse, datetime
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image, ImageOps
from scipy.stats import spearmanr
from transformers import Sam3Model, Sam3Processor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pavement_codes as pc

Image.MAX_IMAGE_PIXELS = None
OUT = "/root/autodl-tmp/proj_bridge"
W = "/root/autodl-tmp/sam3_weights"
CK_V3 = os.path.join(OUT, "ckpts", "sam3_ft_v3_textonly_best.pt")
ROOT = "/root/autodl-tmp/SUT-Crack"
XMLD = os.path.join(ROOT, "2-Object Detection")
IMGD = os.path.join(ROOT, "1-Segmentation", "Original Image")
GTD = os.path.join(ROOT, "1-Segmentation", "Ground Truth")
DEV = "cuda"; TEXT = "crack"; MMPX = pc.SUTCRACK_MMPX
CONDS = ["text_full", "roi_crop", "box_prompt"]
STATS = ["w_median", "w_mean", "w_p95", "w_max"]

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
CSVP = os.path.join(OUT, "results", f"closeloop_v2_{ts}.csv")


def dice_np(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def numkey(p):
    m = re.findall(r'\d+', os.path.basename(p)); return int(m[0]) if m else 0


def parse_xml(p):
    out = []
    for o in ET.parse(p).getroot().findall('object'):
        b = o.find('bndbox')
        out.append([float(b.findtext('xmin')), float(b.findtext('ymin')),
                    float(b.findtext('xmax')), float(b.findtext('ymax'))])
    return out


@torch.no_grad()
def seg(model, processor, img, box_px=None):
    """box_px=None -> 纯文本; 否则按【像素xyxy】传框（已验证的正确格式）"""
    if box_px is None:
        inp = processor(images=img, text=TEXT, return_tensors="pt")
    else:
        inp = processor(images=img, text=TEXT, input_boxes=[[list(box_px)]],
                        input_boxes_labels=[[1]], return_tensors="pt")
    inp = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}
    with torch.autocast(DEV, dtype=torch.bfloat16):
        o = model(**inp)
    Wd, Hd = img.size
    return F.interpolate(o["semantic_seg"][0, 0][None, None].float(),
                         size=(Hd, Wd), mode="bilinear")[0, 0].cpu().numpy() > 0


def predict(model, processor, img, boxes, cond):
    Wd, Hd = img.size
    if cond == "text_full":
        return seg(model, processor, img)
    if cond == "box_prompt":
        m = np.zeros((Hd, Wd), bool)
        for b in boxes:
            m |= seg(model, processor, img, box_px=b)
        return m
    # roi_crop: 裁到框(+10%边距)，框内纯文本，再贴回原图
    m = np.zeros((Hd, Wd), bool)
    for b in boxes:
        bw, bh = b[2] - b[0], b[3] - b[1]
        x0 = int(max(0, b[0] - 0.1 * bw)); y0 = int(max(0, b[1] - 0.1 * bh))
        x1 = int(min(Wd, b[2] + 0.1 * bw)); y1 = int(min(Hd, b[3] + 0.1 * bh))
        if x1 - x0 < 32 or y1 - y0 < 32: continue
        sub = seg(model, processor, img.crop((x0, y0, x1, y1)))
        m[y0:y1, x0:x1] |= sub
    return m


def geom(mask, mm=MMPX):
    """裁到bbox加速，几何量与尺度无关部分不受影响"""
    if mask.sum() < 30: return None
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - 60), min(mask.shape[0], ys.max() + 61)
    x0, x1 = max(0, xs.min() - 60), min(mask.shape[1], xs.max() + 61)
    g = pc.measure(mask[y0:y1, x0:x1], mm_per_px=mm)
    if g: g["area_px"] = int(mask.sum())
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    print("=" * 100)
    print(f"SUT-Crack 严重度闭环 v2 | 标定 {MMPX} mm/px | semantic单头 τ=0")
    print("=" * 100)

    processor = Sam3Processor.from_pretrained(W)
    model = Sam3Model.from_pretrained(W).to(DEV)
    base_sd = {k: v.clone() for k, v in model.state_dict().items()}
    model.eval(); torch.set_grad_enabled(False)
    for p in model.parameters(): p.requires_grad_(False)

    xmls = sorted(glob.glob(os.path.join(XMLD, "*.xml")), key=numkey)
    if args.limit: xmls = xmls[:args.limit]
    print(f"图像 {len(xmls)} 张")

    # ---- GT 几何量：每图只算一次 ----
    print("\n[1] GT 几何量", flush=True)
    GT = {}
    for i, xp in enumerate(xmls):
        stem = os.path.splitext(os.path.basename(xp))[0]
        gp = os.path.join(GTD, stem + ".png")
        if not os.path.exists(gp): continue
        gt = np.array(Image.open(gp).convert("L")) > 127
        g = geom(gt)
        if g: GT[stem] = (gt, g)
        if (i + 1) % 40 == 0: print(f"  ...{i+1}/{len(xmls)} {time.time()-t0:.0f}s", flush=True)
    print(f"  有效 {len(GT)} 张，线状 {sum(1 for _,g in GT.values() if g['ctype']=='线状')}")

    VARIANTS = [("zeroshot", None)] + ([("ft_v3", CK_V3)] if os.path.exists(CK_V3) else [])
    rows = []
    for vname, ck in VARIANTS:
        model.load_state_dict(base_sd, strict=True)
        if ck:
            _, un = model.load_state_dict(torch.load(ck, map_location=DEV), strict=False)
            if un: print(f"!! {vname} 键名不匹配"); continue
        model.eval()
        print(f"\n[2] 推理 · {vname}", flush=True)
        for i, xp in enumerate(xmls):
            stem = os.path.splitext(os.path.basename(xp))[0]
            if stem not in GT: continue
            ip = os.path.join(IMGD, stem + ".jpg")
            img = ImageOps.exif_transpose(Image.open(ip)).convert("RGB")
            gt, gg = GT[stem]
            if (img.size[1], img.size[0]) != gt.shape:
                print(f"  !! {stem} 尺寸不符，跳过"); continue
            boxes = parse_xml(xp)
            for cond in CONDS:
                try:
                    pm = predict(model, processor, img, boxes, cond)
                except Exception as e:
                    torch.cuda.empty_cache()
                    print(f"  !! {stem}/{cond}: {type(e).__name__}: {str(e)[:90]}"); continue
                pg = geom(pm)
                r = dict(variant=vname, cond=cond, stem=stem, nbox=len(boxes),
                         dice=dice_np(pm, gt), gt_ctype=gg["ctype"],
                         fg_ratio=(pm.mean() / gt.mean()) if gt.mean() > 0 else np.nan)
                for s in STATS:
                    r["gt_" + s] = gg[s + "_mm"]
                    r["pred_" + s] = pg[s + "_mm"] if pg else None
                rows.append(r)
            if (i + 1) % 20 == 0:
                el = time.time() - t0
                print(f"  ...{i+1}/{len(xmls)} 已用{el/60:.1f}min", flush=True)

    if not rows:
        print("!! 无结果"); sys.exit(1)

    def sel(**kw): return [r for r in rows if all(r[k] == v for k, v in kw.items())]
    lin = lambda rs: [r for r in rs if r["gt_ctype"] == "线状"]

    # ================= [3] 分割与框用法 =================
    print("\n" + "=" * 100)
    print("[3] 分割质量：检测框该当 ROI 还是 prompt？")
    print("=" * 100)
    print(f"  {'变体':<10}{'条件':<12}{'n':>5}{'Dice':>9}{'预测/GT前景比':>15}")
    for vname, _ in VARIANTS:
        for c in CONDS:
            s = sel(variant=vname, cond=c)
            if not s: continue
            print(f"  {vname:<10}{c:<12}{len(s):>5}{np.mean([r['dice'] for r in s]):>9.4f}"
                  f"{np.nanmean([r['fg_ratio'] for r in s]):>15.2f}")
    print("\n  前景比 >1 = 过分割（缝宽系统性偏大）；<1 = 欠分割。")
    print("  若 roi_crop 明显优于 box_prompt -> 检测框应当作ROI裁剪，而非作为prompt送入模型。")

    # ================= [4] 缝宽保真度 =================
    print("\n" + "=" * 100)
    print("[4] 缝宽保真度（预测掩码 vs GT掩码，仅线状）")
    print("=" * 100)
    print(f"  {'变体':<10}{'条件':<12}{'口径':<10}{'GT均值mm':>10}{'预测均值mm':>11}"
          f"{'偏差mm':>9}{'相对偏差':>9}{'MAE mm':>9}{'Spearman':>10}")
    for vname, _ in VARIANTS:
        for c in CONDS:
            s = [r for r in lin(sel(variant=vname, cond=c)) if r["pred_w_median"] is not None]
            if len(s) < 5: continue
            for st in STATS:
                g = np.array([r["gt_" + st] for r in s]); p = np.array([r["pred_" + st] for r in s])
                rho = spearmanr(p, g).correlation
                print(f"  {vname:<10}{c:<12}{st:<10}{g.mean():>10.2f}{p.mean():>11.2f}"
                      f"{p.mean()-g.mean():>+9.2f}{(p.mean()-g.mean())/g.mean():>+9.1%}"
                      f"{np.mean(np.abs(p-g)):>9.2f}{rho:>10.3f}")
    print("\n  Spearman高而相对偏差大 = 排序可靠、绝对值不可靠 -> 相对分级可用，绝对mm需谨慎。")

    # ================= [5] 分级敏感性矩阵 ★核心★ =================
    print("\n" + "=" * 100)
    print("[5] 分级敏感性：等级分布对【掩码来源 × 缝宽口径 × 规范】有多敏感  ★核心★")
    print("=" * 100)
    print("  JTG 5210-2018 线状裂缝: 轻度<=3mm / 重度>3mm (两级)")
    print("  ASTM D6433 纵横向裂缝 : L<10 / M 10~75 / H>75mm (三级)")
    print(f"\n  {'掩码来源':<22}{'口径':<10}{'JTG轻':>7}{'JTG重':>7}{'JTG重占比':>10}"
          f"{'ASTM_L':>8}{'ASTM_M':>8}{'背离带':>9}")
    srcs = [("GT掩码", None, None)] + [(f"{v}/{c}", v, c) for v, _ in VARIANTS for c in CONDS]
    jtg_share = {}; astm_share = {}
    for label, v, c in srcs:
        if v is None:
            s = [g for _, g in GT.values() if g["ctype"] == "线状"]
            get = lambda r, st: r[st + "_mm"]
        else:
            s = [r for r in lin(sel(variant=v, cond=c)) if r["pred_w_median"] is not None]
            get = lambda r, st: r["pred_" + st]
        if len(s) < 5: continue
        for st in STATS:
            vals = np.array([get(r, st) for r in s], float)
            j = Counter(pc.grade(x, pc.JTG_LIN) for x in vals)
            a = Counter(pc.grade(x, pc.ASTM_LT) for x in vals)
            band = int(np.sum((vals > 3) & (vals < 10)))
            jtg_share[(label, st)] = j["重度"] / len(vals)
            astm_share[(label, st)] = a["L"] / len(vals)
            print(f"  {label:<22}{st:<10}{j['轻度']:>7}{j['重度']:>7}{j['重度']/len(vals):>10.1%}"
                  f"{a['L']:>8}{a['M']:>8}{band:>6}/{len(vals)}")

    if jtg_share:
        jv = list(jtg_share.values()); av = list(astm_share.values())
        print(f"\n  ★ 仅因这些通常被隐含的选择，同一批裂缝的等级分布摆动幅度：")
        print(f"     JTG  重度占比  {min(jv):.1%} ~ {max(jv):.1%}   (极差 {max(jv)-min(jv):.1%})")
        print(f"     ASTM L 级占比  {min(av):.1%} ~ {max(av):.1%}   (极差 {max(av)-min(av):.1%})")
        kj = min(jtg_share, key=jtg_share.get); Kj = max(jtg_share, key=jtg_share.get)
        ka = min(astm_share, key=astm_share.get); Ka = max(astm_share, key=astm_share.get)
        print(f"     JTG  极值组合: {kj} -> {jtg_share[kj]:.1%}   vs   {Kj} -> {jtg_share[Kj]:.1%}")
        print(f"     ASTM 极值组合: {ka} -> {astm_share[ka]:.1%}   vs   {Ka} -> {astm_share[Ka]:.1%}")
        print("     自动化严重度评级的输出不是一个数，而是一个区间。")

    # ---- Dice 与缝宽保真度是否同序 ----
    print("\n" + "-" * 100)
    print("[5b] Dice 排序 vs 缝宽保真度排序（若不同序，说明Dice不是严重度任务的正确选型指标）")
    print(f"  {'掩码来源':<22}{'Dice':>9}{'中位缝宽相对偏差':>18}{'中位缝宽MAE':>13}{'Spearman':>10}")
    tab = []
    for v, _ in VARIANTS:
        for c in CONDS:
            s = [r for r in lin(sel(variant=v, cond=c)) if r["pred_w_median"] is not None]
            if len(s) < 5: continue
            g = np.array([r["gt_w_median"] for r in s]); p = np.array([r["pred_w_median"] for r in s])
            tab.append((f"{v}/{c}", np.mean([r["dice"] for r in s]),
                        (p.mean() - g.mean()) / g.mean(), np.mean(np.abs(p - g)),
                        spearmanr(p, g).correlation))
    for nm, d, rb, mae, rho in sorted(tab, key=lambda x: -x[1]):
        print(f"  {nm:<22}{d:>9.4f}{rb:>+18.1%}{mae:>13.2f}{rho:>10.3f}")
    if len(tab) >= 3:
        byd = [t[0] for t in sorted(tab, key=lambda x: -x[1])]
        bym = [t[0] for t in sorted(tab, key=lambda x: x[3])]
        print(f"\n  按Dice最优排序 : {byd}")
        print(f"  按缝宽MAE最优排序: {bym}")
        print(f"  两序{'一致' if byd == bym else '★不一致★ -> Dice 不能用于严重度任务的模型选型'}")

    # ================= [6] 跨规范背离 =================
    print("\n" + "=" * 100)
    print("[6] 跨规范背离（GT掩码，JTG用中位/ASTM用P95，各自贴合规范文本口径）")
    print("=" * 100)
    s = [g for _, g in GT.values() if g["ctype"] == "线状"]
    if s:
        jm = np.array([g["w_median_mm"] for g in s]); ap_ = np.array([g["w_p95_mm"] for g in s])
        jj = [pc.grade(x, pc.JTG_LIN) for x in jm]; aa = [pc.grade(x, pc.ASTM_LT) for x in ap_]
        print(f"  联合分布: {dict(Counter(f'{x}|{y}' for x, y in zip(jj, aa)))}")
        print(f"  缝宽落在 3~10mm（JTG最高档 而 ASTM最低档）: "
              f"{int(np.sum((jm>3)&(jm<10)))}/{len(jm)}")
        print("\n  【结构性·数据无关】JTG重度起点3mm < ASTM的L级上限10mm。此为规范文本性质。")
        print("  【经验分布·须带警告】本数据集为近景显眼裂缝，过采样宽裂缝，不可外推。")

    # ================= [7] 可靠性 vs 掩码质量（按图聚合） =================
    print("\n" + "=" * 100)
    print("[7] 严重度可靠性 vs 掩码质量（★按图聚合，避免旧版的伪重复★）")
    print("=" * 100)
    for vname, _ in VARIANTS:
        s = [r for r in lin(sel(variant=vname, cond="text_full")) if r["pred_w_p95"] is not None]
        if len(s) < 20: continue
        print(f"  {vname}  n={len(s)}")
        for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, 1.01)]:
            b = [r for r in s if lo <= r["dice"] < hi]
            if len(b) < 5:
                print(f"    Dice[{lo:.1f},{hi:.1f})  n={len(b):<4} 样本不足"); continue
            rho = spearmanr([r["pred_w_p95"] for r in b], [r["gt_w_p95"] for r in b]).correlation
            mae = np.mean([abs(r["pred_w_p95"] - r["gt_w_p95"]) for r in b])
            print(f"    Dice[{lo:.1f},{hi:.1f})  n={len(b):<4} Spearman {rho:>6.3f}  MAE {mae:>5.2f}mm")

    with open(CSVP, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print(f"\n>> 逐图CSV: {CSVP}")
    print("=" * 100)
    print(f"完成，耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()