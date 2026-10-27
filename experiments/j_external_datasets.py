#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
j_external_datasets.py —— R2-8 / R2-9：宽度分布与采集设备不同的外部数据集

D2  CRKWH100 (100) / CrackLS315 (315)：线扫相机、激光照明、GSD 1 mm/px、512×512、像素级 GT
    - SAM3 零样本 Dice；GT 宽度分布（mm）与 SUT 对比；偏差/ρ/斜率/截距；log 模型 (b,σ)
    - 校准：JTG 3 mm / 5 mm / ASTM 10 mm，P95 与中值归约；200 次切分自标定 vs 硬阈值
    - 跨设备移植：把 SUT 的 (b,σ) 直接套用（P95: 0.175/0.262，median: 0.412/0.277）vs 目标域自标定
D3  UAV-CrackX ×4/×8/×16（同一航次三档放大，512×512）：GT 像素宽随放大的变化；SAM3 截距 a（px）是否随放大不变
    —— 粒度模型的预测：输出格子由图像尺寸决定（512/288=1.78 px），a_px 应近似常数，而物理宽度随 GSD 变化
输出 results/j_ext_<ts>.txt + j_ext_perimage_<ts>.csv
用法 python3 j_external_datasets.py [--limit N]
"""
import os, sys, csv, glob, re, io, time, zipfile, argparse, datetime, traceback
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr, norm

sys.path.insert(0, "/root/autodl-tmp/proj_bridge")
import e1_collapse_mitigation as E
from transformers import Sam3Model, Sam3Processor

ROOT = "/root/autodl-tmp"; EXT = os.path.join(ROOT, "ext"); OUT = E.OUT
ZIP_ONEDRIVE = os.path.join(ROOT, "OneDrive_1_2026-10-20.zip")
ZIP_CRK_GT = os.path.join(ROOT, "CRKWH100_gt.zip"); ZIP_LS_GT = os.path.join(ROOT, "CrackLS315_gt.zip"); ZIP_UAV = os.path.join(ROOT, "UAV-CrackX-Datasets.zip")
GSD_LINESCAN = 1.0            # mm/px（CRKWH100 / CrackLS315 文献给出 1 mm 地面采样距离）
SUT_CAL = {"p95": (0.175, 0.262), "median": (0.412, 0.277)}   # 论文 / a_csv A2 的 zs_full (b,σ)
SEED = 20260728; NSPLIT = 200
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"j_ext_{ts}.txt"); CSVP = os.path.join(OUT, "results", f"j_ext_perimage_{ts}.csv")


# ---------------------------------------------------------------- 解包
def ensure_extracted():
    os.makedirs(EXT, exist_ok=True)
    if not os.path.isdir(os.path.join(EXT, "CRKWH100")) or not os.path.isdir(os.path.join(EXT, "CrackLS315")):
        with zipfile.ZipFile(ZIP_ONEDRIVE) as z:
            z.extractall(EXT)
    if not os.path.isdir(os.path.join(EXT, "CRKWH100_gt")):
        with zipfile.ZipFile(ZIP_CRK_GT) as z: z.extractall(EXT)
    if not os.path.isdir(os.path.join(EXT, "CrackLS315_gt_bmp")):
        with zipfile.ZipFile(ZIP_LS_GT) as z:
            inner = [n for n in z.namelist() if n.endswith(".zip")]
            data = z.read(inner[0])
        with zipfile.ZipFile(io.BytesIO(data)) as z2: z2.extractall(os.path.join(EXT, "CrackLS315_gt_bmp"))
    if not os.path.isdir(os.path.join(EXT, "UAV-CrackX-Datasets-master")):
        with zipfile.ZipFile(ZIP_UAV) as z: z.extractall(EXT)


def find_gt(gtdir, stem):
    for ext in (".bmp", ".png", ".jpg"):
        p = os.path.join(gtdir, stem + ext)
        if os.path.exists(p): return p
    hits = glob.glob(os.path.join(gtdir, "**", stem + ".*"), recursive=True)
    return hits[0] if hits else None


def load_gt(p):
    g = np.array(Image.open(p).convert("L"))
    u = np.unique(g)
    m = g > 127 if u.max() > 1 else g > 0
    if m.mean() > 0.5: m = ~m      # 极性保护：裂缝应是少数像素
    return m


def pairs_linescan(name):
    if name == "CRKWH100":
        imgs = sorted(glob.glob(os.path.join(EXT, "CRKWH100", "*.png"))); gtd = os.path.join(EXT, "CRKWH100_gt")
    else:
        imgs = sorted(glob.glob(os.path.join(EXT, "CrackLS315", "*.jpg"))); gtd = os.path.join(EXT, "CrackLS315_gt_bmp")
    out = []
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]; gp = find_gt(gtd, stem)
        if gp: out.append((ip, gp))
    return out


def pairs_uav(mag):
    base = os.path.join(EXT, "UAV-CrackX-Datasets-master", "UAV-CrackX")
    imgd = os.path.join(base, "leftImg8bit", "train", f"UAV-CrackX{mag}"); gtd = os.path.join(base, "gtFine", "train", f"UAV-CrackX{mag}")
    imgs = {os.path.basename(p): p for p in glob.glob(os.path.join(imgd, "*"))}
    out = []
    for gp in sorted(glob.glob(os.path.join(gtd, "*.png"))):
        stem = os.path.splitext(os.path.basename(gp))[0]            # 图像与 GT 同名（含 _z），只差扩展名
        cands = [k for k in imgs if os.path.splitext(k)[0] == stem]
        if not cands:
            key = re.sub(r"(_z|_gtFine_labelIds|_gtFine_color)$", "", stem)
            cands = [k for k in imgs if os.path.splitext(k)[0] == key or k.startswith(key + ".")]
        if cands: out.append((imgs[sorted(cands, key=len)[0]], gp))
    return out


# ---------------------------------------------------------------- 测量 / 校准
def widths_px(mask):
    r = E.width_profile_px(mask)
    return (0.0, 0.0) if r is None else (r[0], r[2])


def calib(wp, wr, T, ncal, nsim=300):
    rng = np.random.RandomState(SEED); n = len(wp); y_all = (wr > T).astype(float)
    lw, lr = np.log(np.maximum(wp, 1e-3)), np.log(np.maximum(wr, 1e-3))
    acc = dict(brier_pt=[], acc_pt=[], brier=[], acc=[], ece_fail=[], b=[], s=[])
    def ece(p, yy, nb=10):
        e = 0.0; edges = np.linspace(0, 1, nb + 1)
        for i in range(nb):
            mk = (p >= edges[i]) & (p < edges[i + 1] + (1e-9 if i == nb - 1 else 0))
            if mk.any(): e += mk.mean() * abs(p[mk].mean() - yy[mk].mean())
        return e
    for _ in range(NSPLIT):
        perm = rng.permutation(n); cal, te = perm[:ncal], perm[ncal:]
        res = lw[cal] - lr[cal]; b = res.mean(); s = res.std() + 1e-6
        P = np.clip(norm.cdf((lw[te] - b - np.log(T)) / s), 1e-6, 1 - 1e-6); y = y_all[te]
        acc["brier"].append(np.mean((P - y) ** 2)); acc["acc"].append(np.mean((P > 0.5) == (y > 0.5)))
        g = (wp[te] > T).astype(float); acc["brier_pt"].append(np.mean((g - y) ** 2)); acc["acc_pt"].append(np.mean(g == y))
        e_obs = ece(P, y); e_null = np.array([ece(P, (rng.rand(len(P)) < P).astype(float)) for _ in range(nsim)])
        acc["ece_fail"].append(float(e_obs > np.percentile(e_null, 95))); acc["b"].append(b); acc["s"].append(s)
    o = {k: float(np.mean(v)) for k, v in acc.items()}; o["pos"] = float(y_all.mean()); return o


def transfer(wp, wr, T, b, s):
    y = (wr > T).astype(float); P = np.clip(norm.cdf((np.log(np.maximum(wp, 1e-3)) - b - np.log(T)) / s), 1e-6, 1 - 1e-6)
    return float(np.mean((P - y) ** 2)), float(np.mean((P > 0.5) == (y > 0.5)))


def fit_report(name, g, p, unit):
    ok = (p > 0) & (g > 0)
    b_, a_ = np.polyfit(g[ok], p[ok], 1); rho = spearmanr(g[ok], p[ok]).correlation
    lr = np.log(p[ok] / g[ok])
    print(f"    {name:8s} n={ok.sum():3d}  GT中值 {np.median(g[ok]):6.2f}{unit} IQR[{np.percentile(g[ok],25):.2f},{np.percentile(g[ok],75):.2f}]  "
          f"pred中值 {np.median(p[ok]):6.2f}  bias {100*(p[ok].mean()/g[ok].mean()-1):+6.1f}%  ρ {rho:.3f}  slope {b_:.2f} icpt {a_:.2f}{unit}  log b={lr.mean():.3f} σ={lr.std():.3f}  空掩膜 {int((~ok).sum())}")
    return dict(n=int(ok.sum()), rho=rho, slope=b_, icpt=a_, b=lr.mean(), s=lr.std())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); args = ap.parse_args()
    sys.stdout = E.Tee(LOGP); T0 = time.time()
    print(f"j_external_datasets.py  {ts}  {torch.cuda.get_device_name(0)}\nlog -> {LOGP}\n")
    ensure_extracted()
    processor = Sam3Processor.from_pretrained(E.W); model = Sam3Model.from_pretrained(E.W).cuda().eval(); torch.set_grad_enabled(False)

    def seg(img):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return E.sem_mask(model, processor, img)

    rows = []
    def run_set(setname, pairs, gsd):
        pairs = pairs[:args.limit] if args.limit else pairs
        recs = []
        for ip, gp in pairs:
            try:
                img = Image.open(ip).convert("RGB"); gt = load_gt(gp)
                if (img.size[1], img.size[0]) != gt.shape: img = img.resize((gt.shape[1], gt.shape[0]), Image.BILINEAR)
                pm = seg(img); gm, g95 = widths_px(gt); pmed, p95 = widths_px(pm)
                r = dict(set=setname, file=os.path.basename(ip), H=gt.shape[0], W=gt.shape[1], dice=E.dice_np(pm, gt), gt_fg=float(gt.mean()), pred_fg=float(pm.mean()),
                         gt_med_px=gm, gt_p95_px=g95, pred_med_px=pmed, pred_p95_px=p95, gsd=gsd)
                recs.append(r); rows.append(r)
            except Exception:
                torch.cuda.empty_cache(); traceback.print_exc()
        return recs

    # ---------------- D2
    print("=" * 100 + "\n[D2] 线扫激光数据集：CRKWH100 / CrackLS315（GSD 1 mm/px；SUT 0.2219 mm/px，格子 10.5×14 px；此处格子 512/288 = 1.78 px = 1.78 mm）\n" + "=" * 100)
    for setname in ["CRKWH100", "CrackLS315"]:
        pairs = pairs_linescan(setname); print(f"\n  {setname}: 配对 {len(pairs)} 张", flush=True)
        recs = run_set(setname, pairs, GSD_LINESCAN)
        if not recs: continue
        d = np.array([r["dice"] for r in recs]); print(f"  Dice 均值 {d.mean():.3f}（中位 {np.median(d):.3f}；SUT 零样本 0.759）  GT 前景比中值 {np.median([r['gt_fg'] for r in recs]):.4f}")
        gm = np.array([r["gt_med_px"] for r in recs]) * GSD_LINESCAN; pm = np.array([r["pred_med_px"] for r in recs]) * GSD_LINESCAN
        g95 = np.array([r["gt_p95_px"] for r in recs]) * GSD_LINESCAN; p95 = np.array([r["pred_p95_px"] for r in recs]) * GSD_LINESCAN
        print("  宽度（mm；SUT 对照：GT 中值 6.63 mm，zs 偏差 +42.4%，ρ 0.758，a 4.61，log b 0.412 σ 0.283）")
        fm = fit_report("median", gm, pm, "mm"); f9 = fit_report("p95", g95, p95, "mm")
        print(f"    截距/格子: median {fm['icpt']/1.78:.2f}  p95 {f9['icpt']/1.78:.2f}（SUT zs_full 1.73）")
        print(f"\n  校准（{NSPLIT} 次切分；标定集 42 或 n/3）：")
        print(f"    {'归约':7s}{'T':>5s}{'pos%':>6s} | {'Brier_pt':>9s}{'acc_pt':>7s} | {'自标定Brier':>11s}{'acc':>6s}{'ECEf%':>6s}{'b':>7s}{'σ':>6s} | {'套用SUT(b,σ) Brier':>18s}{'acc':>6s}")
        for red, wp, wr in [("p95", p95, g95), ("median", pm, gm)]:
            ok = (wp > 0) & (wr > 0); wp_, wr_ = wp[ok], wr[ok]
            for T in [3.0, 5.0, 10.0]:
                pos = (wr_ > T).mean()
                if pos < 0.05 or pos > 0.95:
                    print(f"    {red:7s}{T:5.1f}{100*pos:6.1f} | 正样本率退化，不可校准"); continue
                ncal = 42 if len(wp_) >= 84 else max(10, len(wp_) // 3)
                o = calib(wp_, wr_, T, ncal); tb, ta = transfer(wp_, wr_, T, *SUT_CAL[red])
                print(f"    {red:7s}{T:5.1f}{100*pos:6.1f} | {o['brier_pt']:9.3f}{100*o['acc_pt']:7.1f} | {o['brier']:11.3f}{100*o['acc']:6.1f}{100*o['ece_fail']:6.1f}{o['b']:7.3f}{o['s']:6.3f} | {tb:18.3f}{100*ta:6.1f}")
        print("  判读：'套用 SUT(b,σ)' 与 '自标定' 的差 = 跨设备不重标定的代价；ECEf% 高说明 log 模型在该设备上不成立，应触发报警。")

    # ---------------- D3
    print("\n" + "=" * 100 + "\n[D3] UAV-CrackX ×4/×8/×16：同一航次三档放大（GSD 比 1 : 1/2 : 1/4），像素宽度与 SAM3 截距随放大的变化\n" + "=" * 100)
    summ = {}
    for mag in [4, 8, 16]:
        pairs = pairs_uav(mag); print(f"\n  ×{mag}: 配对 {len(pairs)} 张", flush=True)
        recs = run_set(f"UAV-CrackX{mag}", pairs, np.nan)
        if not recs: continue
        d = np.array([r["dice"] for r in recs]); gm = np.array([r["gt_med_px"] for r in recs]); pm = np.array([r["pred_med_px"] for r in recs])
        print(f"  Dice 均值 {d.mean():.3f}  GT 像素宽中值 {np.median(gm[gm>0]):.2f} px（格子 {np.median([r['H'] for r in recs])/288:.2f} px）  GT 宽/格子 {np.median(gm[gm>0])/(np.median([r['H'] for r in recs])/288):.2f}")
        summ[mag] = fit_report("median", gm, pm, "px")
    if len(summ) == 3:
        print("\n  粒度模型检验：")
        print(f"    截距 a_px: ×4 {summ[4]['icpt']:.2f}  ×8 {summ[8]['icpt']:.2f}  ×16 {summ[16]['icpt']:.2f}  "
              f"log b: {summ[4]['b']:.2f} / {summ[8]['b']:.2f} / {summ[16]['b']:.2f}   ρ: {summ[4]['rho']:.2f} / {summ[8]['rho']:.2f} / {summ[16]['rho']:.2f}")
        print("    若 a_px 近似不变而 GT 像素宽随放大翻倍，则偏差由输出格子决定而非物理宽度——与论文机制一致；相对偏差 b 应随放大（像素宽变大）单调下降。")

    if rows:
        with open(CSVP, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\n逐图 CSV {CSVP}\n总耗时 {(time.time()-T0)/60:.1f} min")


if __name__ == "__main__":
    main()