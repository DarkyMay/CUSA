#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a_csv_analyses.py —— A 组：纯 CSV / CPU，无 GPU
  A1  R2-5  拒答阈值 c_α：拒答率–准确率–Brier 曲线；按类别流行率重加权；不同阈值 T；给出"按拒答预算选 c_α"的对照表
  A2  R2-9  (b,σ) 跨设备/分辨率移植：YOLO 四尺寸 + SAM3 的截距 a 与输出格子尺寸的关系；
            源分辨率 (b,σ) 直接套用 vs 经粒度模型平移 vs 目标域自标定 的 Brier
  A3  R3-1  不确定度源的量级与交互：log 宽度空间的三类项并列；掩膜残差与归约的交互（median/P95 残差相关性）
  A4  R3-7  数据泄漏：620 训练图 vs SUT / NHA12D / Crack500 / DeepCrack-test / ds400-test 的感知哈希最近邻距离
输出 results/a_csv_<ts>.txt + a_calpha_<ts>.csv + a_transfer_<ts>.csv + a_phash_<ts>.csv
用法 python3 a_csv_analyses.py [--skip_phash]
"""
import os, sys, csv, glob, re, time, argparse, datetime, traceback
import numpy as np
from scipy.stats import norm, spearmanr
from PIL import Image, ImageOps

OUT = "/root/autodl-tmp/proj_bridge"
CLOSELOOP = os.path.join(OUT, "results", "closeloop_v2_20260726_170420.csv")
E6 = os.path.join(OUT, "results", "e6_yolo_perimage_20260906_141239.csv")
DS400 = "/root/autodl-tmp/dataset_400_clean"; DEEPCRACK = "/root/autodl-tmp/DeepCrack-master/dataset"
NHA = "/root/autodl-tmp/NHA12D/NHA12D_dataset"; CRACK500 = "/root/autodl-tmp/CRACK500"
SUT_IMG = "/root/autodl-tmp/SUT-Crack/1-Segmentation/Original_Image"
sys.path.insert(0, OUT)
try:
    import pavement_codes as pc; MMPX = pc.SUTCRACK_MMPX
except Exception:
    MMPX = 0.2219
SEED = 20260728; NSPLIT = 200; NCAL = 42; T_ASTM = 10.0
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"a_csv_{ts}.txt")
CSV_CA = os.path.join(OUT, "results", f"a_calpha_{ts}.csv")
CSV_TR = os.path.join(OUT, "results", f"a_transfer_{ts}.csv")
CSV_PH = os.path.join(OUT, "results", f"a_phash_{ts}.csv")
Image.MAX_IMAGE_PIXELS = None


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()


def load_closeloop():
    rows = {}
    with open(CLOSELOOP, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["variant"] == "zeroshot" and r["cond"] == "text_full" and r["gt_ctype"] == "线状":
                rows[str(r["stem"])] = {k: float(r[k]) for k in ["gt_w_median", "gt_w_p95", "gt_w_mean", "pred_w_median", "pred_w_p95", "pred_w_mean"]}
    stems = sorted(rows, key=int)
    return stems, {k: np.array([rows[s][k] for s in stems]) for k in rows[stems[0]]}


def splits(n, rng):
    for _ in range(NSPLIT):
        perm = rng.permutation(n); yield perm[:NCAL], perm[NCAL:]


def fit_log(wp, wr):
    res = np.log(np.maximum(wp, 1e-3)) - np.log(np.maximum(wr, 1e-3)); return res.mean(), res.std() + 1e-6


def prob(wp, b, s, T):
    return np.clip(norm.cdf((np.log(np.maximum(wp, 1e-3)) - b - np.log(T)) / s), 1e-6, 1 - 1e-6)


# ---------------------------------------------------------------- A1
def stage_a1(D):
    print("=" * 100 + "\n[A1] 拒答阈值 c_α（R2-5）：zs_full, P95, 200 次 42/84 切分\n" + "=" * 100)
    wp, wr = D["pred_w_p95"], D["gt_w_p95"]; n = len(wp)
    calphas = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    out = []

    def run(T, prevalence=None, label=""):
        rng = np.random.RandomState(SEED); y_all = (wr > T).astype(float)
        acc = {c: [] for c in calphas}; ref = {c: [] for c in calphas}; bri = {c: [] for c in calphas}; acc_all = []
        for cal, te in splits(n, rng):
            b, s = fit_log(wp[cal], wr[cal]); P = prob(wp[te], b, s, T); y = y_all[te]
            wgt = np.ones(len(te))
            if prevalence is not None:   # 重要性权重把测试集正类流行率重加权到目标值
                p0 = y.mean(); p0 = min(max(p0, 1e-3), 1 - 1e-3)
                wgt = np.where(y > 0.5, prevalence / p0, (1 - prevalence) / (1 - p0))
            conf = np.maximum(P, 1 - P); correct = ((P > 0.5) == (y > 0.5)).astype(float)
            acc_all.append(np.average(correct, weights=wgt))
            for c in calphas:
                keep = conf >= c
                ref[c].append(1 - np.average(keep.astype(float), weights=wgt))
                acc[c].append(np.average(correct[keep], weights=wgt[keep]) if keep.sum() > 0 else np.nan)
                bri[c].append(np.average((P[keep] - y[keep]) ** 2, weights=wgt[keep]) if keep.sum() > 0 else np.nan)
        print(f"\n  T={T:.0f} mm  流行率={'自然 %.1f%%' % (100*y_all.mean()) if prevalence is None else '%.0f%%' % (100*prevalence)}  全体准确率 {100*np.mean(acc_all):.1f}%")
        print(f"    {'c_α':>5s}{'拒答率%':>9s}{'保留准确率%':>12s}{'保留Brier':>10s}")
        for c in calphas:
            r_, a_, b_ = np.mean(ref[c]), np.nanmean(acc[c]), np.nanmean(bri[c])
            print(f"    {c:5.2f}{100*r_:9.1f}{100*a_:12.1f}{b_:10.3f}")
            out.append(dict(T=T, prevalence=("natural" if prevalence is None else prevalence), c_alpha=c, referral=r_, acc_retained=a_, brier_retained=b_, acc_all=np.mean(acc_all)))

    run(T_ASTM)
    for pv in [0.10, 0.30, 0.50, 0.90]: run(T_ASTM, pv)
    for T in [8.0, 12.0]: run(T)
    # 反查表：给定拒答预算，c_α 取多少
    print("\n  按拒答预算反查 c_α（T=10，自然流行率）：")
    nat = [o for o in out if o["T"] == T_ASTM and o["prevalence"] == "natural"]
    for budget in [0.10, 0.20, 0.30, 0.40]:
        cand = [o for o in nat if o["referral"] <= budget]
        best = max(cand, key=lambda o: o["c_alpha"]) if cand else None
        if best: print(f"    预算 ≤{100*budget:.0f}% → c_α = {best['c_alpha']:.2f}（实际拒答 {100*best['referral']:.1f}%，保留准确率 {100*best['acc_retained']:.1f}%）")
    print("  判读：c_α 不是先验常数，而是由'拒答预算 ↔ 保留准确率'曲线在部署时选定；流行率变化下曲线形状如何变、拒答率是否稳定，看上面各表。")
    with open(CSV_CA, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)


# ---------------------------------------------------------------- A2
def stage_a2(D, stems):
    print("\n" + "=" * 100 + "\n[A2] (b,σ) 跨分辨率/设备移植（R2-9）：截距 a 与输出格子尺寸；源→目标 Brier\n" + "=" * 100)
    e6 = {}
    with open(E6, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r["stem"]) in set(stems):
                e6.setdefault((r["source"], int(r["imgsz"])), {})[str(r["stem"])] = r
    cfgs = sorted(e6.keys())
    print(f"  {'source':12s}{'imgsz':>6s}{'cell_mm':>8s}{'n':>5s}{'a(mm)':>7s}{'slope':>7s}{'rho':>7s}{'a/cell':>7s} | {'log b (med)':>11s}{'σ':>6s}{'b (p95)':>8s}{'σ':>6s}")
    fit = {}
    sam3_cell = np.sqrt(10.5 * 14.0) * MMPX
    for key, est_med, est_p95, cell_mm in [("SAM3 zs_full", D["pred_w_median"], D["pred_w_p95"], sam3_cell)]:
        b_, a_ = np.polyfit(D["gt_w_median"], est_med, 1); rho = spearmanr(D["gt_w_median"], est_med).correlation
        lb, ls = fit_log(est_med, D["gt_w_median"]); lb2, ls2 = fit_log(est_p95, D["gt_w_p95"])
        fit[key] = dict(cell=cell_mm, a=a_, slope=b_, b_med=lb, s_med=ls, b_p95=lb2, s_p95=ls2, wp95=est_p95)
        print(f"  {key:12s}{'1008':>6s}{cell_mm:8.2f}{len(est_med):5d}{a_:7.2f}{b_:7.2f}{rho:7.3f}{a_/cell_mm:7.2f} | {lb:11.3f}{ls:6.3f}{lb2:8.3f}{ls2:6.3f}")
    for src, sz in cfgs:
        d = e6[(src, sz)]; ss = [s for s in stems if s in d]
        gm = np.array([float(d[s]["gt_w_median"]) for s in ss]); pm = np.array([float(d[s]["pred_w_median"]) for s in ss])
        g95 = np.array([float(d[s]["gt_w_p95"]) for s in ss]); p95 = np.array([float(d[s]["pred_w_p95"]) for s in ss])
        ok = pm > 0
        cell_mm = (16128.0 / sz) * MMPX           # retina_masks: 原图上的格子 = 4032/(imgsz/4) px
        b_, a_ = np.polyfit(gm[ok], pm[ok], 1); rho = spearmanr(gm[ok], pm[ok]).correlation
        lb, ls = fit_log(pm[ok], gm[ok]); lb2, ls2 = fit_log(p95[ok], g95[ok])
        fit[(src, sz)] = dict(cell=cell_mm, a=a_, slope=b_, b_med=lb, s_med=ls, b_p95=lb2, s_p95=ls2, wp95=p95, g95=g95, ok=ok, n_empty=int((~ok).sum()))
        print(f"  {src:12s}{sz:6d}{cell_mm:8.2f}{ok.sum():5d}{a_:7.2f}{b_:7.2f}{rho:7.3f}{a_/cell_mm:7.2f} | {lb:11.3f}{ls:6.3f}{lb2:8.3f}{ls2:6.3f}   空掩膜 {int((~ok).sum())}")
    print("  判读：若 a/cell 在同一模型的各尺寸间近似常数，则截距（=偏差的主体）可由输出格子尺寸预测，这是 (b,σ) 中 b 可部分迁移的依据；σ 则看是否随尺寸稳定。")

    # 迁移实验：yolo_rob 各尺寸两两
    print("\n  源→目标 校准迁移（ASTM 10 mm, P95；目标域全部 126 张为测试；'自标定' 为目标域 200 次 42/84 切分的均值）")
    print(f"  {'源':>12s}{'目标':>8s} | {'硬阈值':>7s}{'源(b,σ)直接套用':>16s}{'源(b,σ)+粒度平移a':>18s}{'目标自标定':>10s} | {'b_src':>6s}{'b_tgt':>6s}{'a平移后等效b':>12s}")
    trows = []
    rb_cfgs = [k for k in cfgs if k[0] == "yolo_rob"]
    for src_k in rb_cfgs:
        for tgt_k in rb_cfgs:
            if src_k == tgt_k: continue
            S, Tg = fit[src_k], fit[tgt_k]
            ok = Tg["ok"] & (Tg["wp95"] > 0); wp = Tg["wp95"][ok]; wr = Tg["g95"][ok]; y = (wr > T_ASTM).astype(float)
            brier_pt = np.mean(((wp > T_ASTM).astype(float) - y) ** 2)
            P1 = prob(wp, S["b_p95"], S["s_p95"], T_ASTM); brier_direct = np.mean((P1 - y) ** 2)
            # 粒度平移：按加性截距模型，目标域预测宽度先减去 (a_src·cell_tgt/cell_src − a_src)，再套源 (b,σ)
            shift = S["a"] * (Tg["cell"] / S["cell"] - 1.0)
            wadj = np.maximum(wp - shift, 0.05); P2 = prob(wadj, S["b_p95"], S["s_p95"], T_ASTM); brier_shift = np.mean((P2 - y) ** 2)
            b_eq = np.mean(np.log(np.maximum(wadj, 1e-3)) - np.log(np.maximum(wr, 1e-3)))
            # 目标自标定
            rng = np.random.RandomState(SEED); bs = []
            for cal, te in splits(len(wp), rng):
                b, s = fit_log(wp[cal], wr[cal]); bs.append(np.mean((prob(wp[te], b, s, T_ASTM) - y[te]) ** 2))
            print(f"  {src_k[0]+'@'+str(src_k[1]):>12s}{'@'+str(tgt_k[1]):>8s} | {brier_pt:7.3f}{brier_direct:16.3f}{brier_shift:18.3f}{np.mean(bs):10.3f} | {S['b_p95']:6.3f}{Tg['b_p95']:6.3f}{b_eq:12.3f}")
            trows.append(dict(src=f"{src_k[0]}@{src_k[1]}", tgt=f"{tgt_k[0]}@{tgt_k[1]}", brier_pt=brier_pt, brier_direct=brier_direct, brier_shift=brier_shift, brier_self=np.mean(bs), b_src=S["b_p95"], b_tgt=Tg["b_p95"], b_eq=b_eq))
    # SAM3 → YOLO 跨模型
    S = fit["SAM3 zs_full"]
    for tgt_k in rb_cfgs:
        Tg = fit[tgt_k]; ok = Tg["ok"] & (Tg["wp95"] > 0); wp = Tg["wp95"][ok]; wr = Tg["g95"][ok]; y = (wr > T_ASTM).astype(float)
        P1 = prob(wp, S["b_p95"], S["s_p95"], T_ASTM); shift = S["a"] * (Tg["cell"] / S["cell"] - 1.0)
        wadj = np.maximum(wp - shift, 0.05); P2 = prob(wadj, S["b_p95"], S["s_p95"], T_ASTM)
        print(f"  {'SAM3@1008':>12s}{'yolo_rob@'+str(tgt_k[1]):>8s} | {np.mean(((wp>T_ASTM)-y)**2):7.3f}{np.mean((P1-y)**2):16.3f}{np.mean((P2-y)**2):18.3f}{'—':>10s} | {S['b_p95']:6.3f}{Tg['b_p95']:6.3f}")
    print("  判读：'直接套用' 与 '目标自标定' 的差 = 不移植的代价；'粒度平移' 若明显优于 '直接套用'，说明 b 可经输出分辨率部分迁移；σ 跨模型差异大（YOLO σ≫SAM3）则 σ 不可迁移，需目标域标注。")
    print(f"  标注成本：目标域标定需 {NCAL} 张带参考宽度的图像（论文 Table calib 的标定集规模曲线：5→0.142, 15→0.129, 42→0.125, 60→0.124）。")
    with open(CSV_TR, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trows[0].keys())); w.writeheader(); w.writerows(trows)


# ---------------------------------------------------------------- A3
def stage_a3(D):
    print("\n" + "=" * 100 + "\n[A3] 不确定度源的量级与交互（R3-1 / R3-2）：log 宽度空间，126 线状\n" + "=" * 100)
    lr_med = np.log(D["pred_w_median"] / D["gt_w_median"]); lr_p95 = np.log(D["pred_w_p95"] / D["gt_w_p95"]); lr_mean = np.log(D["pred_w_mean"] / D["gt_w_mean"])
    defi = np.log(D["gt_w_p95"] / D["gt_w_median"]); defi_mean = np.log(D["gt_w_mean"] / D["gt_w_median"])
    phi = np.deg2rad(33.6)
    def kb(t):
        t = np.deg2rad(t); return max(abs(np.log(np.cos(p) ** 2 / np.cos(t + p) ** 2)) for p in (-phi, phi))
    print(f"  {'项':34s}{'中值|·|':>9s}{'IQR':>16s}{'P90':>7s}   说明")
    print(f"  {'定义性 log(P95/median) [GT]':34s}{np.median(defi):9.3f}  [{np.percentile(defi,25):.3f},{np.percentile(defi,75):.3f}]{np.percentile(defi,90):7.3f}   同一掩膜换归约的移动量")
    print(f"  {'定义性 log(mean/median) [GT]':34s}{np.median(np.abs(defi_mean)):9.3f}  [{np.percentile(defi_mean,25):.3f},{np.percentile(defi_mean,75):.3f}]{np.percentile(np.abs(defi_mean),90):7.3f}")
    print(f"  {'估计性 残差 log(pred/gt) median':34s}{np.median(np.abs(lr_med-lr_med.mean())):9.3f}  b={lr_med.mean():.3f} σ={lr_med.std():.3f}")
    print(f"  {'估计性 残差 log(pred/gt) P95':34s}{np.median(np.abs(lr_p95-lr_p95.mean())):9.3f}  b={lr_p95.mean():.3f} σ={lr_p95.std():.3f}")
    print(f"  {'估计性 残差 log(pred/gt) mean':34s}{np.median(np.abs(lr_mean-lr_mean.mean())):9.3f}  b={lr_mean.mean():.3f} σ={lr_mean.std():.3f}")
    for t in [2, 5, 10]:
        print(f"  {'几何性 |log κ| 上界 θ=%d°' % t:34s}{kb(t):9.3f}   画面边缘最坏位置")
    print(f"  {'几何性 畸变 |k1|≤0.05 边缘上界':34s}{np.log(1.17):9.3f}   裂缝像素中值处 {np.log(1.02):.3f}")
    print("\n  交互：")
    print(f"    估计×定义：median 残差与 P95 残差的逐图相关 ρ={spearmanr(lr_med, lr_p95).correlation:.3f}，Pearson r={np.corrcoef(lr_med, lr_p95)[0,1]:.3f}；"
          f"两者之差 log(predP95/predMed) − log(gtP95/gtMed) 的 sd={np.std(lr_p95-lr_med):.3f}")
    print(f"    → 掩膜残差不是对整条剖面的常数乘子：预测剖面比 GT 更平（pred log(P95/med) 中值 {np.median(np.log(D['pred_w_p95']/D['pred_w_median'])):.3f} vs GT {np.median(defi):.3f}），"
          f"所以 (b,σ) 必须按所选归约分别标定（论文 Table calib 已分别给出），这就是 R3-1 所说的'分割不确定度影响归约结果'。")
    print(f"    估计×宽度：残差与 GT 宽度的相关 ρ={spearmanr(D['gt_w_median'], lr_med).correlation:.3f}（负值=窄缝相对偏差更大，即格子效应），"
          f"最窄/最宽四分位的 b: {lr_med[D['gt_w_median']<=np.percentile(D['gt_w_median'],25)].mean():.3f} / {lr_med[D['gt_w_median']>=np.percentile(D['gt_w_median'],75)].mean():.3f}")
    print("    几何×阈值：κ 作用于 mm 值，故越阈概率的区间宽度 ∝ |log κ|/σ；θ=5° 时 |log κ|≤0.128 ≈ 0.5σ，θ=10° 时 0.280 ≈ 1.1σ —— 这是 Fig abstain 中倾角区间与校准区间的相对量级。")
    print("    R3-2：归约的'选择'是定义性项（条件化在协议上，不进入 σ）；归约在有噪掩膜上的'实现'是估计性项（进入该归约下的 (b,σ)）。")


# ---------------------------------------------------------------- A4
def stage_a4():
    print("\n" + "=" * 100 + "\n[A4] 数据泄漏检查（R3-7）：训练 620 张 vs 各测试集，感知哈希 pHash(16×16) 最近邻汉明距离\n" + "=" * 100)
    import imagehash
    def ph(p):
        im = ImageOps.exif_transpose(Image.open(p)).convert("L"); im.thumbnail((512, 512))
        return imagehash.phash(im, hash_size=16)
    train = sorted(glob.glob(os.path.join(DS400, "images/train", "*.jpg"))) + sorted(glob.glob(os.path.join(DEEPCRACK, "train_img", "*.jpg")))
    tests = {"SUT-Crack": sorted(glob.glob(os.path.join(SUT_IMG, "*.jpg"))),
             "NHA12D": sorted(glob.glob(os.path.join(NHA, "*", "Raw_images", "*.jpg"))),
             "Crack500-test(100)": [], "DeepCrack-test": sorted(glob.glob(os.path.join(DEEPCRACK, "test_img", "*.jpg")))[:60],
             "ds400-test": sorted(glob.glob(os.path.join(DS400, "images/test", "*.jpg")))}
    with open(os.path.join(CRACK500, "test.txt")) as f:
        for line in f:
            p = line.split()
            if len(p) >= 1 and os.path.exists(os.path.join(CRACK500, p[0])): tests["Crack500-test(100)"].append(os.path.join(CRACK500, p[0]))
    tests["Crack500-test(100)"] = tests["Crack500-test(100)"][:100]
    t0 = time.time(); th = []
    for p in train:
        try: th.append((p, ph(p)))
        except Exception: pass
    print(f"  训练集哈希 {len(th)} 张 ({time.time()-t0:.0f}s)")
    rows = []
    print(f"  {'测试集':22s}{'n':>5s}{'最小距离':>9s}{'中值':>6s}{'≤4':>4s}{'≤8':>4s}{'≤12':>5s}   最近的一对")
    for name, paths in tests.items():
        dmin = []; pairs = []
        for p in paths:
            try: h = ph(p)
            except Exception: continue
            ds = [(h - th_[1], th_[0]) for th_ in th]; d, q = min(ds, key=lambda x: x[0]); dmin.append(d); pairs.append((d, p, q))
            rows.append(dict(test_set=name, test_img=os.path.basename(p), nn_train_img=os.path.relpath(q, "/root/autodl-tmp"), hamming=d))
        dmin = np.array(dmin); best = min(pairs, key=lambda x: x[0]) if pairs else None
        print(f"  {name:22s}{len(dmin):5d}{dmin.min() if len(dmin) else -1:9d}{np.median(dmin) if len(dmin) else -1:6.0f}{(dmin<=4).sum():4d}{(dmin<=8).sum():4d}{(dmin<=12).sum():5d}   "
              f"{os.path.basename(best[1]) if best else ''} ↔ {os.path.relpath(best[2], '/root/autodl-tmp') if best else ''}")
    # 训练集内部 ds400 vs DeepCrack 作参照：不同来源的随机图像对的典型距离
    rng = np.random.RandomState(0); ref = []
    for _ in range(500):
        i, j = rng.randint(len(th)), rng.randint(len(th))
        if i != j: ref.append(th[i][1] - th[j][1])
    print(f"  参照：训练集内随机图像对的距离 中值 {np.median(ref):.0f}，P1 {np.percentile(ref,1):.0f}（16×16 pHash 共 256 位；≤8 通常视为近重复/同一场景再拍，≤4 视为同一图像）")
    print("  判读：若所有测试集最小距离都远大于 8 且与随机对参照同量级，则不存在训练–测试重叠；ds400-test 行是论文已做过的 in-domain 检查。")
    with open(CSV_PH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip_phash", action="store_true"); args = ap.parse_args()
    sys.stdout = Tee(LOGP); T0 = time.time()
    print(f"a_csv_analyses.py  {ts}  MMPX={MMPX}\nlog -> {LOGP}\n")
    stems, D = load_closeloop(); print(f"线状 {len(stems)} 张\n")
    for fn, a in [(stage_a1, (D,)), (stage_a2, (D, stems)), (stage_a3, (D,))] + ([] if args.skip_phash else [(stage_a4, ())]):
        try: fn(*a)
        except Exception: traceback.print_exc()
    print(f"\nCSV: {CSV_CA}\n     {CSV_TR}\n     {CSV_PH}\n总耗时 {(time.time()-T0)/60:.1f} min")


if __name__ == "__main__":
    main()