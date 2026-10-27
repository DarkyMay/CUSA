#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
probe_width_mechanism.py —— zeroshot/text_full 中位缝宽 +42.4% 偏差、roi_crop 排序退化：机制到底是什么？

【背景】closeloop_sutcrack_v2.seg() 已把 logit 双线性插值回原图 (Hd,Wd) 再二值化，
        宽度是在【原图像素】上测的。因此偏差不是"测在重采样像素里"，论文 3.2 节该句需改。
【假说】输出分辨率地板：semantic 头在固定分辨率 R 输出 logit，映射回原图后一个输出格子
        对应 (3024/R) x (4032/R) 原图像素；细于一格的裂缝二值化后至少一格宽 -> 加性下限。
        全图推理每图恒定 -> 保序不保值；roi_crop 随裁剪框长宽比变 -> 排序被抹。
【预测】P1 加性：pred = a + b*gt 截距显著 >0；gt 越细相对偏差越大
        P2 方向：纵向格子(4px) > 横向格子(3px)，横向裂缝(宽度沿y)膨胀 > 纵向裂缝
        P3 解码：先阈值再最近邻 / 双线性 / 双三次 / 不同 logit 阈值 给出不同地板

Part A（无 GPU，~2min）：读 closeloop_v2 与 scale_preserving 的 CSV + GT 掩膜，验 P1/P2 及 roi_crop 因子相关
Part B（GPU，~10-15min）：text_full 推理一次，记录 semantic_seg 原生形状，5 种解码同批 logit，验 P3

用法：python3 probe_width_mechanism.py --skip-gpu          # 先只跑 Part A
      python3 probe_width_mechanism.py                     # A + B 全量
      python3 probe_width_mechanism.py --limit 20          # B 只跑前 20 张冒烟
输出：results/width_mechanism_<ts>.txt  results/width_mechanism_perimage_<ts>.csv
"""
import os, sys, csv, re, glob, time, argparse, datetime
import numpy as np
from PIL import Image, ImageOps
from scipy.stats import spearmanr

sys.path.insert(0, "/root/autodl-tmp/proj_bridge")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pavement_codes as pc

Image.MAX_IMAGE_PIXELS = None
OUT = "/root/autodl-tmp/proj_bridge"
W = "/root/autodl-tmp/sam3_weights"
ROOT = "/root/autodl-tmp/SUT-Crack"
def _resolve(*cands):
    for c in cands:
        if os.path.isdir(c): return c
    return cands[0]
IMGD = _resolve(os.path.join(ROOT, "1-Segmentation", "Original_Image"),
                os.path.join(ROOT, "1-Segmentation", "Original Image"))
GTD = _resolve(os.path.join(ROOT, "1-Segmentation", "Ground_Truth"),
               os.path.join(ROOT, "1-Segmentation", "Ground Truth"))
CL_CSV = os.path.join(OUT, "results", "closeloop_v2_20260726_170420.csv")
SP_CSV = os.path.join(OUT, "results", "scale_preserving_probe.csv")
MMPX = pc.SUTCRACK_MMPX
TEXT = "crack"; DEV = "cuda"
STATS = ["w_median", "w_p95"]

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"width_mechanism_{ts}.txt")
CSVP = os.path.join(OUT, "results", f"width_mechanism_perimage_{ts}.csv")


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()


def fl(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def numkey(s):
    m = re.findall(r'\d+', s); return int(m[0]) if m else 0


def load_csv(p):
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def linfit(x, y, nboot=1000, seed=0):
    """OLS y = a + b x；截距 a 的 bootstrap 95% CI；R2"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    b, a = np.polyfit(x, y, 1)
    yhat = a + b * x
    r2 = 1 - np.sum((y - yhat) ** 2) / max(1e-12, np.sum((y - y.mean()) ** 2))
    rng = np.random.default_rng(seed); A = []
    for _ in range(nboot):
        i = rng.integers(0, len(x), len(x))
        if len(np.unique(x[i])) < 2: continue
        bb, aa = np.polyfit(x[i], y[i], 1); A.append(aa)
    lo, hi = (np.percentile(A, [2.5, 97.5]) if A else (np.nan, np.nan))
    return dict(a=a, b=b, r2=r2, a_lo=lo, a_hi=hi, n=int(len(x)))


def orientation_deg(mask):
    """GT 掩膜主轴与 x 轴的夹角 [0,90]：0=横向裂缝(宽度沿y)，90=纵向裂缝(宽度沿x)"""
    ys, xs = np.nonzero(mask)
    if len(xs) < 10: return np.nan
    if len(xs) > 200000:
        idx = np.random.default_rng(0).choice(len(xs), 200000, replace=False)
        xs, ys = xs[idx], ys[idx]
    c = np.cov(np.vstack([xs - xs.mean(), ys - ys.mean()]))
    val, vec = np.linalg.eigh(c)
    v = vec[:, np.argmax(val)]
    return float(np.degrees(np.arctan2(abs(v[1]), abs(v[0]))))


def dice_np(a, b):
    s = a.sum() + b.sum(); return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def geom(mask):
    if mask.sum() < 30: return None
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - 60), min(mask.shape[0], ys.max() + 61)
    x0, x1 = max(0, xs.min() - 60), min(mask.shape[1], xs.max() + 61)
    return pc.measure(mask[y0:y1, x0:x1], mm_per_px=MMPX)


# =====================================================================================
#                                    PART A
# =====================================================================================
def part_A(orient):
    print("=" * 100)
    print("PART A  现有 CSV 的再分析（zeroshot 主线；标定 %.4f mm/px）" % MMPX)
    print("=" * 100)
    cl = load_csv(CL_CSV)
    lin = [r for r in cl if r["gt_ctype"] == "线状"]
    conds = [("zeroshot", "text_full"), ("zeroshot", "roi_crop"), ("zeroshot", "box_prompt"),
             ("ft_v3", "text_full"), ("ft_v3", "roi_crop"), ("ft_v3", "box_prompt")]

    # ---------- A1: 加性 or 乘性 ----------
    print("\n[A1] P1 加性地板？  pred = a + b*gt  （a 为截距 mm，若显著 >0 且 b≈1 则为加性地板）")
    print("      同时给出乘性拟合 log(pred)=c+d*log(gt) 的 R2 作对照")
    print(f"  {'变体/条件':<22}{'口径':<10}{'n':>4}{'a mm':>8}{'a 95%CI':>18}{'b':>7}{'R2 lin':>8}{'R2 log':>8}"
          f"{'pred>gt':>9}{'min pred':>9}{'min gt':>8}")
    A1 = {}
    for v, c in conds:
        s = [r for r in lin if r["variant"] == v and r["cond"] == c and r["pred_w_median"] not in ("", None)]
        if len(s) < 10: continue
        for st in STATS:
            g = np.array([fl(r["gt_" + st]) for r in s]); p = np.array([fl(r["pred_" + st]) for r in s])
            f = linfit(g, p)
            lf = linfit(np.log(g), np.log(p), nboot=0)
            A1[(v, c, st)] = f
            ci = "[%.2f,%.2f]" % (f["a_lo"], f["a_hi"])
            print(f"  {v+'/'+c:<22}{st:<10}{f['n']:>4}{f['a']:>8.2f}{ci:>18}"
                  f"{f['b']:>7.2f}{f['r2']:>8.3f}{lf['r2']:>8.3f}{np.mean(p > g):>9.1%}{p.min():>9.2f}{g.min():>8.2f}")
    print("\n[A1c] 模糊地板模型  pred^2 = f^2 + k*gt^2  （低分辨率模糊使宽度按平方相加；k≈1 且 f>0 即成立）")
    print(f"  {'变体/条件':<22}{'口径':<10}{'f mm':>8}{'k':>7}{'R2 quad':>9}{'R2 lin':>8}   (f = 等效模糊宽度)")
    for v, c in conds:
        s_ = [r for r in lin if r["variant"] == v and r["cond"] == c and r["pred_w_median"] not in ("", None)]
        if len(s_) < 10: continue
        for st in STATS:
            g = np.array([fl(r["gt_" + st]) for r in s_]); p = np.array([fl(r["pred_" + st]) for r in s_])
            fq = linfit(g ** 2, p ** 2, nboot=0); fl_ = A1.get((v, c, st))
            fmm = np.sqrt(max(fq["a"], 0.0))
            print(f"  {v+'/'+c:<22}{st:<10}{fmm:>8.2f}{fq['b']:>7.2f}{fq['r2']:>9.3f}{(fl_['r2'] if fl_ else np.nan):>8.3f}")
    print("\n  判读：text_full 若 a 的 CI 不含 0 且 b 接近 1，则偏差是加性地板（细裂缝相对偏差更大）；")
    print("        ft_v3 前景比<1（欠分割），预期 b<1、a≈0，是另一种机制。")

    # ---------- A1b: 相对偏差 vs gt 宽度 ----------
    print("\n[A1b] 相对偏差 pred/gt 是否随 gt 变细而增大（加性地板的直接后果）  zeroshot/text_full, w_median")
    s = [r for r in lin if r["variant"] == "zeroshot" and r["cond"] == "text_full" and r["pred_w_median"] not in ("", None)]
    g = np.array([fl(r["gt_w_median"]) for r in s]); p = np.array([fl(r["pred_w_median"]) for r in s])
    ratio = p / g
    q = np.percentile(g, [25, 50, 75])
    bins = [(0, q[0]), (q[0], q[1]), (q[1], q[2]), (q[2], 1e9)]
    print(f"  {'gt中位缝宽 mm':<22}{'n':>4}{'pred/gt 中位':>14}{'pred-gt 中位 mm':>17}")
    for lo, hi in bins:
        m = (g >= lo) & (g < hi)
        print(f"  [{lo:5.2f},{hi if hi<1e8 else np.inf:5.2f})           {m.sum():>4}{np.median(ratio[m]):>14.2f}{np.median((p-g)[m]):>17.2f}")
    print(f"  Spearman(gt, pred/gt) = {spearmanr(g, ratio).correlation:+.3f}   （加性地板 -> 显著负相关）")
    print(f"  Spearman(gt, pred-gt) = {spearmanr(g, p-g).correlation:+.3f}   （纯加性 -> 接近 0）")

    # ---------- A2: 方向依赖 ----------
    print("\n[A2] P2 方向依赖  （GT 主轴角：0°=横向裂缝，宽度沿 y，纵向格子 4px；90°=纵向裂缝，宽度沿 x，横向格子 3px）")
    print("      预测：横向裂缝的 pred-gt 与 pred/gt 都应大于纵向裂缝。 zeroshot/text_full")
    ang = np.array([orient.get(r["stem"], np.nan) for r in s])
    ok = np.isfinite(ang)
    for st in STATS:
        g = np.array([fl(r["gt_" + st]) for r in s]); p = np.array([fl(r["pred_" + st]) for r in s])
        print(f"  口径 {st}")
        print(f"  {'角度区间':<14}{'n':>4}{'pred-gt 中位 mm':>17}{'pred/gt 中位':>14}{'gt 中位 mm':>12}")
        for lo, hi, nm in [(0, 30, "0-30 横向"), (30, 60, "30-60 斜向"), (60, 90.01, "60-90 纵向")]:
            m = ok & (ang >= lo) & (ang < hi)
            if m.sum() == 0: continue
            print(f"  {nm:<14}{m.sum():>4}{np.median((p-g)[m]):>17.2f}{np.median((p/g)[m]):>14.2f}{np.median(g[m]):>12.2f}")
        print(f"    Spearman(角度, pred-gt) = {spearmanr(ang[ok], (p-g)[ok]).correlation:+.3f}   "
              f"Spearman(角度, pred/gt) = {spearmanr(ang[ok], (p/g)[ok]).correlation:+.3f}   （预测为负）")
    print(f"  角度分布：中位 {np.nanmedian(ang):.1f}°，<30°: {np.sum(ang<30)}，30-60°: {np.sum((ang>=30)&(ang<60))}，>=60°: {np.sum(ang>=60)}")

    # ---------- A3: roi_crop 因子相关 ----------
    print("\n[A3] roi_crop 的排序退化是否由逐图重采样因子驱动  （scale_preserving_probe.csv）")
    sp = load_csv(SP_CSV)
    rc = [r for r in sp if r["cond"] == "roi_crop" and r["pred_w_median"] not in ("", None)]
    if rc:
        sx = np.array([fl(r["scale_x"]) for r in rc]); sy = np.array([fl(r["scale_y"]) for r in rc])
        g = np.array([fl(r["gt_w_median"]) for r in rc]); p = np.array([fl(r["pred_w_median"]) for r in rc])
        lr = np.log(p / g)
        # 地板假说：一个输出格子在原图中的边长 ∝ 1/scale；宽度沿裂缝法向，取较小 scale 的方向近似最坏
        smin = np.minimum(sx, sy)
        print(f"  n={len(rc)}  scale_x {sx.mean():.3f}±{sx.std():.3f}  scale_y {sy.mean():.3f}±{sy.std():.3f}")
        print(f"  Spearman(log scale_y , log pred/gt) = {spearmanr(np.log(sy), lr).correlation:+.3f}")
        print(f"  Spearman(log scale_x , log pred/gt) = {spearmanr(np.log(sx), lr).correlation:+.3f}")
        print(f"  Spearman(log min(sx,sy), log pred/gt) = {spearmanr(np.log(smin), lr).correlation:+.3f}   （地板假说 -> 负相关：因子越小格子越大越膨胀）")
        # 二元回归 log(pred/gt) ~ log sx + log sy，看两轴各自的偏效应与解释力
        X = np.column_stack([np.ones_like(sx), np.log(sx), np.log(sy)])
        beta, *_ = np.linalg.lstsq(X, lr, rcond=None)
        resid = lr - X @ beta
        r2 = 1 - resid.var() / lr.var()
        print(f"  二元回归 log(pred/gt) = {beta[0]:+.3f} {beta[1]:+.3f}*log sx {beta[2]:+.3f}*log sy   R2={r2:.3f}")
        # 只用 GT 中位宽 + 两个因子 预测 pred，看能解释多少
        X2 = np.column_stack([np.ones_like(sx), g, np.log(sx), np.log(sy)])
        b2, *_ = np.linalg.lstsq(X2, p, rcond=None)
        r22 = 1 - (p - X2 @ b2).var() / p.var()
        Xg = np.column_stack([np.ones_like(sx), g]); bg, *_ = np.linalg.lstsq(Xg, p, rcond=None)
        r2g = 1 - (p - Xg @ bg).var() / p.var()
        print(f"  pred ~ gt 的 R2 = {r2g:.3f} ；加入 log sx, log sy 后 R2 = {r22:.3f}   （增量 = 因子能解释的部分）")
        # 裁剪框几何：高宽比与 pred/gt
        ar = sy / sx   # = 框宽/框高
        print(f"  Spearman(log 框宽/框高, log pred/gt) = {spearmanr(np.log(ar), lr).correlation:+.3f}")
        # 若按 1/smin 校正后排序是否恢复
        rho_raw = spearmanr(p, g).correlation
        # 用截距模型的粗校正：pred_corr = pred - k*(1/smin) ，k 由回归估计
        f = linfit(1.0 / smin, p - g, nboot=0)
        pc_ = p - (f["a"] + f["b"] / smin)
        rho_c = spearmanr(pc_ + g.mean(), g).correlation
        print(f"  pred 与 gt 的 Spearman：原始 {rho_raw:.3f} -> 回归掉 1/min(scale) 后 {rho_c:.3f}   "
              f"（明显回升 = 排序损失确由逐图因子造成）")
        p_adj = np.exp(np.log(p) - (beta[1] * np.log(sx) + beta[2] * np.log(sy)))
        print(f"  用二元因子模型校正 pred 后 Spearman = {spearmanr(p_adj, g).correlation:.3f}")
        print("  若两种校正都不能把 Spearman 拉回 0.7 量级，则 roi_crop 的排序损失主要不是重采样因子造成，")
        print("  更可能来自裁剪去掉了上下文（裂缝充满画面时模型行为改变）；论文 4.3.2 的机制叙述须相应修改。")
        # 同一批图 text_full 的 Spearman 作参照
        tf = [r for r in sp if r["cond"] == "text_full" and r["pred_w_median"] not in ("", None)]
        if tf:
            g2 = np.array([fl(r["gt_w_median"]) for r in tf]); p2 = np.array([fl(r["pred_w_median"]) for r in tf])
            print(f"  参照 text_full Spearman = {spearmanr(p2, g2).correlation:.3f}")
    return orient


# =====================================================================================
#                                    PART B
# =====================================================================================
def part_B(orient, limit):
    import torch, torch.nn.functional as F
    from transformers import Sam3Model, Sam3Processor
    print("\n" + "=" * 100)
    print("PART B  semantic 头原生输出分辨率 + 同批 logit 的 5 种解码（P3）  zeroshot/text_full")
    print("=" * 100)
    processor = Sam3Processor.from_pretrained(W)
    model = Sam3Model.from_pretrained(W).to(DEV); model.eval(); torch.set_grad_enabled(False)

    # processor 的 resize 设定
    try:
        ip = getattr(processor, "image_processor", None)
        print("  image_processor:", type(ip).__name__ if ip else None)
        for k in ("size", "do_resize", "resample", "do_pad", "pad_size", "do_center_crop", "crop_size"):
            if ip is not None and hasattr(ip, k): print(f"    {k} = {getattr(ip, k)}")
    except Exception as e:
        print("  image_processor 属性读取失败:", e)

    cl = load_csv(CL_CSV)
    gtw = {r["stem"]: r for r in cl if r["variant"] == "zeroshot" and r["cond"] == "text_full" and r["gt_ctype"] == "线状"}
    stems = sorted(gtw.keys(), key=numkey)
    if limit: stems = stems[:limit]
    print(f"  线状图像 {len(stems)} 张")

    DECODES = ["bilinear_t0", "nearest_t0", "bicubic_t0", "bilinear_t-1", "bilinear_t+1"]
    rows = []; shape_printed = False; t0 = time.time()
    for i, stem in enumerate(stems):
        ip_ = os.path.join(IMGD, stem + ".jpg"); gp = os.path.join(GTD, stem + ".png")
        if not (os.path.exists(ip_) and os.path.exists(gp)): continue
        img = ImageOps.exif_transpose(Image.open(ip_)).convert("RGB")
        gt = np.array(Image.open(gp).convert("L")) > 127
        Wd, Hd = img.size
        if (Hd, Wd) != gt.shape: continue
        inp = processor(images=img, text=TEXT, return_tensors="pt")
        if not shape_printed:
            pv = inp.get("pixel_values", None)
            print(f"  pixel_values shape = {tuple(pv.shape) if pv is not None else None}   原图 (H,W)=({Hd},{Wd})")
        inp = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}
        with torch.autocast(DEV, dtype=torch.bfloat16):
            o = model(**inp)
        logit = o["semantic_seg"][0, 0].float()          # (h, w)
        h, w = logit.shape
        if not shape_printed:
            print(f"  semantic_seg 原生形状 = {tuple(o['semantic_seg'].shape)}")
            print(f"  一个输出格子对应原图  x: {Wd/w:.2f} px = {Wd/w*MMPX:.2f} mm ,  y: {Hd/h:.2f} px = {Hd/h*MMPX:.2f} mm")
            print(f"  （若掩膜存在一格宽的地板，细裂缝的 pred 中位宽应聚集在该量级）")
            shape_printed = True
        L = logit[None, None]
        masks = {
            "bilinear_t0": (F.interpolate(L, size=(Hd, Wd), mode="bilinear")[0, 0] > 0).cpu().numpy(),
            "bicubic_t0": (F.interpolate(L, size=(Hd, Wd), mode="bicubic")[0, 0] > 0).cpu().numpy(),
            "nearest_t0": (F.interpolate((L > 0).float(), size=(Hd, Wd), mode="nearest")[0, 0] > 0.5).cpu().numpy(),
            "bilinear_t-1": (F.interpolate(L, size=(Hd, Wd), mode="bilinear")[0, 0] > -1).cpu().numpy(),
            "bilinear_t+1": (F.interpolate(L, size=(Hd, Wd), mode="bilinear")[0, 0] > 1).cpu().numpy(),
        }
        # 原生分辩率下的骨架宽度（以输出格子为单位）：直接看地板
        nat = (logit > 0).cpu().numpy()
        nat_w = None
        if nat.sum() >= 5:
            gn = pc.measure(nat)  # 像素单位=输出格子
            if gn: nat_w = gn["w_median"]
        base = dict(stem=stem, orient_deg=orient.get(stem, np.nan), out_h=h, out_w=w,
                    native_w_median_cells=nat_w,
                    gt_w_median=fl(gtw[stem]["gt_w_median"]), gt_w_p95=fl(gtw[stem]["gt_w_p95"]))
        for d in DECODES:
            m = masks[d]; g_ = geom(m)
            r = dict(base); r["decode"] = d; r["dice"] = dice_np(m, gt)
            r["fg_ratio"] = m.mean() / gt.mean() if gt.mean() > 0 else np.nan
            for st in STATS:
                r["pred_" + st] = g_[st + "_mm"] if g_ else np.nan
            rows.append(r)
        del o, logit, L, masks; torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            print(f"  ...{i+1}/{len(stems)} 已用 {el/60:.1f} min  预计总 {el/(i+1)*len(stems)/60:.0f} min", flush=True)

    if not rows:
        print("!! Part B 无结果"); return
    with open(CSVP, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    print("\n[B1] 原生分辩率下预测掩膜的中位宽（单位=输出格子）")
    nw = np.array([r["native_w_median_cells"] for r in rows if r["decode"] == "bilinear_t0" and r["native_w_median_cells"] is not None], float)
    if nw.size:
        print(f"  n={nw.size}  中位 {np.median(nw):.2f} 格  P10 {np.percentile(nw,10):.2f}  P90 {np.percentile(nw,90):.2f}  最小 {nw.min():.2f}")
        print("  若大多数图在 1~3 格，说明细裂缝在输出分辩率上确实处于一格量级 -> 地板机制成立")

    print("\n[B2] 五种解码的分割与缝宽保真度（同一批 logit，仅后处理不同）")
    print(f"  {'解码':<14}{'n':>4}{'Dice':>8}{'前景比':>8}" + "".join(f"{s+' 偏差%/MAE/ρ':>28}" for s in STATS)
          + f"{'截距a mm':>10}{'斜率b':>7}")
    for d in DECODES:
        s = [r for r in rows if r["decode"] == d and np.isfinite(r["pred_w_median"])]
        if len(s) < 5: continue
        line = f"  {d:<14}{len(s):>4}{np.mean([r['dice'] for r in s]):>8.4f}{np.nanmean([r['fg_ratio'] for r in s]):>8.2f}"
        for st in STATS:
            g = np.array([r["gt_" + st] for r in s]); p = np.array([r["pred_" + st] for r in s])
            line += f"{f'{(p.mean()-g.mean())/g.mean():+.1%} / {np.mean(np.abs(p-g)):.2f} / {spearmanr(p,g).correlation:.3f}':>28}"
        g = np.array([r["gt_w_median"] for r in s]); p = np.array([r["pred_w_median"] for r in s])
        f = linfit(g, p, nboot=0)
        line += f"{f['a']:>10.2f}{f['b']:>7.2f}"
        print(line)
    print("\n  判读：")
    print("   · nearest_t0 与 bilinear_t0 的截距若都 ≈ 一格的 mm 数 -> 地板来自输出分辩率本身，与插值方式无关；")
    print("   · 阈值 t=+1 会收窄掩膜、t=-1 会加宽：若截距随阈值单调变化而斜率≈1，则阈值只是平移地板，不能消除它；")
    print("   · 所有解码 Spearman 相近 -> 排序信息由 logit 决定，后处理无法增减，与论文'排序保真在全图推理下稳定'一致。")
    print(f"\n>> 逐图CSV: {CSVP}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Part B 只跑前 N 张")
    args = ap.parse_args()
    sys.stdout = Tee(LOGP)
    t0 = time.time()
    print(f"probe_width_mechanism.py  {ts}")
    for p in (CL_CSV, SP_CSV, IMGD, GTD):
        print(("  OK " if os.path.exists(p) else "  !! 缺失 ") + p)
    print(f"  GT 掩膜数 {len(glob.glob(os.path.join(GTD, '*.png')))}  原图数 {len(glob.glob(os.path.join(IMGD, '*.jpg')))}")

    # GT 主轴角（Part A2 与 Part B 都用）
    print("\n[0] 计算 GT 掩膜主轴角 ...", flush=True)
    orient = {}
    gps = sorted(glob.glob(os.path.join(GTD, "*.png")), key=lambda p: numkey(os.path.basename(p)))
    for i, gp in enumerate(gps):
        stem = os.path.splitext(os.path.basename(gp))[0]
        try:
            orient[stem] = orientation_deg(np.array(Image.open(gp).convert("L")) > 127)
        except Exception as e:
            print(f"  !! {stem}: {e}")
        if (i + 1) % 40 == 0: print(f"  ...{i+1}/{len(gps)} {time.time()-t0:.0f}s", flush=True)
    print(f"  完成 {len(orient)} 张")

    part_A(orient)
    if not args.skip_gpu:
        part_B(orient, args.limit)
    print(f"\n>> 日志: {LOGP}")
    print(f"总耗时 {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()