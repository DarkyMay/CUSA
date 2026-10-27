#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
b_estimators_gating.py —— B 组：纯 CPU，只用已落盘掩膜，回应
  R2-2 / R2-7 / R3-4  宽度估计器敏感性（EDT 内切 vs 垂直边界距 vs 面积/长度 vs 亚像素 logit 零交叉）
  R2-4 / R3-6         线状/网状门控：GT 掩膜 vs 专家标签、预测掩膜一致性、降采样 1/2 1/4 鲁棒性、因子扫描、失败分析
  R2-3                其他操作化（ASTM max / 均值 / 中值）下的校准与准确率
  R2-8                ASTM 阈值 6–14 mm 扫描下的 Brier / ECE

阶段：
  B0  自检：本脚本 EDT 中值宽 vs closeloop CSV 的 pred_w_median / gt_w_median（应一致）
  B1  七个掩膜源 + GT，四种估计器 × 四种归约，逐图宽度 -> widths CSV；汇总 bias/MAE/ρ（参考 = 同一估计器用于 GT）
  B2  门控
  B3  校准：估计器 × 归约 × 阈值，200 次 42/84 切分（seed 20260728），log 模型 vs 硬阈值；含拒答 30%
输出： results/b_estgate_<ts>.txt + b_widths_<ts>.csv + b_gate_<ts>.csv + b_calib_<ts>.csv
用法： python3 b_estimators_gating.py [--limit N] [--workers 4]
"""
import os, sys, glob, re, csv, time, argparse, datetime, traceback
from multiprocessing import Pool
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from scipy.stats import spearmanr, norm
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

Image.MAX_IMAGE_PIXELS = None
OUT = "/root/autodl-tmp/proj_bridge"
MASKD = os.path.join(OUT, "results", "masks")
SUT_GT = "/root/autodl-tmp/SUT-Crack/1-Segmentation/Ground_Truth"
CLOSELOOP = os.path.join(OUT, "results", "closeloop_v2_20260726_170420.csv")
sys.path.insert(0, OUT)
try:
    import pavement_codes as pc; MMPX = pc.SUTCRACK_MMPX
except Exception:
    MMPX = 0.2219
VARIANTS = ["zs_full", "zs_crop", "zs_box", "ft_full", "yolo_rob_640", "yolo_rob_2016", "yolo_cv12k_640"]
REDUCTIONS = ["median", "mean", "p95", "max"]
ESTIMATORS = ["edt", "perp", "area_len", "subpix"]   # subpix 仅 zs_full（需要 logit）
T_ASTM = 10.0; T_JTG = 3.0
SEED = 20260728; NSPLIT = 200; NCAL = 42
GATE_FACTOR = 14.0
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"b_estgate_{ts}.txt")
CSV_W = os.path.join(OUT, "results", f"b_widths_{ts}.csv")
CSV_G = os.path.join(OUT, "results", f"b_gate_{ts}.csv")
CSV_C = os.path.join(OUT, "results", f"b_calib_{ts}.csv")


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()


def numkey(p):
    m = re.findall(r'\d+', os.path.basename(p)); return int(m[0]) if m else 0


def load_mask(p):
    return np.array(Image.open(p).convert("L")) > 127


def crop_bbox(mask, pad=60):
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    return mask[y0:y1, x0:x1], (y0, x0)


def reduce_all(w):
    w = np.asarray(w, float)
    if w.size == 0: return dict(median=np.nan, mean=np.nan, p95=np.nan, max=np.nan)
    return dict(median=float(np.median(w)), mean=float(np.mean(w)), p95=float(np.percentile(w, 95)), max=float(np.max(w)))


# ---------------------------------------------------------------- 估计器
def skeleton_and_edt(m):
    sk = skeletonize(m); d = ndi.distance_transform_edt(m)
    return sk, d


def est_edt(m, sk, d):
    """论文估计器：骨架点处 2*EDT（内切圆直径）"""
    return 2.0 * d[sk]


def skeleton_degree(sk):
    """8 邻域内骨架邻居数（不含自身）"""
    k = np.ones((3, 3), int); k[1, 1] = 0
    return ndi.convolve(sk.astype(int), k, mode="constant") * sk


PERP_MAXRATIO = 2.5   # 弦长 > 2.5 × 内切直径 视为切向估计失败（直带上对应角度误差 > 66°）


def est_perp(m, sk, d, logit=None, lo_shape=None, offset=(0, 0), full_shape=None, step=0.5, maxlen=400.0, sub=4, r_tan=10.0, return_stats=False):
    """垂直边界距：骨架点局部切向(PCA 半径 r_tan)的法线上双向射线到掩膜边界的距离之和。
    排除：骨架度≠2 的点（端点/分叉，切向无定义）；弦长 > PERP_MAXRATIO × 2·EDT 的点（切向失败，射线顺裂缝走）。
    logit 不为 None 时改为亚像素：沿法线找 logit 零交叉（线性插值）。
    sub: 骨架点抽样步长（每 sub 个点取 1 个），加速。"""
    pts = np.argwhere(sk)
    if len(pts) < 5: return (np.array([]), dict(n=0, n_deg=0, n_ratio=0)) if return_stats else np.array([])
    deg = skeleton_degree(sk)
    tree = cKDTree(pts)
    sel = pts[::sub] if sub > 1 else pts
    ok_deg = deg[sel[:, 0], sel[:, 1]] == 2
    n_deg = int((~ok_deg).sum()); sel = sel[ok_deg]
    if len(sel) < 3: return (np.array([]), dict(n=0, n_deg=n_deg, n_ratio=0)) if return_stats else np.array([])
    edt2 = 2.0 * d[sel[:, 0], sel[:, 1]]
    nbrs = tree.query_ball_point(sel, r_tan)
    normals = np.zeros((len(sel), 2))
    for i, nb in enumerate(nbrs):
        q = pts[nb] - sel[i]
        if len(nb) < 3: normals[i] = (1.0, 0.0); continue
        c = q.T @ q
        evals, evecs = np.linalg.eigh(c)
        t = evecs[:, 1]; normals[i] = (-t[1], t[0])
    steps = np.arange(0.0, maxlen + step, step)
    H, W = m.shape
    widths = np.zeros(len(sel))
    if logit is None:
        for sign in (1, -1):
            yy = sel[:, 0:1] + sign * normals[:, 0:1] * steps[None]
            xx = sel[:, 1:2] + sign * normals[:, 1:2] * steps[None]
            yi = np.clip(np.rint(yy).astype(int), 0, H - 1); xi = np.clip(np.rint(xx).astype(int), 0, W - 1)
            inside = m[yi, xi] & (yy >= -0.5) & (yy < H - 0.5) & (xx >= -0.5) & (xx < W - 0.5)
            first_out = np.argmin(inside, axis=1)                       # 第一个 False 的位置
            none_out = inside.all(axis=1); first_out[none_out] = len(steps) - 1
            widths += steps[first_out] - step / 2.0                     # 边界在最后一个 inside 与第一个 outside 之间
        keep = widths <= PERP_MAXRATIO * np.maximum(edt2, 1.0)
        out = widths[keep]
        return (out, dict(n=int(len(out)), n_deg=n_deg, n_ratio=int((~keep).sum()))) if return_stats else out
    # 亚像素：沿法线采样 logit（288 格 bilinear 场，align_corners=False 的坐标映射），找零交叉
    Hl, Wl = lo_shape; Hf, Wf = full_shape
    sy, sx = Hl / Hf, Wl / Wf
    for sign in (1, -1):
        yy = sel[:, 0:1] + offset[0] + sign * normals[:, 0:1] * steps[None]
        xx = sel[:, 1:2] + offset[1] + sign * normals[:, 1:2] * steps[None]
        ly = np.clip((yy + 0.5) * sy - 0.5, 0, Hl - 1); lx = np.clip((xx + 0.5) * sx - 0.5, 0, Wl - 1)
        f = ndi.map_coordinates(logit, [ly.ravel(), lx.ravel()], order=1, mode="nearest").reshape(ly.shape)
        oob = (yy < 0) | (yy >= Hf) | (xx < 0) | (xx >= Wf); f[oob] = -1.0
        neg = f < 0
        first = np.argmax(neg, axis=1); none = ~neg.any(axis=1); first[none] = len(steps) - 1
        idx = np.arange(len(sel))
        f1 = f[idx, first]; f0 = f[idx, np.maximum(first - 1, 0)]
        frac = np.where((f0 - f1) > 1e-9, f0 / (f0 - f1), 0.5)
        t0 = steps[np.maximum(first - 1, 0)]
        widths += np.where(first > 0, t0 + frac * step, 0.0)
    keep = widths <= PERP_MAXRATIO * np.maximum(edt2, 1.0)
    out = widths[keep]
    return (out, dict(n=int(len(out)), n_deg=n_deg, n_ratio=int((~keep).sum()))) if return_stats else out


def est_area_len(m, sk):
    """面积 / 骨架长度（像素计数）"""
    n = sk.sum(); return np.array([m.sum() / n]) if n > 0 else np.array([])


# ---------------------------------------------------------------- 门控
def gate(mask, factor=GATE_FACTOR, min_cells=1, w_override=None):
    """论文 §3.4：封闭区域数由欧拉示性数给出，面积 > (factor * 中值局部宽)^2 者计为一个 cell。
    w_override: 用指定宽度(px)代替掩膜自身的中值宽来算 A_min（用于检验偏差校正）。
    返回 (is_network, n_cells, n_holes_total, median_w_px, top3_sqrt_area_over_w)"""
    if mask.sum() < 30: return False, 0, 0, 0.0, ""
    m, _ = crop_bbox(mask, 5)
    sk, d = skeleton_and_edt(m)
    w = 2.0 * d[sk]
    wm = float(np.median(w)) if w.size else 0.0
    wa = w_override if w_override else wm
    holes = ndi.binary_fill_holes(m) & ~m
    lab, nh = ndi.label(holes)
    if nh == 0: return False, 0, 0, wm, ""
    areas = np.sort(ndi.sum(holes, lab, index=np.arange(1, nh + 1)))[::-1]
    amin = (factor * max(wa, 1.0)) ** 2
    nc = int((areas > amin).sum())
    top = ";".join(f"{np.sqrt(a)/max(wm,1.0):.1f}" for a in areas[:3])   # 洞的边长 / 宽度，直接与 factor 比
    return nc >= min_cells, nc, int(nh), wm, top


def downsample(mask, k):
    H, W = mask.shape
    return np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((W // k, H // k), Image.BOX)) > 127


# ---------------------------------------------------------------- 每图任务
def work_image(args):
    stem, gtp, variants, logit_dir = args
    rows = []; grows = []
    try:
        gt = load_mask(gtp)
    except Exception as e:
        return [], [], f"{stem}: GT 读取失败 {e}"
    srcs = [("gt", gt)]
    for v in variants:
        p = os.path.join(MASKD, v, stem + ".png")
        if os.path.exists(p): srcs.append((v, load_mask(p)))
    logit = None
    lp = os.path.join(logit_dir, stem + ".npy")
    if os.path.exists(lp): logit = np.load(lp).astype(np.float32)
    gt_w = None
    for name, mask in srcs:
        if mask.sum() < 30:
            for est in (ESTIMATORS if name == "zs_full" else ESTIMATORS[:3]):
                rows.append(dict(stem=stem, source=name, estimator=est, n_pts=0, n_excl_deg=0, n_excl_ratio=0, **{k: 0.0 for k in REDUCTIONS}))
            grows.append(dict(stem=stem, source=name, scale=1, factor=GATE_FACTOR, variant="plain", is_network=0, n_cells=0, n_holes=0, w_med_px=0.0, top3=""))
            continue
        m, off = crop_bbox(mask)
        sk, d = skeleton_and_edt(m)
        w_edt = est_edt(m, sk, d)
        w_perp, st = est_perp(m, sk, d, return_stats=True)
        w_al = est_area_len(m, sk)
        for est, w, ex in [("edt", w_edt, (0, 0)), ("perp", w_perp, (st["n_deg"], st["n_ratio"])), ("area_len", w_al, (0, 0))]:
            rows.append(dict(stem=stem, source=name, estimator=est, n_pts=int(len(w)), n_excl_deg=ex[0], n_excl_ratio=ex[1], **{k: v * MMPX for k, v in reduce_all(w).items()}))
        if name == "zs_full" and logit is not None:
            w_sp, st2 = est_perp(m, sk, d, logit=logit, lo_shape=logit.shape, offset=off, full_shape=mask.shape, return_stats=True)
            rows.append(dict(stem=stem, source=name, estimator="subpix", n_pts=int(len(w_sp)), n_excl_deg=st2["n_deg"], n_excl_ratio=st2["n_ratio"], **{k: v * MMPX for k, v in reduce_all(w_sp).items()}))
        # 门控：原分辨率 + 因子扫描（仅 gt / zs_full）+ 降采样 + 偏差校正变体（预测掩膜用 w/1.42 算 A_min）
        isn, nc, nh, wm, top = gate(mask)
        if name == "gt": gt_w = wm
        grows.append(dict(stem=stem, source=name, scale=1, factor=GATE_FACTOR, variant="plain", is_network=int(isn), n_cells=nc, n_holes=nh, w_med_px=wm, top3=top))
        if name != "gt":
            isn4, nc4, nh4, _, top4 = gate(mask, w_override=wm / 1.424)      # 论文 zs_full 中值宽偏差 +42.4% 的全局校正
            grows.append(dict(stem=stem, source=name, scale=1, factor=GATE_FACTOR, variant="biascorr", is_network=int(isn4), n_cells=nc4, n_holes=nh4, w_med_px=wm, top3=top4))
        if name in ("gt", "zs_full"):
            for fac in [6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0]:
                isn2, nc2, nh2, _, top2 = gate(mask, factor=fac)
                grows.append(dict(stem=stem, source=name, scale=1, factor=fac, variant="plain", is_network=int(isn2), n_cells=nc2, n_holes=nh2, w_med_px=wm, top3=top2))
            for k in (2, 4):
                mk = downsample(mask, k); isn3, nc3, nh3, wm3, top3 = gate(mk)
                grows.append(dict(stem=stem, source=name, scale=k, factor=GATE_FACTOR, variant="plain", is_network=int(isn3), n_cells=nc3, n_holes=nh3, w_med_px=wm3, top3=top3))
    return rows, grows, None


# ---------------------------------------------------------------- 校准
def calib_eval(w_pred, w_ref, T, nsplit=NSPLIT, ncal=NCAL, seed=SEED, nsim=400, refer=0.30):
    """log 模型 vs 点估计；返回各指标在 nsplit 次切分上的均值。w_pred/w_ref: mm, 线状样本。"""
    rng = np.random.RandomState(seed)
    n = len(w_pred); y_all = (w_ref > T).astype(float)
    lw = np.log(np.maximum(w_pred, 1e-3)); lr = np.log(np.maximum(w_ref, 1e-3))
    acc = dict(brier_pt=[], acc_pt=[], ref_pt=[], brier=[], nll=[], acc_log=[], ref_log=[], ece_fail=[], b=[], s=[])
    for _ in range(nsplit):
        perm = rng.permutation(n); cal, te = perm[:ncal], perm[ncal:]
        res = lw[cal] - lr[cal]; b = res.mean(); s = res.std(ddof=0) + 1e-6
        P = norm.cdf((lw[te] - b - np.log(T)) / s); y = y_all[te]
        P = np.clip(P, 1e-6, 1 - 1e-6)
        acc["brier"].append(np.mean((P - y) ** 2)); acc["nll"].append(-np.mean(y * np.log(P) + (1 - y) * np.log(1 - P)))
        acc["acc_log"].append(np.mean((P > 0.5) == (y > 0.5)))
        keep = np.argsort(-np.abs(P - 0.5))[: int(round(len(te) * (1 - refer)))]
        acc["ref_log"].append(np.mean((P[keep] > 0.5) == (y[keep] > 0.5)))
        # 点估计
        g = (w_pred[te] > T).astype(float)
        acc["brier_pt"].append(np.mean((g - y) ** 2)); acc["acc_pt"].append(np.mean(g == y))
        keep2 = np.argsort(-np.abs(w_pred[te] - T))[: int(round(len(te) * (1 - refer)))]
        acc["ref_pt"].append(np.mean(g[keep2] == y[keep2]))
        # ECE 与模拟零分布
        def ece(p, yy, nb=10):
            e = 0.0; edges = np.linspace(0, 1, nb + 1)
            for i in range(nb):
                mk = (p >= edges[i]) & (p < edges[i + 1] + (1e-9 if i == nb - 1 else 0))
                if mk.any(): e += mk.mean() * abs(p[mk].mean() - yy[mk].mean())
            return e
        e_obs = ece(P, y)
        e_null = np.array([ece(P, (rng.rand(len(P)) < P).astype(float)) for _ in range(nsim)])
        acc["ece_fail"].append(float(e_obs > np.percentile(e_null, 95)))
        acc["b"].append(b); acc["s"].append(s)
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    out["b_sd"] = float(np.std(acc["b"])); out["s_sd"] = float(np.std(acc["s"]))
    out["pos_rate"] = float(y_all.mean()); out["n"] = n
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    sys.stdout = Tee(LOGP); T0 = time.time()
    print(f"b_estimators_gating.py  {ts}  MMPX={MMPX}  workers={args.workers}\nlog -> {LOGP}\n")

    # closeloop 参考
    cl = {}; linear = set(); ctype = {}
    with open(CLOSELOOP, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["variant"] == "zeroshot" and r["cond"] == "text_full":
                s = str(r["stem"]); cl[s] = r; ctype[s] = r["gt_ctype"]
                if r["gt_ctype"] == "线状": linear.add(s)
    stems = sorted(cl.keys(), key=int)
    if args.limit: stems = stems[:args.limit]
    present = [v for v in VARIANTS if os.path.isdir(os.path.join(MASKD, v))]
    print(f"图像 {len(stems)} 张（线状 {len([s for s in stems if s in linear])}）；掩膜源: {present}")
    missing = [v for v in VARIANTS if v not in present]
    if missing: print(f"  !! 缺掩膜目录: {missing}（YOLO 掩膜由 e1 脚本 S0 阶段生成）")

    # B1 逐图
    print("\n" + "=" * 100 + "\n[B1] 逐图宽度：估计器 × 归约（CPU 并行）\n" + "=" * 100, flush=True)
    tasks = [(s, os.path.join(SUT_GT, s + ".png"), present, os.path.join(MASKD, "zs_full_logit")) for s in stems]
    rows = []; grows = []; t1 = time.time()
    with Pool(args.workers) as pool:
        for i, (r, g, err) in enumerate(pool.imap_unordered(work_image, tasks)):
            if err: print("  " + err)
            rows += r; grows += g
            if (i + 1) % 20 == 0: print(f"  {i+1}/{len(tasks)}  {time.time()-t1:.0f}s", flush=True)
    with open(CSV_W, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(CSV_G, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grows[0].keys())); w.writeheader(); w.writerows(grows)
    print(f"  宽度 CSV {CSV_W}\n  门控 CSV {CSV_G}  ({time.time()-t1:.0f}s)")

    def get(source, est, red, only_linear=True):
        d = {r["stem"]: r[red] for r in rows if r["source"] == source and r["estimator"] == est}
        ss = [s for s in stems if s in d and (s in linear or not only_linear)]
        return ss, np.array([d[s] for s in ss])

    # B0 自检
    print("\n" + "=" * 100 + "\n[B0] 自检：本脚本 EDT 中值 vs closeloop CSV\n" + "=" * 100)
    for src, col in [("gt", "gt_w_median"), ("zs_full", "pred_w_median")]:
        ss, mine = get(src, "edt", "median")
        ref = np.array([float(cl[s][col]) for s in ss])
        print(f"  {src:8s} n={len(ss)}  中值差 {np.median(mine-ref):+.3f} mm  最大|差| {np.max(np.abs(mine-ref)):.3f} mm  ρ={spearmanr(mine, ref).correlation:.4f}")
    ss, g_p95 = get("gt", "edt", "p95"); ref = np.array([float(cl[s]["gt_w_p95"]) for s in ss])
    print(f"  gt p95   最大|差| {np.max(np.abs(g_p95-ref)):.3f} mm")

    # B1 汇总
    print("\n" + "=" * 100 + "\n[B1] 估计器 × 掩膜源：偏差/MAE/ρ（126 线状；参考 = 同一估计器用于 GT 掩膜；subpix 的参考 = perp 用于 GT）\n" + "=" * 100)
    print(f"  {'source':16s}{'estimator':10s}{'red':8s}{'bias%':>8s}{'MAE':>7s}{'rho':>7s}{'slope':>7s}{'icpt':>7s}   |  ref median(mm)  pred median(mm)")
    summary = {}
    for red in ["median", "p95"]:
        for est in ESTIMATORS:
            ref_est = "perp" if est == "subpix" else est
            ssr, wr = get("gt", ref_est, red)
            for src in present:
                ssp, wp = get(src, est, red)
                if len(ssp) == 0: continue
                common = [s for s in ssr if s in set(ssp)]
                a = np.array([wr[ssr.index(s)] for s in common]); b_ = np.array([wp[ssp.index(s)] for s in common])
                ok = a > 0
                bias = 100 * (np.mean(b_[ok]) / np.mean(a[ok]) - 1); mae = np.mean(np.abs(b_ - a)); rho = spearmanr(a, b_).correlation
                sl, ic = np.polyfit(a[ok], b_[ok], 1)
                summary[(src, est, red)] = dict(bias=bias, mae=mae, rho=rho, slope=sl, icpt=ic, n=len(common))
                print(f"  {src:16s}{est:10s}{red:8s}{bias:+8.1f}{mae:7.2f}{rho:7.3f}{sl:7.2f}{ic:7.2f}   |  {np.median(a):6.2f}  {np.median(b_):6.2f}")
    # 估计器之间在 GT 上的一致性
    print("\n  估计器之间在 GT 掩膜上的一致性（中值归约，126 线状）:")
    ssA, eA = get("gt", "edt", "median"); _, pA = get("gt", "perp", "median"); _, aA = get("gt", "area_len", "median")
    print(f"    perp/edt 比值中值 {np.median(pA/eA):.3f}  ρ={spearmanr(pA, eA).correlation:.3f}   area_len/edt(mean) 比值中值 {np.median(aA/get('gt','edt','mean')[1]):.3f}")

    # B2 门控
    print("\n" + "=" * 100 + "\n[B2] 门控：GT 掩膜 vs 专家标签（gt_ctype），预测掩膜一致性，降采样，因子扫描\n" + "=" * 100)
    gt_net = {s for s in stems if ctype.get(s) == "网状"}
    print(f"  专家标注网状: {sorted(gt_net, key=int)}  （n={len(gt_net)} / {len(stems)}）")
    def confusion(src, scale=1, factor=GATE_FACTOR, variant="plain"):
        d = {g["stem"]: g for g in grows if g["source"] == src and g["scale"] == scale and abs(g["factor"] - factor) < 1e-6 and g["variant"] == variant}
        tp = sum(1 for s in d if s in gt_net and d[s]["is_network"]); fn = sum(1 for s in d if s in gt_net and not d[s]["is_network"])
        fp = sum(1 for s in d if s not in gt_net and d[s]["is_network"]); tn = sum(1 for s in d if s not in gt_net and not d[s]["is_network"])
        fps = sorted([s for s in d if s not in gt_net and d[s]["is_network"]], key=int); fns = sorted([s for s in d if s in gt_net and not d[s]["is_network"]], key=int)
        return tp, fn, fp, tn, fps, fns, d
    print(f"\n  {'source':16s}{'variant':9s}{'scale':>6s}{'factor':>7s}{'TP':>4s}{'FN':>4s}{'FP':>4s}{'TN':>5s}{'acc':>7s}   FP stems / FN stems")
    for src in ["gt"] + present:
        for variant in (["plain"] if src == "gt" else ["plain", "biascorr"]):
            for scale in ([1, 2, 4] if (src in ("gt", "zs_full") and variant == "plain") else [1]):
                tp, fn, fp, tn, fps, fns, _ = confusion(src, scale, GATE_FACTOR, variant)
                n = tp + fn + fp + tn
                if n == 0: continue
                print(f"  {src:16s}{variant:9s}{scale:6d}{GATE_FACTOR:7.1f}{tp:4d}{fn:4d}{fp:4d}{tn:5d}{(tp+tn)/n:7.3f}   FP={fps[:12]} FN={fns}")
    print("\n  因子扫描（原分辨率）：")
    for src in ["gt", "zs_full"]:
        for fac in [6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0]:
            tp, fn, fp, tn, fps, fns, _ = confusion(src, 1, fac)
            if tp + fn + fp + tn == 0: continue
            print(f"    {src:8s} factor={fac:5.1f}  TP={tp} FN={fn} FP={fp} TN={tn}  FP={fps[:10]} FN={fns}")
    print("\n  网状样本 + 任何 FP 的逐图细节（cells / holes / 中值宽 px / 最大三个洞的 sqrt(面积)/宽度，与 factor=14 直接比）：")
    _, _, _, _, fps_gt, _, dgt = confusion("gt"); _, _, _, _, fps_zs, _, dzs = confusion("zs_full")
    others = {src: confusion(src)[6] for src in present if src != "zs_full"}
    for s in sorted(gt_net | set(fps_gt) | set(fps_zs), key=int):
        a = dgt.get(s); b_ = dzs.get(s)
        print(f"    {s:5s} 专家={ctype.get(s)}  GT: cells={a['n_cells'] if a else '-'} holes={a['n_holes'] if a else '-'} w={a['w_med_px'] if a else 0:.1f} top3=[{a['top3'] if a else ''}]  "
              f"| zs_full: cells={b_['n_cells'] if b_ else '-'} holes={b_['n_holes'] if b_ else '-'} w={b_['w_med_px'] if b_ else 0:.1f} top3=[{b_['top3'] if b_ else ''}]")
        for src, d in others.items():
            c = d.get(s)
            if c: print(f"          {src:16s} cells={c['n_cells']} holes={c['n_holes']} w={c['w_med_px']:.1f} top3=[{c['top3']}]")
    # 预测掩膜门控 vs GT 掩膜门控的一致性
    print("\n  预测掩膜门控 vs GT 掩膜门控（同一图，两者判定一致的比例）：plain / biascorr")
    for src in present:
        _, _, _, _, _, _, d = confusion(src); _, _, _, _, _, _, d2 = confusion(src, 1, GATE_FACTOR, "biascorr")
        agree = np.mean([d[s]["is_network"] == dgt[s]["is_network"] for s in d if s in dgt])
        agree2 = np.mean([d2[s]["is_network"] == dgt[s]["is_network"] for s in d2 if s in dgt]) if d2 else np.nan
        print(f"    {src:16s} 一致 {agree:.3f} / {agree2:.3f}")
    # 垂直估计器排除率
    print("\n  perp 估计器的排除比例（端点/分叉；切向失败）：")
    for src in ["gt", "zs_full"]:
        rr = [r for r in rows if r["source"] == src and r["estimator"] == "perp" and r["n_pts"] > 0]
        tot = sum(r["n_pts"] + r["n_excl_deg"] + r["n_excl_ratio"] for r in rr)
        print(f"    {src:8s} 端点/分叉 {sum(r['n_excl_deg'] for r in rr)/tot:.1%}  切向失败 {sum(r['n_excl_ratio'] for r in rr)/tot:.1%}")

    # B3 校准
    print("\n" + "=" * 100 + f"\n[B3] 校准：{NSPLIT} 次 {NCAL}/{126-NCAL} 切分，seed {SEED}；log 模型 vs 硬阈值。论文对照（zs_full, edt, p95, T=10）: 点估计 Brier 0.219/acc 78.1/ref30 88.5；log 0.125/83.1/91.5，ECE fail 4.0%，b=0.175±0.032 σ=0.260±0.044\n" + "=" * 100)
    crow = []
    hdr = f"  {'source':16s}{'est':8s}{'red':7s}{'T':>5s}{'pos%':>6s} | {'Brier_pt':>8s}{'acc_pt':>7s}{'ref_pt':>7s} | {'Brier':>6s}{'NLL':>6s}{'acc':>6s}{'ref30':>6s}{'ECEf%':>6s}  {'b':>6s}{'sigma':>6s}"
    def run(src, est, red, T, label=""):
        ref_est = "perp" if est == "subpix" else est
        ssr, wr = get("gt", ref_est, red); ssp, wp = get(src, est, red)
        common = [s for s in ssr if s in set(ssp)]
        a = np.array([wr[ssr.index(s)] for s in common]); b_ = np.array([wp[ssp.index(s)] for s in common])
        if len(common) < 8: return None
        ncal = NCAL if len(common) >= NCAL + 10 else max(4, len(common) // 3)   # 冒烟模式缩小标定集
        o = calib_eval(b_, a, T, ncal=ncal, nsplit=NSPLIT if len(common) >= NCAL + 10 else 20, nsim=400 if len(common) >= NCAL + 10 else 50)
        o.update(source=src, estimator=est, reduction=red, T=T, label=label); crow.append(o)
        print(f"  {src:16s}{est:8s}{red:7s}{T:5.1f}{100*o['pos_rate']:6.1f} | {o['brier_pt']:8.3f}{100*o['acc_pt']:7.1f}{100*o['ref_pt']:7.1f} | "
              f"{o['brier']:6.3f}{o['nll']:6.3f}{100*o['acc_log']:6.1f}{100*o['ref_log']:6.1f}{100*o['ece_fail']:6.1f}  {o['b']:6.3f}{o['s']:6.3f}")
        return o
    print("\n  (a) 估计器敏感性 —— zs_full, P95, ASTM 10 mm（回应 R2-2 / R2-7 / R3-4）")
    print(hdr)
    for est in ESTIMATORS: run("zs_full", est, "p95", T_ASTM, "estimator")
    print("\n  (b) 掩膜源 × 估计器 —— 中值归约，ASTM 10 mm")
    print(hdr)
    for src in present:
        for est in ["edt", "perp"]: run(src, est, "median", T_ASTM, "source")
    print("\n  (c) 其他操作化 —— zs_full, edt（回应 R2-3）：ASTM 用 max / mean / median，JTG 3 mm 四种归约")
    print(hdr)
    for red in REDUCTIONS: run("zs_full", "edt", red, T_ASTM, "operationalisation")
    for red in REDUCTIONS: run("zs_full", "edt", red, T_JTG, "operationalisation")
    print("\n  (d) ASTM 阈值扫描 —— zs_full, edt, P95（回应 R2-8）；75 mm 无正样本不可评估")
    print(hdr)
    for T in [6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 20]: run("zs_full", "edt", "p95", float(T), "threshold")
    print("\n  (e) 阈值扫描 —— 中值归约")
    print(hdr)
    for T in [4, 5, 6, 7, 8, 9, 10, 12]: run("zs_full", "edt", "median", float(T), "threshold_median")
    if crow:
        with open(CSV_C, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(crow[0].keys())); w.writeheader(); w.writerows(crow)
        print(f"\n校准 CSV {CSV_C}")
    else:
        print("\n校准：样本不足，未运行")
    print(f"总耗时 {(time.time()-T0)/60:.1f} min")


if __name__ == "__main__":
    try: main()
    except Exception: traceback.print_exc()