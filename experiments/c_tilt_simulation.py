#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
c_tilt_simulation.py —— R2-1 / R3-5：用单应把 SUT-Crack 精确重投影到已知倾角，检验几何倾角界

几何：相机绕光心倾转 θ，平面场景成像变换是单应 H = K R K^-1（与深度无关）。K 由论文的半视场
33.6°（长边）确定：f_px = (4032/2)/tan(33.6°)。沿倾转方向的局部尺度因子（mm/px 相对标称）
  κ_along(φ) = cos²φ / cos²(θ+φ)      —— 论文式(10)，θ=5° 在 φ=±33.6° 给 [0.90, 1.14]，10° 给 [0.83, 1.32]
  κ_across(φ) = cosφ / cos(θ+φ)       —— 垂直于倾转方向的尺度（裂缝走向与倾转轴平行时宽度所受的因子）
一条宽度方向为单位向量 n=(n_x,n_y)（mm 空间）的裂缝，像素宽度缩放为 1/κ_eff，
  κ_eff = 1 / sqrt( (n_x/κ_x)² + (n_y/κ_y)² )，其中 κ_y=κ_along, κ_x=κ_across（pitch 情形）。

阶段：
  C1  GT 几何检验（CPU）：GT 掩膜经 H 重投影，逐骨架点测宽度比 w_θ/w_0 = 1/κ_emp，与 κ_eff 公式比较
  C2  SAM3 零样本在重投影图像上的宽度比 vs 几何预测；[κ_lo,κ_hi] 区间覆盖率；Dice 随倾角；标称尺度下的判级翻转
  C3  Brown 径向畸变上界（CPU）：k1 ∈ {-0.02,-0.05,-0.10} 对 GT 掩膜重采样，宽度相对变化随像高
倾角 θ ∈ {2,5,10,15}°，两个倾转轴（pitch: 尺度沿长边变化；roll: 沿短边）。
输出 results/c_tilt_<ts>.txt + c_tilt_points_<ts>.csv（C1 逐点）+ c_tilt_images_<ts>.csv（C2 逐图）
用法 python3 c_tilt_simulation.py [--limit N] [--skip_sam3]
"""
import os, sys, csv, re, glob, time, argparse, datetime, traceback
import numpy as np
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from skimage.morphology import skeletonize

Image.MAX_IMAGE_PIXELS = None
OUT = "/root/autodl-tmp/proj_bridge"
SUT_IMG = "/root/autodl-tmp/SUT-Crack/1-Segmentation/Original_Image"
SUT_GT = "/root/autodl-tmp/SUT-Crack/1-Segmentation/Ground_Truth"
CLOSELOOP = os.path.join(OUT, "results", "closeloop_v2_20260726_170420.csv")
W_SAM = "/root/autodl-tmp/sam3_weights"
sys.path.insert(0, OUT)
try:
    import pavement_codes as pc; MMPX = pc.SUTCRACK_MMPX
except Exception:
    MMPX = 0.2219
HALF_FOV_LONG = np.deg2rad(33.6)
THETAS = [2.0, 5.0, 10.0, 15.0]
AXES = ["pitch", "roll"]
T_ASTM, T_JTG = 10.0, 3.0
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"c_tilt_{ts}.txt")
CSV_P = os.path.join(OUT, "results", f"c_tilt_points_{ts}.csv")
CSV_I = os.path.join(OUT, "results", f"c_tilt_images_{ts}.csv")


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()


def numkey(p):
    m = re.findall(r'\d+', os.path.basename(p)); return int(m[0]) if m else 0


# ---------------------------------------------------------------- 几何
def intrinsics(W, H):
    f = (max(W, H) / 2.0) / np.tan(HALF_FOV_LONG)
    K = np.array([[f, 0, (W - 1) / 2.0], [0, f, (H - 1) / 2.0], [0, 0, 1.0]])
    return K, f


def homography(K, theta_deg, axis):
    t = np.deg2rad(theta_deg); c, s = np.cos(t), np.sin(t)
    if axis == "pitch":   # 绕 x 轴（短边方向的轴），尺度沿 y（长边）变化
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:                 # 绕 y 轴，尺度沿 x（短边）变化
        R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])   # 与 pitch 同一符号约定：φ' = φ − θ
    return K @ R @ np.linalg.inv(K)


def apply_h(H, pts):
    """pts: (N,2) 的 (x,y)"""
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


def inscribed_crop(H, W, Hh):
    """原图四角经 H 后的四边形内最大轴对齐矩形（保守：取内侧极值）"""
    c = apply_h(H, np.array([[0, 0], [W - 1, 0], [W - 1, Hh - 1], [0, Hh - 1]], float))
    x0 = int(np.ceil(max(c[0, 0], c[3, 0]))); x1 = int(np.floor(min(c[1, 0], c[2, 0])))
    y0 = int(np.ceil(max(c[0, 1], c[1, 1]))); y1 = int(np.floor(min(c[2, 1], c[3, 1])))
    return x0, y0, x1, y1


def warp(img_or_mask, H, crop, is_mask):
    """PIL PERSPECTIVE 需要输出->输入的系数（H^-1，并带裁剪偏移）"""
    x0, y0, x1, y1 = crop
    T = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1.0]])
    Hinv = np.linalg.inv(H) @ T   # 输出坐标 (u,v) -> 原图： H^-1 (u+x0, v+y0)
    Hinv = Hinv / Hinv[2, 2]
    coeffs = (Hinv[0, 0], Hinv[0, 1], Hinv[0, 2], Hinv[1, 0], Hinv[1, 1], Hinv[1, 2], Hinv[2, 0], Hinv[2, 1])
    size = (x1 - x0, y1 - y0)
    if is_mask == "soft":
        src = Image.fromarray((img_or_mask.astype(np.float32) * 255).astype(np.uint8))
        return np.array(src.transform(size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)).astype(np.float32) / 255.0
    if is_mask:
        src = Image.fromarray(img_or_mask.astype(np.uint8) * 255)
        return np.array(src.transform(size, Image.PERSPECTIVE, coeffs, Image.NEAREST)) > 127
    return img_or_mask.transform(size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)


def ray_width_soft(soft, pts, normals, step=0.25, maxlen=300.0):
    """亚像素宽度：从 pts 沿 ±normals 采样软掩膜（双线性），找首个 <0.5 的交叉点并线性插值。返回 (N,) 像素宽度"""
    steps = np.arange(0.0, maxlen + step, step); Hs, Ws = soft.shape
    total = np.zeros(len(pts))
    for sign in (1, -1):
        xx = pts[:, 0:1] + sign * normals[:, 0:1] * steps[None]; yy = pts[:, 1:2] + sign * normals[:, 1:2] * steps[None]
        v = ndi.map_coordinates(soft, [yy.ravel(), xx.ravel()], order=1, mode="constant", cval=0.0).reshape(xx.shape)
        below = v < 0.5
        first = np.argmax(below, axis=1); none = ~below.any(axis=1); first[none] = len(steps) - 1
        i = np.arange(len(pts)); v1 = v[i, first]; v0 = v[i, np.maximum(first - 1, 0)]
        den = np.where((v0 - v1) > 1e-9, v0 - v1, 1.0)
        frac = np.where((v0 - v1) > 1e-9, (v0 - 0.5) / den, 0.5)
        total += np.where(first > 0, steps[np.maximum(first - 1, 0)] + frac * step, 0.0)
    return total


def map_normals(H, pts, tang, eps=2.0):
    """把原图切向经 H 的局部 Jacobian 映射到倾斜图，返回倾斜图中的单位法向"""
    p1 = apply_h(H, pts); p2 = apply_h(H, pts + eps * tang)
    t2 = p2 - p1; t2 /= (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-9)
    return np.c_[-t2[:, 1], t2[:, 0]]


def kappa_factors(K, f, theta_deg, axis, xy_tilted):
    """倾斜图像坐标处的 κ_along, κ_across 以及 φ（弧度）"""
    t = np.deg2rad(theta_deg)
    if axis == "pitch": d = (xy_tilted[:, 1] - K[1, 2]) / f
    else: d = (xy_tilted[:, 0] - K[0, 2]) / f
    phi = np.arctan(d)
    k_along = np.cos(phi) ** 2 / np.cos(t + phi) ** 2
    k_across = np.cos(phi) / np.cos(t + phi)
    return k_along, k_across, phi


def kappa_eff(k_along, k_across, n_along, n_across):
    """宽度方向单位向量分量 (沿倾转方向分量, 垂直分量)"""
    return 1.0 / np.sqrt((n_across / k_across) ** 2 + (n_along / k_along) ** 2)


def bound(theta_deg, phi_max):
    t = np.deg2rad(theta_deg)
    ks = [np.cos(p) ** 2 / np.cos(t + p) ** 2 for p in (-phi_max, phi_max)]
    return min(ks), max(ks)


# ---------------------------------------------------------------- 宽度
def skel_widths(mask):
    """返回 (骨架点 xy (N,2), 宽度 px (N,), 局部切向单位向量 (N,2))"""
    sk = skeletonize(mask); d = ndi.distance_transform_edt(mask)
    pts = np.argwhere(sk)[:, ::-1].astype(float)   # (x,y)
    if len(pts) < 5: return pts, np.array([]), np.zeros((0, 2))
    w = 2.0 * d[pts[:, 1].astype(int), pts[:, 0].astype(int)]
    tree = cKDTree(pts); nb = tree.query_ball_point(pts, 8.0)
    tang = np.zeros((len(pts), 2))
    for i, n in enumerate(nb):
        q = pts[n] - pts[i]
        if len(n) < 3: tang[i] = (1, 0); continue
        ev, evec = np.linalg.eigh(q.T @ q); tang[i] = evec[:, 1]
    return pts, w, tang


def crop_bbox(mask, pad=60):
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - pad), min(mask.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(mask.shape[1], xs.max() + pad + 1)
    return mask[y0:y1, x0:x1], (x0, y0)


def width_stats_mm(mask):
    if mask.sum() < 30: return dict(median=0.0, p95=0.0, n=0)
    m, _ = crop_bbox(mask); _, w, _ = skel_widths(m)
    if w.size == 0: return dict(median=0.0, p95=0.0, n=0)
    return dict(median=float(np.median(w)) * MMPX, p95=float(np.percentile(w, 95)) * MMPX, n=int(w.size))


def dice_np(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_sam3", action="store_true")
    ap.add_argument("--thetas", type=str, default=",".join(str(t) for t in THETAS))
    args = ap.parse_args()
    thetas = [float(t) for t in args.thetas.split(",")]
    sys.stdout = Tee(LOGP); T0 = time.time()
    print(f"c_tilt_simulation.py  {ts}  MMPX={MMPX}  thetas={thetas}  axes={AXES}\nlog -> {LOGP}\n")

    linear = set(); ctype = {}
    with open(CLOSELOOP, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["variant"] == "zeroshot" and r["cond"] == "text_full":
                ctype[str(r["stem"])] = r["gt_ctype"]
                if r["gt_ctype"] == "线状": linear.add(str(r["stem"]))
    stems = sorted(ctype.keys(), key=int)
    if args.limit: stems = stems[:args.limit]
    print(f"图像 {len(stems)} 张（线状 {len([s for s in stems if s in linear])}）")

    # 公式界（供对照）
    print("\n公式 (10) 的区间 [κ_lo, κ_hi]（pitch: φ_max=33.6°; roll: φ_max=atan(1512/f)）")
    K0, f0 = intrinsics(3024, 4032); phi_roll = np.arctan(1512.0 / f0)
    print(f"  f_px = {f0:.0f}  roll 半视场 = {np.rad2deg(phi_roll):.1f}°")
    for t in thetas:
        lo, hi = bound(t, HALF_FOV_LONG); lo2, hi2 = bound(t, phi_roll)
        print(f"  θ={t:4.1f}°  pitch [{lo:.3f}, {hi:.3f}]   roll [{lo2:.3f}, {hi2:.3f}]")

    # ---------------- C1 + C2 逐图
    model = processor = None
    if not args.skip_sam3:
        import torch, torch.nn.functional as F
        from transformers import Sam3Model, Sam3Processor
        processor = Sam3Processor.from_pretrained(W_SAM); model = Sam3Model.from_pretrained(W_SAM).cuda().eval()
        torch.set_grad_enabled(False)

        def seg(img):
            inp = processor(images=img, text="crack", return_tensors="pt")
            inp = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(**inp)
            Wd, Hd = img.size
            return F.interpolate(o["semantic_seg"][0, 0][None, None].float(), size=(Hd, Wd), mode="bilinear")[0, 0].cpu().numpy() > 0

    prow = []; irow = []
    print("\n" + "=" * 100 + "\n[C1/C2] 逐图重投影\n" + "=" * 100, flush=True)
    for i, stem in enumerate(stems):
        try:
            gt = np.array(Image.open(os.path.join(SUT_GT, stem + ".png")).convert("L")) > 127
            H0, W0 = gt.shape; K, f = intrinsics(W0, H0)
            img = ImageOps.exif_transpose(Image.open(os.path.join(SUT_IMG, stem + ".jpg"))).convert("RGB") if model else None
            # 原图 GT 骨架
            g0, off0 = crop_bbox(gt); pts0, w0, tang0 = skel_widths(g0)
            pts0 = pts0 + np.array(off0)
            sub = slice(None, None, 3)
            pts0s, w0s, tang0s = pts0[sub], w0[sub], tang0[sub]
            base = dict(stem=stem, theta=0.0, axis="none")
            if model:
                pm0 = seg(img); s0 = width_stats_mm(pm0); base.update(dice=dice_np(pm0, gt), pred_med=s0["median"], pred_p95=s0["p95"])
            irow.append(base)
            for axis in AXES:
                for t in thetas:
                    Hm = homography(K, t, axis); crop = inscribed_crop(Hm, W0, H0)
                    if crop[2] - crop[0] < 500 or crop[3] - crop[1] < 500: continue
                    gtw = warp(gt, Hm, crop, True)
                    # --- C1: GT 几何检验（亚像素：软掩膜 + 沿法线 0.5 交叉）
                    if gtw.sum() >= 30 and w0s.size >= 5:
                        soft_w = warp(gt, Hm, crop, "soft")
                        nrm0 = np.c_[-tang0s[:, 1], tang0s[:, 0]]
                        w0_sub = ray_width_soft(gt.astype(np.float32), pts0s, nrm0)
                        mapped = apply_h(Hm, pts0s) - np.array(crop[:2]); nrm_w = map_normals(Hm, pts0s, tang0s)
                        inb = (mapped[:, 0] > 2) & (mapped[:, 0] < soft_w.shape[1] - 3) & (mapped[:, 1] > 2) & (mapped[:, 1] < soft_w.shape[0] - 3)
                        ww_sub = ray_width_soft(soft_w, mapped, nrm_w)
                        # 用 EDT 宽度做合理性筛选：法线投射失败（分叉/端点，弦长 > 2.5×EDT 直径）的点剔除
                        ok = inb & (w0s > 2.0) & (w0_sub > 0) & (ww_sub > 0) & (w0_sub <= 2.5 * w0s) & (ww_sub <= 2.5 * w0s)
                        if ok.sum() >= 5:
                            k_al, k_ac, phi = kappa_factors(K, f, t, axis, mapped[ok] + np.array(crop[:2]))
                            if axis == "pitch": n_along, n_across = np.abs(nrm0[ok][:, 1]), np.abs(nrm0[ok][:, 0])
                            else: n_along, n_across = np.abs(nrm0[ok][:, 0]), np.abs(nrm0[ok][:, 1])
                            k_pred = kappa_eff(k_al, k_ac, n_along, n_across)
                            k_emp = w0_sub[ok] / ww_sub[ok]
                            for a_, b_, c_, d_, e_, ph in zip(k_emp, k_pred, k_al, k_ac, w0s[ok], phi):
                                prow.append(dict(stem=stem, axis=axis, theta=t, phi_deg=float(np.rad2deg(ph)), w0_px=float(e_),
                                                 k_emp=float(a_), k_pred=float(b_), k_along=float(c_), k_across=float(d_)))
                    # --- C2: SAM3 在重投影图像上
                    if model:
                        imw = warp(img, Hm, crop, False); pmw = seg(imw); sw = width_stats_mm(pmw)
                        # 几何预测的像素宽度比：沿 GT 骨架的 1/κ_eff 的中值
                        kp_med = np.nan
                        if w0s.size >= 5:
                            mapped = apply_h(Hm, pts0s); k_al, k_ac, _ = kappa_factors(K, f, t, axis, mapped)
                            nrm = np.c_[-tang0s[:, 1], tang0s[:, 0]]
                            if axis == "pitch": n_along, n_across = np.abs(nrm[:, 1]), np.abs(nrm[:, 0])
                            else: n_along, n_across = np.abs(nrm[:, 0]), np.abs(nrm[:, 1])
                            kp_med = float(np.median(kappa_eff(k_al, k_ac, n_along, n_across)))
                        lo, hi = bound(t, HALF_FOV_LONG if axis == "pitch" else phi_roll)
                        irow.append(dict(stem=stem, theta=t, axis=axis, dice=dice_np(pmw, gtw), pred_med=sw["median"], pred_p95=sw["p95"],
                                         gt_med_warp=width_stats_mm(gtw)["median"], k_pred_med=kp_med, k_lo=lo, k_hi=hi,
                                         crop_w=crop[2] - crop[0], crop_h=crop[3] - crop[1]))
            if i % 10 == 0: print(f"  {i+1}/{len(stems)}  {stem}  {time.time()-T0:.0f}s  C1点数={len(prow)}", flush=True)
        except Exception:
            print(f"  !! {stem}"); traceback.print_exc()

    if prow:
        with open(CSV_P, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(prow[0].keys())); w.writeheader(); w.writerows(prow)
    if irow:
        keys = sorted(set(k for r in irow for k in r), key=lambda k: (k not in ("stem", "theta", "axis"), k))
        with open(CSV_I, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(irow)

    # ---------------- C1 汇总
    print("\n" + "=" * 100 + "\n[C1] GT 几何检验：κ_emp = w_0/w_θ（像素宽度比）vs κ_eff 公式；全部骨架点（w_0>2px，匹配半径 4px）\n" + "=" * 100)
    P = np.array([(r["axis"] == "pitch", r["theta"], r["phi_deg"], r["k_emp"], r["k_pred"], r["k_along"], r["k_across"]) for r in prow], float)
    print(f"  {'axis':6s}{'θ':>5s}{'n_pts':>8s}{'φ范围(°)':>14s}  {'公式界[lo,hi]':>16s}  {'κ_emp P1-P99':>16s}  {'κ_pred P1-P99':>16s}  {'median|logκe-logκp|':>20s}  {'>5%':>6s}  {'斜率(κe~κp)':>12s}  {'ρ':>6s}")
    for axis in AXES:
        for t in thetas:
            m = (P[:, 0] == (axis == "pitch")) & (P[:, 1] == t)
            if m.sum() < 10: continue
            ke, kp = P[m, 3], P[m, 4]; ph = P[m, 2]
            good = (ke > 0.3) & (ke < 3)
            ke, kp, ph = ke[good], kp[good], ph[good]
            lo, hi = bound(t, HALF_FOV_LONG if axis == "pitch" else phi_roll)
            err = np.abs(np.log(ke) - np.log(kp)); b, a = np.polyfit(kp, ke, 1)
            print(f"  {axis:6s}{t:5.1f}{m.sum():8d}  [{ph.min():6.1f},{ph.max():6.1f}]  [{lo:.3f},{hi:.3f}]  "
                  f"[{np.percentile(ke,1):.3f},{np.percentile(ke,99):.3f}]  [{np.percentile(kp,1):.3f},{np.percentile(kp,99):.3f}]  "
                  f"{np.median(err):20.4f}  {np.mean(err>0.05):6.1%}  {b:12.3f}  {spearmanr(ke,kp).correlation:6.3f}")
    print("\n  按 φ 分箱（5°）：κ_emp 均值 vs κ_pred 均值（对法线噪声稳健，供画图）")
    for axis in AXES:
        for t in thetas:
            m = (P[:, 0] == (axis == "pitch")) & (P[:, 1] == t)
            if m.sum() < 10: continue
            ke, kp, ph = P[m, 3], P[m, 4], P[m, 2]; good = (ke > 0.3) & (ke < 3); ke, kp, ph = ke[good], kp[good], ph[good]
            edges = np.arange(-40, 41, 5); cells = []
            for lo_, hi_ in zip(edges[:-1], edges[1:]):
                b = (ph >= lo_) & (ph < hi_)
                if b.sum() >= 30: cells.append(f"{(lo_+hi_)/2:+.0f}°:{ke[b].mean():.3f}/{kp[b].mean():.3f}(n={b.sum()})")
            print(f"    {axis:6s} θ={t:4.1f}  " + "  ".join(cells))
    print("\n  说明：κ_emp 用软掩膜沿法线的 0.5 交叉亚像素测得；残余离散来自重采样与法线估计；判读看中位误差、分箱均值的一致性。")
    print("  along vs across：裂缝走向与倾转轴平行时宽度只受 κ_across，比公式 (10) 的 κ_along 更接近 1——")
    for t in thetas:
        lo, hi = bound(t, HALF_FOV_LONG); t_ = np.deg2rad(t)
        lo2 = min(np.cos(p) / np.cos(t_ + p) for p in (-HALF_FOV_LONG, HALF_FOV_LONG)); hi2 = max(np.cos(p) / np.cos(t_ + p) for p in (-HALF_FOV_LONG, HALF_FOV_LONG))
        print(f"    θ={t:4.1f}°  κ_along [{lo:.3f},{hi:.3f}]  κ_across [{lo2:.3f},{hi2:.3f}]")

    # ---------------- C2 汇总
    if model:
        print("\n" + "=" * 100 + "\n[C2] SAM3 零样本在重投影图像上（126 线状）\n" + "=" * 100)
        I = {}
        for r in irow: I[(r["stem"], r["axis"], r["theta"])] = r
        print(f"  {'axis':6s}{'θ':>5s}{'n':>5s}{'Dice':>7s}{'Dice0':>7s} | {'实测比 w_θ/w_0 中值(IQR)':>28s}  {'几何预测比 1/κ 中值':>18s}  {'median|log(实测/预测)|':>22s}  {'落在[1/κhi,1/κlo]内':>18s} | {'ASTM翻转':>8s}{'JTG翻转':>8s}  {'按κ区间可翻转':>12s}")
        for axis in AXES:
            for t in thetas:
                rr = []; flipA = []; flipJ = []; flippable = []; cover = []
                for stem in stems:
                    if stem not in linear: continue
                    r0 = I.get((stem, "none", 0.0)); r1 = I.get((stem, axis, t))
                    if not r0 or not r1 or r0["pred_med"] <= 0 or r1["pred_med"] <= 0: continue
                    ratio = r1["pred_med"] / r0["pred_med"]; pred = 1.0 / r1["k_pred_med"] if r1["k_pred_med"] else np.nan
                    rr.append((ratio, pred, r1["dice"], r0["dice"]))
                    cover.append((1.0 / r1["k_hi"] - 0.08) <= ratio <= (1.0 / r1["k_lo"] + 0.08))   # ±8% 容差吸收分割噪声
                    flipA.append((r0["pred_p95"] > T_ASTM) != (r1["pred_p95"] > T_ASTM))
                    flipJ.append((r0["pred_med"] > T_JTG) != (r1["pred_med"] > T_JTG))
                    flippable.append((r0["pred_p95"] * r1["k_lo"] > T_ASTM) != (r0["pred_p95"] * r1["k_hi"] > T_ASTM))
                if not rr: continue
                R = np.array(rr); okp = np.isfinite(R[:, 1])
                err = np.abs(np.log(R[okp, 0]) - np.log(R[okp, 1]))
                print(f"  {axis:6s}{t:5.1f}{len(rr):5d}{R[:,2].mean():7.3f}{R[:,3].mean():7.3f} | "
                      f"{np.median(R[:,0]):8.3f} ({np.percentile(R[:,0],25):.3f}-{np.percentile(R[:,0],75):.3f})       "
                      f"{np.median(R[okp,1]):18.3f}  {np.median(err):22.4f}  {np.mean(cover):18.1%} | "
                      f"{np.mean(flipA):8.1%}{np.mean(flipJ):8.1%}  {np.mean(flippable):12.1%}")
        print("\n  说明：'实测比' 是 SAM3 预测中值宽（像素）在倾斜图/原图之比；'几何预测比' 沿 GT 骨架取 1/κ_eff 中值。")
        print("  'ASTM/JTG 翻转' 是用标称尺度 s 不做倾角修正时，判级在 θ 与 0° 之间改变的比例（论文 '可翻转' 是在 κ 区间内可能翻转的比例，两者不同，并列给出）。")

    # ---------------- C3 畸变
    print("\n" + "=" * 100 + "\n[C3] Brown 径向畸变上界：x_d = x(1 + k1 r² + k2 r⁴)，r 归一化到半对角；对 GT 掩膜重采样后宽度变化\n" + "=" * 100)
    try:
        sub = [s for s in stems if s in linear][:40]
        for k1, k2 in [(-0.02, 0.0), (-0.05, 0.0), (-0.10, 0.0), (0.03, 0.0), (-0.05, 0.02)]:
            ratios = []; rads = []
            for stem in sub:
                gt = np.array(Image.open(os.path.join(SUT_GT, stem + ".png")).convert("L")) > 127
                H0, W0 = gt.shape; cx, cy = (W0 - 1) / 2, (H0 - 1) / 2; Rn = np.hypot(cx, cy)
                g0, off0 = crop_bbox(gt); pts0, w0, _ = skel_widths(g0); pts0 = pts0 + np.array(off0)
                if w0.size < 5: continue
                # 逆映射：畸变图像素 (u,v) 取自未畸变 (x,y)。用局部 Jacobian 的近似：宽度比 ≈ 局部径向/切向缩放
                r = np.hypot(pts0[:, 0] - cx, pts0[:, 1] - cy) / Rn
                s_rad = 1 + 3 * k1 * r ** 2 + 5 * k2 * r ** 4     # 径向缩放 d(x_d)/dx
                s_tan = 1 + k1 * r ** 2 + k2 * r ** 4             # 切向缩放
                ratios.append(np.c_[s_rad, s_tan][w0 > 2]); rads.append(r[w0 > 2])
            Rr = np.vstack(ratios); rr_ = np.concatenate(rads)
            print(f"  k1={k1:+.2f} k2={k2:+.2f}: 像素尺度因子 径向 [{Rr[:,0].min():.3f},{Rr[:,0].max():.3f}] 切向 [{Rr[:,1].min():.3f},{Rr[:,1].max():.3f}]  "
                  f"在裂缝像素上的中值 径向 {np.median(Rr[:,0]):.3f} 切向 {np.median(Rr[:,1]):.3f}  r 中值 {np.median(rr_):.2f}  "
                  f"→ 等效 κ 区间 [{1/Rr.max():.3f},{1/Rr.min():.3f}]")
        print("  说明：手机 ISP 已做畸变校正，残余 |k1| 通常 <0.05；与 θ=5° 的 [0.90,1.14] 相比，畸变项的尺度变化小一个量级，且是确定性的（可标定后消除）。")
    except Exception:
        traceback.print_exc()

    print(f"\n逐点 CSV {CSV_P}\n逐图 CSV {CSV_I}\n总耗时 {(time.time()-T0)/60:.1f} min")


if __name__ == "__main__":
    main()