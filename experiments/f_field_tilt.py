#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f_field_tilt.py —— R2-1 / R3-5 的实拍验证：5 条裂缝 × 4 个倾角（手机水平仪读数），画面内两张 10 mm 网格纸 + 3 个卡尺点

输入  /root/autodl-tmp/field/<crack>-<k>.jpg（3072×4096，EXIF 含 35 mm 等效焦距）与 field_log.csv
F1  网格纸检测 → 按条带测网格周期 → 局部 mm/px（沿倾转方向 / 垂直方向）→ 与公式 κ_along=cos²φ/cos²(θ+φ)、κ_across=cosφ/cos(θ+φ) 比较
    以 0° 图定标称尺度 s0；对每张倾斜图：按读数 θ 的预测残差、以及自由拟合的 θ_fit（与水平仪读数对照）
F2  0° 图上 mm/px 随像高的残余变化 → 该手机残余畸变上界
F3  SAM3 零样本（排除纸和粉笔）→ 宽度中值/P95（px）→ 标称尺度 vs 倾角修正后的 mm；w_θ/w_0 实测比 vs 几何预测比
F4  人工定点（field/click_widths_*.csv，本机 k_click_width.py 产出）：同一位置三方对照 卡尺 / 人工图像开口 / SAM3 缝芯；SAM3 掩膜缓存在 results/field_masks
输出  results/f_field_<ts>.txt + f_field_bands/widths/points_<ts>.csv + results/field_overlays/*.jpg（人工复核）
用法  python3 f_field_tilt.py [--crack 3] [--no_sam3]
"""
import os, sys, csv, re, time, argparse, datetime, traceback
import numpy as np
from PIL import Image, ImageOps, ImageDraw
from PIL.ExifTags import TAGS
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage import measure, transform, morphology
from skimage.morphology import skeletonize

sys.path.insert(0, "/root/autodl-tmp/proj_bridge")
import e1_collapse_mitigation as E

FIELD = "/root/autodl-tmp/field"; OUT = E.OUT
LOG = os.path.join(FIELD, "field_log.csv")
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"f_field_{ts}.txt")
CSV_B = os.path.join(OUT, "results", f"f_field_bands_{ts}.csv"); CSV_W = os.path.join(OUT, "results", f"f_field_widths_{ts}.csv"); CSV_P = os.path.join(OUT, "results", f"f_field_points_{ts}.csv")
OVD = os.path.join(OUT, "results", "field_overlays"); os.makedirs(OVD, exist_ok=True)
MSK = os.path.join(OUT, "results", "field_masks"); os.makedirs(MSK, exist_ok=True)   # SAM3 掩膜缓存（原始 sem_mask，未做排除）


def load_clicks():
    """本机 k_click_width.py 的人工点选结果：{file: [(tick,x1,y1,x2,y2,manual_px), ...]}，取最新一份"""
    import glob
    fs = sorted(glob.glob(os.path.join(FIELD, "click_widths_*.csv")))
    if not fs: return None, None
    out = {}
    with open(fs[-1], encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["file"], []).append((int(r["tick"]), float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]), float(r["width_px"])))
    return out, fs[-1]


def mask_width_along(m, cx, cy, ux, uy, half=200, search=80):
    """沿方向 (ux,uy) 过点 (cx,cy) 采样掩膜剖面，取中心最接近该点（≤search px）的连续掩膜段长度；返回 (宽px, 段中心偏移px)"""
    H, W = m.shape
    t = np.arange(-half, half + 1)
    xs = np.clip(np.round(cx + t * ux).astype(int), 0, W - 1); ys = np.clip(np.round(cy + t * uy).astype(int), 0, H - 1)
    prof = m[ys, xs]
    runs = []; i = 0
    while i < len(prof):
        if prof[i]:
            j = i
            while j + 1 < len(prof) and prof[j + 1]: j += 1
            runs.append((i, j)); i = j + 1
        else: i += 1
    if not runs: return np.nan, np.nan
    best = min(runs, key=lambda r: abs((r[0] + r[1]) / 2 - half))
    off = (best[0] + best[1]) / 2 - half
    if abs(off) > search: return np.nan, off
    return float(best[1] - best[0] + 1), float(off)
DS = 4                      # 检测用降采样倍数
N_BAND, N_STRIP = 6, 3      # 每张纸沿长边 6 条带（测 across 周期）、沿短边 3 条（测 along 周期）
LAG_MIN, LAG_MAX = 25, 90   # 10 mm 网格在 0.11–0.40 mm/px 下的像素周期范围（标称 46 px）


# ---------------------------------------------------------------- 工具
def read_img(p):
    im = Image.open(p); ex = im.getexif(); f35 = None
    try:
        ifd = ex.get_ifd(0x8769)
        for k, v in ifd.items():
            if TAGS.get(k) == "FocalLengthIn35mmFilm": f35 = float(v)
    except Exception: pass
    return ImageOps.exif_transpose(im).convert("RGB"), f35


def f_px_from_f35(f35, W, H):
    half = np.arctan(18.0 / f35)                 # 35 mm 底片长边 36 mm
    return (max(W, H) / 2.0) / np.tan(half), half


def kappa_along(phi, th): return np.cos(phi) ** 2 / np.cos(th + phi) ** 2
def kappa_across(phi, th): return np.cos(phi) / np.cos(th + phi)


def autocorr_period(profile, return_peak=False):
    """1D 强度剖面的主周期（像素，亚像素抛物线精化）；要求是区间内部的局部极大且峰值≥0.15；失败返回 nan"""
    x = profile.astype(float)
    if len(x) < int(2.5 * LAG_MAX): return (np.nan, 0.0) if return_peak else np.nan
    x = x - ndi.uniform_filter1d(x, 151); x = x - x.mean()
    if x.std() < 1e-6: return (np.nan, 0.0) if return_peak else np.nan
    n = len(x); fx = np.fft.rfft(x, 2 * n); ac = np.fft.irfft(fx * np.conj(fx))[:n]; ac /= ac[0] + 1e-12
    seg = ac[LAG_MIN:LAG_MAX]
    loc = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] >= 0.15]
    if not loc: return (np.nan, 0.0) if return_peak else np.nan
    i = max(loc, key=lambda j: seg[j]); k = i + LAG_MIN
    a, b, c = seg[i - 1], seg[i], seg[i + 1]; den = a - 2 * b + c
    if abs(den) > 1e-9: k += 0.5 * (a - c) / den
    return (float(k), float(seg[i])) if return_peak else float(k)


def detect_papers(img):
    """返回两张纸的 (bbox 全分辨率 y0,x0,y1,x1, 方向角 rad)。两张纸粘连时沿长轴在中段白像素最少处切开。"""
    small = np.asarray(img.resize((img.width // DS, img.height // DS), Image.BILINEAR)).astype(float)
    lum = small.mean(2); sat = small.max(2) - small.min(2)
    road = np.median(lum); thr = max(170.0, road + 45.0)
    white = (lum > thr) & (sat < 95)                       # 纸偏蓝，饱和度阈值要宽
    white = morphology.closing(white, morphology.disk(3)); white = ndi.binary_fill_holes(white); white = morphology.opening(white, morphology.disk(5))
    lab = measure.label(white); A = white.size
    pieces = []
    for q in measure.regionprops(lab):
        if not (0.005 * A < q.area < 0.3 * A) or q.solidity < 0.8: continue
        h = q.bbox[2] - q.bbox[0]; w = q.bbox[3] - q.bbox[1]; ar = max(h, w) / max(1, min(h, w))
        sub = (lab[q.bbox[0]:q.bbox[2], q.bbox[1]:q.bbox[3]] == q.label)
        if ar > 1.7:                                       # 两张纸粘连
            prof = sub.sum(1) if h > w else sub.sum(0); n = len(prof); lo, hi = int(0.38 * n), int(0.62 * n)
            cut = lo + int(np.argmin(prof[lo:hi]))
            parts = [(sub[:cut], (0, 0)), (sub[cut:], (cut, 0))] if h > w else [(sub[:, :cut], (0, 0)), (sub[:, cut:], (0, cut))]
            for part, (oy, ox) in parts:
                for pq in measure.regionprops(measure.label(part)):
                    if pq.area > 0.004 * A: pieces.append((pq, q.bbox[0] + oy, q.bbox[1] + ox))
        elif 1.1 < ar < 1.7:
            pieces.append((q, 0, 0))
    pieces = sorted(pieces, key=lambda t: -t[0].area)[:2]
    out = []
    for q, oy, ox in pieces:
        y0, x0, y1, x1 = q.bbox
        y0, x0, y1, x1 = [(v + o) * DS for v, o in ((y0, oy), (x0, ox), (y1, oy), (x1, ox))]
        out.append(dict(bbox=(y0, x0, y1, x1), angle=float(q.orientation), area=q.area * DS * DS, cy=(y0 + y1) / 2, cx=(x0 + x1) / 2))
    return out


def paper_bands(gray, paper, f, cy_img, s_unit=10.0):
    """在一张纸内测 across(x) 与 along(y) 的网格周期。返回 list of dict(kind, y_img, x_img, phi, period_px, mmpx)"""
    y0, x0, y1, x1 = paper["bbox"]
    crop = gray[y0:y1, x0:x1]; ang = np.degrees(paper["angle"])
    # 旋转方向由数据决定：在 {-ang, 0, +ang} 中取自相关峰最强者
    best = None
    for rot in sorted({-ang, 0.0, ang}):
        rc = transform.rotate(crop, rot, resize=False, mode="edge", preserve_range=True); H, W = rc.shape
        inner = rc[int(0.1 * H):int(0.9 * H), int(0.1 * W):int(0.9 * W)]
        _, pa = autocorr_period(inner.mean(0), True); _, pb = autocorr_period(inner.mean(1), True)
        if best is None or pa + pb > best[0]: best = (pa + pb, rot, rc)
    _, rot, rc = best
    H, W = rc.shape; my, mx = int(0.10 * H), int(0.10 * W)
    inner = rc[my:H - my, mx:W - mx]; h, w = inner.shape
    out = []
    for b in range(N_BAND):                      # 横向条带：沿 x 的剖面 → 竖直网格线周期 → across 尺度
        ya, yb = int(b * h / N_BAND), int((b + 1) * h / N_BAND)
        px = autocorr_period(inner[ya:yb].mean(0))
        yc = y0 + my + (ya + yb) / 2; xc = x0 + W / 2
        out.append(dict(kind="across", y_img=yc, x_img=xc, phi=float(np.arctan((yc - cy_img) / f)), period_px=px, mmpx=(s_unit / px if px == px else np.nan)))
    for s_ in range(N_STRIP):                    # 纵向条带：沿 y 的剖面 → 水平网格线周期 → along 尺度；按 y 分 2 段
        xa, xb = int(s_ * w / N_STRIP), int((s_ + 1) * w / N_STRIP)
        for b in range(2):
            ya, yb = int(b * h / 2), int((b + 1) * h / 2)
            py = autocorr_period(inner[ya:yb, xa:xb].mean(1))
            yc = y0 + my + (ya + yb) / 2; xc = x0 + mx + (xa + xb) / 2
            out.append(dict(kind="along", y_img=yc, x_img=xc, phi=float(np.arctan((yc - cy_img) / f)), period_px=py, mmpx=(s_unit / py if py == py else np.nan)))
    return out, rot


def _resid(bands, th_deg, s0=None):
    """给定 θ，若 s0=None 则取最优 s0（log 均值闭式解）；返回 (rms%, s0_used)"""
    th = np.radians(th_deg); lp = []; lm = []
    for b in bands:
        if b["mmpx"] != b["mmpx"]: continue
        k = kappa_along(b["phi"], th) if b["kind"] == "along" else kappa_across(b["phi"], th)
        lp.append(np.log(k)); lm.append(np.log(b["mmpx"]))
    if len(lp) < 4: return np.nan, np.nan
    lp, lm = np.array(lp), np.array(lm)
    ls0 = np.log(s0) if s0 else float(np.mean(lm - lp))
    r = lm - lp - ls0
    return 100 * float(np.sqrt(np.mean(r * r))), float(np.exp(ls0))


def fit_theta(bands, s0=None):
    """自由拟合 θ（含符号；s0 可固定或同时拟合）；返回 θ_fit(deg), rms(%), s0_fit"""
    best = (None, 1e9, np.nan)
    for th_deg in np.arange(-25, 25.01, 0.1):
        rms, s0u = _resid(bands, th_deg, s0)
        if rms == rms and rms < best[1]: best = (float(th_deg), rms, s0u)
    return best


def resid_for_theta(bands, s0, th_deg):
    return _resid(bands, th_deg, s0)[0]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--crack", type=int, default=0); ap.add_argument("--no_sam3", action="store_true"); args = ap.parse_args()
    sys.stdout = E.Tee(LOGP); T0 = time.time()
    print(f"f_field_tilt.py  {ts}\nlog -> {LOGP}\n")
    log = list(csv.DictReader(open(LOG, encoding="utf-8")))
    if args.crack: log = [r for r in log if int(r["crack"]) == args.crack]
    model = processor = None
    if not args.no_sam3:
        import torch
        from transformers import Sam3Model, Sam3Processor
        processor = Sam3Processor.from_pretrained(E.W); model = Sam3Model.from_pretrained(E.W).cuda().eval(); torch.set_grad_enabled(False)

    brows, wrows, prows = [], [], []
    per_img = {}
    print("=" * 100 + "\n[F1] 网格纸局部尺度 vs 几何公式\n" + "=" * 100)
    for r in log:
        name = r["file"]; crack = int(r["crack"]); th_read = float(r["tilt_deg"]); grid = float(r["grid_mm"])
        try:
            img, f35 = read_img(os.path.join(FIELD, name)); f35 = f35 or float(r["f35_mm"])
            W, H = img.size; f, half = f_px_from_f35(f35, W, H); cy = (H - 1) / 2
            gray = np.asarray(img.convert("L")).astype(float)
            papers = detect_papers(img)
            bands = []; rots = []
            for p in papers:
                bb, rot = paper_bands(gray, p, f, cy, grid); bands += bb; rots.append(rot)
            ok = [b for b in bands if b["mmpx"] == b["mmpx"]]
            if ok:                                           # 剔除谐波误检：周期偏离该图中值 ×[0.7,1.4] 之外的条带
                medp = np.median([b["period_px"] for b in ok]); n_before = len(ok)
                ok = [b for b in ok if 0.7 * medp <= b["period_px"] <= 1.4 * medp]
            per_img[name] = dict(img=img, f=f, half=half, papers=papers, bands=ok, crack=crack, th=th_read, W=W, H=H)
            for b in ok: brows.append(dict(file=name, crack=crack, tilt_deg=th_read, **b))
            print(f"  {name}: {W}x{H} f35={f35:.0f} f_px={f:.0f} 半视场={np.degrees(half):.1f}°  纸 {len(papers)} 张(旋转校正 {', '.join(f'{x:.1f}°' for x in rots)})  有效条带 {len(ok)}/{len(bands)}  "
                  f"mm/px across 中值 {np.median([b['mmpx'] for b in ok if b['kind']=='across']) if ok else np.nan:.4f}  along {np.median([b['mmpx'] for b in ok if b['kind']=='along']) if ok else np.nan:.4f}  φ 范围 [{np.degrees(min(b['phi'] for b in ok)) if ok else 0:+.1f},{np.degrees(max(b['phi'] for b in ok)) if ok else 0:+.1f}]°")
        except Exception:
            print(f"  !! {name}"); traceback.print_exc()

    # 每条裂缝：0° 定 s0，倾斜图检验
    print("\n  逐图：s0 来自该裂缝的 0° 图（两方向所有条带的中值）；'读数残差' = 用水平仪读数 θ 预测的 log-RMS；θ_fit = 自由拟合（符号表示俯仰方向）")
    print(f"  {'file':9s}{'θ读数':>6s}{'s0(mm/px)':>10s}{'标称h/f':>8s}{'读数残差%':>9s}{'θ_fit':>7s}{'fit残差%':>8s}{'无倾角残差%':>11s}{'s0_fit/s0':>9s} | across: 实测κ范围 / 预测κ范围 | along: 实测 / 预测")
    print("  （残差均为 s0 自由时的 log-RMS，只检验 κ(φ) 的形状；s0_fit/s0 反映各次拍摄的高度抖动）")
    s0_of = {}
    for name, d in per_img.items():
        if d["th"] == 0.0 and d["bands"]:
            s0_of[d["crack"]] = float(np.median([b["mmpx"] for b in d["bands"]]))
    for name, d in per_img.items():
        c = d["crack"]; s0 = s0_of.get(c)
        if s0 is None or not d["bands"]: continue
        nominal = 670.0 / d["f"]
        th_fit, rms_fit, s0_fit = fit_theta(d["bands"], None)
        rp, rm = _resid(d["bands"], d["th"], None)[0], _resid(d["bands"], -d["th"], None)[0]
        rms_read = min(rp, rm); rms_zero = _resid(d["bands"], 0.0, None)[0]
        sign = 1.0 if rp <= rm else -1.0
        th = np.radians(sign * d["th"])
        ka = [b["mmpx"] / s0 for b in d["bands"] if b["kind"] == "across"]; kal = [b["mmpx"] / s0 for b in d["bands"] if b["kind"] == "along"]
        pa = [kappa_across(b["phi"], th) for b in d["bands"] if b["kind"] == "across"]; pal = [kappa_along(b["phi"], th) for b in d["bands"] if b["kind"] == "along"]
        d.update(s0=s0, sign=sign, th_fit=th_fit, s0_fit=s0_fit)
        print(f"  {name:9s}{d['th']:6.1f}{s0:10.4f}{nominal:8.4f}{rms_read:9.2f}{(th_fit if th_fit is not None else np.nan):7.1f}{rms_fit:8.2f}{rms_zero:11.2f}{s0_fit/s0:9.3f} | "
              f"[{min(ka):.3f},{max(ka):.3f}] / [{min(pa):.3f},{max(pa):.3f}] | [{min(kal):.3f},{max(kal):.3f}] / [{min(pal):.3f},{max(pal):.3f}]")
    reads = [(d["th"], abs(d["th_fit"])) for d in per_img.values() if d.get("th_fit") is not None and d["th"] > 0]
    if reads:
        R = np.array(reads); print(f"\n  θ_fit vs 水平仪读数（{len(R)} 张倾斜图）：平均差 {np.mean(R[:,1]-R[:,0]):+.2f}°，|差| 中位 {np.median(np.abs(R[:,1]-R[:,0])):.2f}°，最大 {np.max(np.abs(R[:,1]-R[:,0])):.2f}°")
    s0s = list(s0_of.values()); print(f"  五条裂缝的 s0：{', '.join(f'{v:.4f}' for v in s0s)} mm/px（标称 670/f = {670/np.mean([d['f'] for d in per_img.values()]):.4f}；SUT-Crack 0.2219）")

    # F2 畸变：0° 图的 mm/px 随像高
    print("\n" + "=" * 100 + "\n[F2] 残余畸变：0° 图中 mm/px 随到主点距离的变化（纸在画面一侧，覆盖 r/r_max 约 0.2–0.8）\n" + "=" * 100)
    for name, d in per_img.items():
        if d["th"] != 0.0 or not d["bands"]: continue
        cx, cy = (d["W"] - 1) / 2, (d["H"] - 1) / 2; rmax = np.hypot(cx, cy)
        rr = np.array([np.hypot(b["x_img"] - cx, b["y_img"] - cy) / rmax for b in d["bands"]]); mm = np.array([b["mmpx"] for b in d["bands"]])
        k = np.polyfit(rr ** 2, mm / np.median(mm), 1)
        print(f"  {name}: r/r_max ∈ [{rr.min():.2f},{rr.max():.2f}]  mm/px 相对变化 (max−min)/中值 = {100*(mm.max()-mm.min())/np.median(mm):.2f}%  拟合 s(r)/s0 ≈ 1 + ({k[0]:+.4f})·(r/r_max)²  → 画面角落 (r=1) 等效 κ ≈ {1+k[0]:.3f}")
    print("  判读：与 θ=5° 的 [0.90,1.14] 相比量级；拟合系数即残余 Brown 一阶项的量级（含条带测量噪声）。")

    # F3 / F4 SAM3
    if model is not None:
        import torch, torch.nn.functional as F
        print("\n" + "=" * 100 + "\n[F3] SAM3 零样本宽度：标称尺度 vs 倾角修正；w_θ/w_0 实测比 vs 几何预测\n" + "=" * 100)
        print(f"  {'file':9s}{'θ':>5s}{'Dice-':>6s}{'px中值':>8s}{'px P95':>8s}{'mm标称(med/p95)':>17s}{'mm修正(med/p95)':>17s}{'实测比w_θ/w_0':>13s}{'几何预测比':>10s}{'ASTM P95>10':>12s}{'JTG med>3':>10s}")
        w0_of = {}
        clicks, click_src = load_clicks()
        print(f"  人工点选文件：{click_src if clicks else '无（F4 跳过）'}")
        for name, d in per_img.items():
            try:
                img = d["img"]; W, H = d["W"], d["H"]
                mp = os.path.join(MSK, name.replace(".jpg", ".npy"))
                if os.path.exists(mp):
                    m = np.load(mp)
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        m = E.sem_mask(model, processor, img)
                    np.save(mp, m)
                m_raw = m.copy()        # F4 剖面用原始掩膜（不抠粉笔、不抠纸）
                arr = np.asarray(img).astype(np.int16); lum = arr.mean(2); sat = arr.max(2) - arr.min(2)
                white = (lum > 165) & (sat < 50)
                for p in d["papers"]:
                    y0, x0, y1, x1 = p["bbox"]; pad = 40
                    m[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = False
                m &= ~morphology.binary_dilation(white, morphology.disk(3))
                lab = measure.label(m);
                if lab.max() > 0:
                    areas = np.bincount(lab.ravel()); areas[0] = 0; keep = areas >= 0.05 * areas.max(); m = keep[lab]
                if m.sum() < 100:
                    print(f"  {name:9s} 掩膜为空"); continue
                sk = skeletonize(m); dist = ndi.distance_transform_edt(m); pts = np.argwhere(sk); wpx = 2 * dist[sk]
                s0 = d.get("s0", 670 / d["f"]); th = np.radians(d.get("sign", 1.0) * d["th"])
                phi = np.arctan((pts[:, 0] - (H - 1) / 2) / d["f"])
                kap = kappa_across(phi, th)                  # 裂缝沿长边走向，宽度方向 ≈ 横向
                med_px, p95_px = float(np.median(wpx)), float(np.percentile(wpx, 95))
                mm_nom = (med_px * s0, p95_px * s0); mm_cor = (float(np.median(wpx * s0 * kap)), float(np.percentile(wpx * s0 * kap, 95)))
                pred_ratio = float(np.median(1.0 / kap))
                if d["th"] == 0.0: w0_of[d["crack"]] = med_px
                ratio = med_px / w0_of[d["crack"]] if d["crack"] in w0_of else np.nan
                wrows.append(dict(file=name, crack=d["crack"], tilt_deg=d["th"], med_px=med_px, p95_px=p95_px, mm_nom_med=mm_nom[0], mm_nom_p95=mm_nom[1], mm_cor_med=mm_cor[0], mm_cor_p95=mm_cor[1], ratio=ratio, pred_ratio=pred_ratio, s0=s0))
                print(f"  {name:9s}{d['th']:5.1f}{'':>6s}{med_px:8.1f}{p95_px:8.1f}{mm_nom[0]:8.2f}/{mm_nom[1]:<8.2f}{mm_cor[0]:8.2f}/{mm_cor[1]:<8.2f}{ratio:13.3f}{pred_ratio:10.3f}{str(mm_nom[1] > 10):>12s}{str(mm_nom[0] > 3):>10s}")
                # F4：人工点选的刻度位置（仅 0° 图）——同一位置三方对照：卡尺 / 人工图像开口 / SAM3 缝芯
                cal = [float(r_[k]) for r_ in log if r_["file"] == name for k in ("w1_mm", "w2_mm", "w3_mm")]
                d["clicks"] = (clicks or {}).get(name, [])
                for (tick, x1, y1, x2, y2, man_px) in d["clicks"]:
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; L = np.hypot(x2 - x1, y2 - y1); ux, uy = (x2 - x1) / L, (y2 - y1) / L
                    # 局部裂缝走向：刻度 50–150 px 内骨架点的主方向（PCA）；法向 n 才是工程意义上的"宽度方向"
                    dd = np.hypot(pts[:, 0] - cy, pts[:, 1] - cx); ann = pts[(dd > 50) & (dd < 150)]
                    if len(ann) >= 10:
                        cvec = np.linalg.eigh(np.cov((ann - ann.mean(0)).T.astype(float)))[1][:, -1]   # (dy, dx)
                        vx, vy = float(cvec[1]), float(cvec[0])
                    else:
                        vx, vy = -uy, ux
                    nx, ny = -vy, vx                                   # 法向
                    cosang = abs(ux * nx + uy * ny); ang = float(np.degrees(np.arccos(min(1.0, cosang))))   # 人工连线与法向夹角
                    # 粉笔横划压在裂缝上（掩膜在刻度处被遮断），故 SAM3 不在刻度正下方量，而在沿裂缝上下各偏 60–150 px 处取法向弦
                    pr = []
                    for t_ in list(range(-150, -59, 10)) + list(range(60, 151, 10)):
                        w_, o_ = mask_width_along(m_raw, cx + t_ * vx, cy + t_ * vy, nx, ny, search=60)
                        if np.isfinite(w_): pr.append(w_)
                    prof_px = float(np.median(pr)) if pr else np.nan; off = float(len(pr))   # off 列 = 有效弦条数（共 20 条）
                    # (b) EDT 骨架局部宽度：距刻度 50–150 px 的骨架点 2·EDT 中值（与 F3/论文一致的量法，避开粉笔）
                    loc = wpx[(dd > 50) & (dd < 150)]
                    edt_px = float(np.median(loc)) if loc.size else np.nan
                    # 人工连线内掩膜像素占比（SAM3 覆盖了开口的多大比例）
                    tt = np.linspace(0, 1, int(L) + 1); xs = np.clip(np.round(x1 + tt * (x2 - x1)).astype(int), 0, W - 1); ys = np.clip(np.round(y1 + tt * (y2 - y1)).astype(int), 0, H - 1)
                    cover = float(m[ys, xs].mean())
                    prows.append(dict(file=name, crack=d["crack"], tilt_deg=d["th"], point=tick, x=int(cx), y=int(cy),
                                      caliper_mm=cal[tick - 1] if tick - 1 < len(cal) else np.nan,
                                      manual_px=man_px, manual_mm=man_px * s0, angle_deg=ang, manual_perp_px=man_px * cosang, manual_perp_mm=man_px * cosang * s0,
                                      sam3_prof_px=prof_px, sam3_prof_mm=prof_px * s0, sam3_offset_px=off,
                                      sam3_edt_px=edt_px, sam3_edt_mm=edt_px * s0, sam3_cover_frac=cover, dir_vx=vx, dir_vy=vy))
                # 叠加图
                ov = img.resize((W // DS, H // DS)); dr = ImageDraw.Draw(ov)
                cont = measure.find_contours(m[::DS, ::DS].astype(float), 0.5)
                for cc in cont[:200]:
                    if len(cc) > 20: dr.line([(float(x), float(y)) for y, x in cc[::3]], fill=(255, 0, 0), width=2)
                for p in d["papers"]:
                    y0, x0, y1, x1 = [v / DS for v in p["bbox"]]; dr.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
                for (tick, x1, y1, x2, y2, _) in d["clicks"]:
                    dr.line([(x1 / DS, y1 / DS), (x2 / DS, y2 / DS)], fill=(255, 255, 0), width=3)
                    dr.ellipse([(x1 + x2) / 2 / DS - 12, (y1 + y2) / 2 / DS - 12, (x1 + x2) / 2 / DS + 12, (y1 + y2) / 2 / DS + 12], outline=(255, 255, 0), width=2)
                    dr.text(((x1 + x2) / 2 / DS + 16, (y1 + y2) / 2 / DS - 8), str(tick), fill=(255, 255, 0))
                ov.save(os.path.join(OVD, name.replace(".jpg", "_ov.jpg")), quality=80)
                # 刻度点局部放大图：400×400 原始像素放大 2 倍（800×800），便于论文插图 / 人工复核
                Z = 2
                for (tick, x1, y1, x2, y2, _) in d["clicks"]:
                    qs = [q for q in prows if q["file"] == name and q["point"] == tick]
                    if not qs: continue
                    q_ = qs[0]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2); r_ = 200
                    box = (max(0, cx - r_), max(0, cy - r_), min(W, cx + r_), min(H, cy + r_))
                    cr = img.crop(box).convert("RGB"); cr = cr.resize((cr.width * Z, cr.height * Z), Image.LANCZOS); dc = ImageDraw.Draw(cr)
                    P = lambda x, y: ((x - box[0]) * Z, (y - box[1]) * Z)
                    sub = m[box[1]:box[3], box[0]:box[2]]
                    for cc in measure.find_contours(sub.astype(float), 0.5):
                        if len(cc) > 10: dc.line([((x + 0.5) * Z, (y + 0.5) * Z) for y, x in cc], fill=(255, 0, 0), width=3)
                    # 骨架（绿）与内切圆（青，每隔 25 个骨架点一个，半径 = EDT）
                    ssk = sk[box[1]:box[3], box[0]:box[2]]; sdist = dist[box[1]:box[3], box[0]:box[2]]
                    spts = np.argwhere(ssk)
                    for (yy, xx) in spts: dc.ellipse([xx * Z - 1, yy * Z - 1, xx * Z + 2, yy * Z + 2], fill=(0, 255, 0))
                    for (yy, xx) in spts[::25]:
                        rr = float(sdist[yy, xx]) * Z
                        if rr >= 2: dc.ellipse([xx * Z - rr, yy * Z - rr, xx * Z + rr, yy * Z + rr], outline=(0, 255, 255), width=2)
                    # 实际用于 SAM3 的法向弦（橙色），刻度上下 60–150 px
                    vx_, vy_ = q_["dir_vx"], q_["dir_vy"]; nx_, ny_ = -vy_, vx_
                    for t_ in (-150, -120, -90, -60, 60, 90, 120, 150):
                        px_, py_ = cx + t_ * vx_, cy + t_ * vy_
                        tt_ = np.arange(-200, 201); xs_ = np.clip(np.round(px_ + tt_ * nx_).astype(int), 0, W - 1); ys_ = np.clip(np.round(py_ + tt_ * ny_).astype(int), 0, H - 1)
                        idx = np.where(m_raw[ys_, xs_])[0]
                        if idx.size:
                            segs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1); seg = min(segs, key=lambda s_: abs((s_[0] + s_[-1]) / 2 - 200))
                            if abs((seg[0] + seg[-1]) / 2 - 200) <= 60:
                                dc.line([P(xs_[seg[0]], ys_[seg[0]]), P(xs_[seg[-1]], ys_[seg[-1]])], fill=(255, 140, 0), width=4)
                    dc.line([P(x1, y1), P(x2, y2)], fill=(255, 255, 0), width=4)
                    # 裂缝走向（白色短线）
                    dc.line([P(cx - 40 * vx_, cy - 40 * vy_), P(cx + 40 * vx_, cy + 40 * vy_)], fill=(255, 255, 255), width=2)
                    dc.rectangle([0, 0, 400 * Z, 40], fill=(0, 0, 0))
                    dc.text((6, 4), f"{name} t{tick}  caliper {q_['caliper_mm']:.1f}  manual {q_['manual_mm']:.1f} (perp {q_['manual_perp_mm']:.1f}, angle {q_['angle_deg']:.0f} deg)  SAM3 chord {q_['sam3_prof_mm']:.1f}  EDT {q_['sam3_edt_mm']:.1f} mm", fill=(255, 255, 255))
                    dc.text((6, 22), "red=SAM3 mask  yellow=manual line  white=crack direction  orange=normal chords used  green=skeleton  cyan=inscribed circles (EDT)", fill=(255, 255, 255))
                    cr.save(os.path.join(OVD, name.replace(".jpg", f"_t{tick}.png")))
            except Exception:
                print(f"  !! {name}"); traceback.print_exc(); torch.cuda.empty_cache()
        print("\n  判读：'mm修正' 用 s0·κ_across(φ) 逐骨架点修正；若实测比接近几何预测比且修正后 mm 跨角度稳定，几何项得到实拍验证。")

        print("\n" + "=" * 100 + "\n[F4] 人工定点（0° 图，15 点）三方对照：卡尺 / 人工图像开口 / SAM3 缝芯（同一位置、同一方向）\n" + "=" * 100)
        print("  走向 = 刻度 50–150 px 内骨架点 PCA 主方向；夹角 = 人工连线与裂缝法向的夹角；人工⊥ = 人工连线投影到法向（消除斜量）")
        print("  SAM3弦 = 刻度上下各偏 60–150 px（避开粉笔）沿法向的原始掩膜连续段长度中值；SAM3-EDT = 同一环带骨架点 2·EDT 中值（论文量法）；有效条 = 取到掩膜段的弦数/20")
        print(f"  {'file':9s}{'点':>3s}{'卡尺mm':>7s}{'人工mm':>7s}{'夹角°':>6s}{'人工⊥mm':>8s}{'SAM3弦px':>9s}{'弦mm':>6s}{'EDTpx':>7s}{'EDTmm':>6s}{'有效条':>7s} | {'人工⊥/卡尺':>9s}{'弦/卡尺':>8s}{'EDT/卡尺':>8s}{'弦/人工⊥':>8s}")
        for q in prows:
            print(f"  {q['file']:9s}{q['point']:3d}{q['caliper_mm']:7.1f}{q['manual_mm']:7.2f}{q['angle_deg']:6.0f}{q['manual_perp_mm']:8.2f}{q['sam3_prof_px']:9.1f}{q['sam3_prof_mm']:6.2f}{q['sam3_edt_px']:7.1f}{q['sam3_edt_mm']:6.2f}{q['sam3_offset_px']:7.0f} | "
                  f"{q['manual_perp_mm']/q['caliper_mm']:9.2f}{q['sam3_prof_mm']/q['caliper_mm']:8.2f}{q['sam3_edt_mm']/q['caliper_mm']:8.2f}{q['sam3_prof_mm']/q['manual_perp_mm']:8.2f}")
        z = [q for q in prows if np.isfinite(q["caliper_mm"]) and np.isfinite(q["manual_mm"])]
        if len(z) >= 5:
            def stat(a, b, lab):
                a, b = np.asarray(a, float), np.asarray(b, float); ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]; lr = np.log(a / b)
                print(f"  {lab:14s} n={len(a):2d}  比值中值 {np.median(a/b):.2f}  均值比 {a.mean()/b.mean():.2f}  log b={lr.mean():+.3f} σ={lr.std(ddof=1):.3f}  Spearman ρ={spearmanr(a, b).correlation:.3f}  MAE {np.mean(np.abs(a-b)):.2f} mm")
            c = [q["caliper_mm"] for q in z]; mn = [q["manual_mm"] for q in z]; mp = [q["manual_perp_mm"] for q in z]; sp = [q["sam3_prof_mm"] for q in z]; se = [q["sam3_edt_mm"] for q in z]
            print()
            stat(mn, c, "人工(原)/卡尺"); stat(mp, c, "人工⊥/卡尺"); stat(sp, c, "SAM3弦/卡尺"); stat(se, c, "SAM3-EDT/卡尺"); stat(sp, mp, "SAM3弦/人工⊥"); stat(se, mp, "SAM3-EDT/人工⊥"); stat(sp, se, "SAM3弦/EDT")
            ang = np.array([q["angle_deg"] for q in z]); cov = np.array([q["sam3_cover_frac"] for q in z]); miss = int(np.sum(~np.isfinite([q["sam3_prof_px"] for q in z])))
            print(f"  人工连线与法向夹角：中值 {np.median(ang):.0f}°，最大 {ang.max():.0f}°；SAM3 对人工开口（粉笔处）的覆盖率中值 {np.median(cov):.2f}；两侧 20 条弦全部无掩膜的点 {miss}/{len(z)}")
            print(f"  参照：SUT-Crack 上 SAM3 相对 GT 掩膜 偏差 +42.4%，b=+0.412，σ=0.283")
            print("  判读：三个量是三种定义。人工/卡尺≈2 说明卡尺读数偏向缝内、人工描边取外缘；SAM3/人工≪1 说明零样本掩膜只覆盖深色缝芯。")
            print("        与 SUT 的 +42% 方向相反，说明 b 的符号由参考约定决定——mm 级宽度只在对某一标注约定校准后才有意义（支持 A2：b 不跨约定迁移）。")

    for path, rows_ in [(CSV_B, brows), (CSV_W, wrows), (CSV_P, prows)]:
        if rows_:
            with open(path, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows_[0].keys())); w.writeheader(); w.writerows(rows_)
    print(f"\nCSV: {CSV_B}\n     {CSV_W}\n     {CSV_P}\n叠加图 {OVD}\n总耗时 {(time.time()-T0)/60:.1f} min")


if __name__ == "__main__":
    main()