#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e6_yolo_mask_source.py —— 非 SAM 掩膜源（YOLO26-seg-s，Roboflow / CV12K 训练）在 SUT-Crack 上的宽度行为（GPU）

目的
  1. 给敏感性网格增加与 SAM 无关的掩膜源（审稿 M10：掩膜源这一轴不应只有一个模型家族）
  2. 跨模型验证"以输出格子为单位的固定膨胀"机制：
     ultralytics 的 mask 原型分辨率 = imgsz/4，retina_masks=True 时上采样到原图，
     原图上的格子 = 4032/(imgsz/4) px：640→25.2px(5.6mm) 1280→12.6px(2.8mm) 2016→8px(1.8mm) 4032→4px(0.9mm)
     若机制普适，截距 a（或模糊地板 f）应随 imgsz 增大而下降；SAM3 参照：格子 10.5×14 px，a=4.61 mm，f=7.23 mm
用法：python3 e6_yolo_mask_source.py --limit 10   # 冒烟
      python3 e6_yolo_mask_source.py              # 全量，约 15–25 min
输出：results/e6_yolo_<ts>.txt, results/e6_yolo_perimage_<ts>.csv
"""
import os, sys, csv, re, time, argparse, datetime
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.stats import spearmanr
from collections import Counter
from ultralytics import YOLO

sys.path.insert(0, "/root/autodl-tmp/proj_bridge")
import pavement_codes as pc

Image.MAX_IMAGE_PIXELS = None
OUT = "/root/autodl-tmp/proj_bridge"
ROOT = "/root/autodl-tmp/SUT-Crack"
CL = os.path.join(OUT, "results", "closeloop_v2_20260726_170420.csv")
WEIGHTS = [("yolo_rob", "/root/autodl-tmp/crack-yolo-seg-best.pt"),
           ("yolo_cv12k", "/root/autodl-tmp/CrackVision12K-yolo-seg-best.pt")]
SIZES = [640, 1280, 2016, 4032]
CONF = 0.25
MMPX = pc.SUTCRACK_MMPX
STATS4 = ["w_median", "w_mean", "w_p95", "w_max"]
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"e6_yolo_{ts}.txt")
CSVP = os.path.join(OUT, "results", f"e6_yolo_perimage_{ts}.csv")


def _resolve(*c):
    for x in c:
        if os.path.isdir(x): return x
    return c[0]
IMGD = _resolve(os.path.join(ROOT, "1-Segmentation", "Original_Image"), os.path.join(ROOT, "1-Segmentation", "Original Image"))
GTD = _resolve(os.path.join(ROOT, "1-Segmentation", "Ground_Truth"), os.path.join(ROOT, "1-Segmentation", "Ground Truth"))


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()


def numkey(s): m = re.findall(r'\d+', s); return int(m[0]) if m else 0
def fl(x):
    try: return float(x)
    except Exception: return np.nan
def dice_np(a, b): s = a.sum() + b.sum(); return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def geom(mask):
    if mask.sum() < 30: return None
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - 60), min(mask.shape[0], ys.max() + 61)
    x0, x1 = max(0, xs.min() - 60), min(mask.shape[1], xs.max() + 61)
    return pc.measure(mask[y0:y1, x0:x1], mm_per_px=MMPX)


def linfit(x, y):
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 5: return np.nan, np.nan, np.nan
    b, a = np.polyfit(x, y, 1); r2 = 1 - np.sum((y - a - b * x) ** 2) / np.sum((y - y.mean()) ** 2)
    return a, b, r2


def yolo_mask(model, img, imgsz):
    res = model.predict(img, imgsz=imgsz, conf=CONF, retina_masks=True, verbose=False, device=0, half=True)[0]
    Wd, Hd = img.size
    if res.masks is None or res.masks.data.shape[0] == 0:
        return np.zeros((Hd, Wd), bool), 0
    m = res.masks.data.any(0).cpu().numpy().astype(bool)
    if m.shape != (Hd, Wd):   # 保险：若未按原图尺寸返回则最近邻缩放
        m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((Wd, Hd), Image.NEAREST)) > 127
    return m, int(res.masks.data.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sizes", type=str, default=",".join(map(str, SIZES)))
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]
    sys.stdout = Tee(LOGP)
    T0 = time.time()
    print(f"e6_yolo_mask_source.py  {ts}  sizes={sizes} conf={CONF}  {torch.cuda.get_device_name(0)}")

    cl = [r for r in csv.DictReader(open(CL, encoding="utf-8"))
          if r["gt_ctype"] == "线状" and r["variant"] == "zeroshot" and r["cond"] == "text_full"]
    gtw = {r["stem"]: r for r in cl}
    stems = sorted(gtw, key=numkey)
    if args.limit: stems = stems[:args.limit]
    G = {st: np.array([fl(gtw[s]["gt_" + st]) for s in stems]) for st in STATS4}
    print(f"  线状图像 {len(stems)} 张")
    GT = {}
    for s in stems:
        GT[s] = np.array(Image.open(os.path.join(GTD, s + ".png")).convert("L")) > 127

    rows = []; summary = []
    for wname, wpath in WEIGHTS:
        model = YOLO(wpath)
        for sz in sizes:
            cell_x, cell_y = 3024 / (sz / 4), 4032 / (sz / 4)
            print(f"\n[{wname} imgsz={sz}] 格子 ≈ {cell_x:.1f}×{cell_y:.1f} px = {cell_x*MMPX:.2f}×{cell_y*MMPX:.2f} mm", flush=True)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time(); d, fg, ninst, P = [], [], [], {st: [] for st in STATS4}
            ok_run = True
            for i, s in enumerate(stems):
                img = ImageOps.exif_transpose(Image.open(os.path.join(IMGD, s + ".jpg"))).convert("RGB")
                gt = GT[s]
                try:
                    m, n = yolo_mask(model, img, sz)
                except torch.cuda.OutOfMemoryError:
                    print(f"  !! OOM at imgsz={sz}，跳过该尺寸"); ok_run = False; torch.cuda.empty_cache(); break
                except Exception as e:
                    print(f"  !! {s}: {type(e).__name__}: {str(e)[:100]}"); m, n = np.zeros(gt.shape, bool), 0
                g = geom(m)
                d.append(dice_np(m, gt)); fg.append(m.mean() / gt.mean()); ninst.append(n)
                for st in STATS4: P[st].append(g[st + "_mm"] if g else np.nan)
                rows.append(dict(source=wname, imgsz=sz, stem=s, dice=d[-1], fg_ratio=fg[-1], n_inst=n,
                                 **{f"gt_{st}": G[st][i] for st in STATS4}, **{f"pred_{st}": P[st][-1] for st in STATS4}))
                if (i + 1) % 30 == 0: print(f"  ...{i+1}/{len(stems)} {(time.time()-t0)/60:.1f} min", flush=True)
            if not ok_run: continue
            d, fg = np.array(d), np.array(fg); P = {k: np.array(v) for k, v in P.items()}
            rec = dict(source=wname, imgsz=sz, cell_x=cell_x, cell_y=cell_y, dice=d.mean(), fg=np.nanmean(fg),
                       n_empty=int(np.isnan(P["w_median"]).sum()), n_inst=np.mean(ninst),
                       mem=torch.cuda.max_memory_allocated() / 2 ** 30, sec=(time.time() - t0) / len(stems))
            for st in ("w_median", "w_p95"):
                ok = np.isfinite(P[st]); g_, p_ = G[st][ok], P[st][ok]
                a, b, r2 = linfit(g_, p_); fa, fk, _ = linfit(g_ ** 2, p_ ** 2)
                rec[st] = dict(bias=(p_.mean() - g_.mean()) / g_.mean() if len(g_) else np.nan,
                               mae=np.mean(np.abs(p_ - g_)) if len(g_) else np.nan,
                               rho=spearmanr(p_, g_).correlation if len(g_) > 4 else np.nan,
                               a=a, b=b, r2=r2, f=np.sqrt(max(fa, 0)) if np.isfinite(fa) else np.nan, k=fk)
            # 等级比例（含空掩膜按缺失剔除）
            rec["astm_low"] = {st: np.nanmean(P[st] <= 10.0) for st in STATS4}
            rec["jtg_sev"] = {st: np.nanmean(P[st] > 3.0) for st in STATS4}
            summary.append(rec)
            print(f"  Dice {rec['dice']:.4f}  前景比 {rec['fg']:.2f}  空掩膜 {rec['n_empty']}  实例/图 {rec['n_inst']:.1f}  "
                  f"显存 {rec['mem']:.1f} GB  {rec['sec']:.2f} s/图")
            for st in ("w_median", "w_p95"):
                r = rec[st]
                print(f"    {st:<9} bias {r['bias']:+.1%}  MAE {r['mae']:.2f}  ρ {r['rho']:.3f}  a {r['a']:.2f}  b {r['b']:.2f}  R2 {r['r2']:.2f}  f {r['f']:.2f}")
        del model; torch.cuda.empty_cache()

    if rows:
        with open(CSVP, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    print("\n" + "=" * 100); print("[汇总 1] 格子大小 vs 截距/地板（跨模型验证；SAM3 参照来自 closeloop_v2 + width_mechanism）"); print("=" * 100)
    print(f"  {'来源':<12}{'imgsz':>6}{'格子y px':>9}{'格子y mm':>9}{'Dice':>7}{'前景比':>7}{'空':>4}"
          f"{'median a':>10}{'b':>6}{'f':>6}{'bias':>8}{'ρ':>7}{'p95 a':>8}{'b':>6}{'f':>6}{'bias':>8}{'ρ':>7}")
    print(f"  {'SAM3 zs/full':<12}{'1008':>6}{14.0:>9.1f}{14.0*MMPX:>9.2f}{0.7587:>7.4f}{1.40:>7.2f}{0:>4}"
          f"{4.61:>10.2f}{0.79:>6.2f}{7.23:>6.2f}{'+42.4%':>8}{0.758:>7.3f}{6.21:>8.2f}{0.70:>6.2f}{11.34:>6.2f}{'+14.8%':>8}{0.773:>7.3f}")
    for rec in summary:
        m, p = rec["w_median"], rec["w_p95"]
        print(f"  {rec['source']:<12}{rec['imgsz']:>6}{rec['cell_y']:>9.1f}{rec['cell_y']*MMPX:>9.2f}{rec['dice']:>7.4f}{rec['fg']:>7.2f}{rec['n_empty']:>4}"
              f"{m['a']:>10.2f}{m['b']:>6.2f}{m['f']:>6.2f}{m['bias']:>+8.1%}{m['rho']:>7.3f}{p['a']:>8.2f}{p['b']:>6.2f}{p['f']:>6.2f}{p['bias']:>+8.1%}{p['rho']:>7.3f}")
    print("  判读：同一权重下，若 a/f 随格子缩小而单调下降 → 格子机制跨模型成立；若 Dice 也随之下降 → 尺度冲突再现。")
    print("        跨域（Roboflow/CV12K→SUT）掩膜质量本身可能较差，Dice<0.5 的行其宽度数字仅作网格用途，不作机制证据。")

    print("\n[汇总 2] 敏感性网格新增行：ASTM 判低比例 / JTG 判重比例（四种归约；空掩膜剔除）")
    print(f"  {'来源':<12}{'imgsz':>6}" + "".join(f"{'ASTM_L '+st[2:]:>13}" for st in STATS4) + "".join(f"{'JTG重 '+st[2:]:>12}" for st in STATS4))
    print(f"  {'参考掩膜':<12}{'':>6}" + "".join(f"{np.mean(G[st]<=10):>13.1%}" for st in STATS4) + "".join(f"{np.mean(G[st]>3):>12.1%}" for st in STATS4))
    for rec in summary:
        print(f"  {rec['source']:<12}{rec['imgsz']:>6}" + "".join(f"{rec['astm_low'][st]:>13.1%}" for st in STATS4)
              + "".join(f"{rec['jtg_sev'][st]:>12.1%}" for st in STATS4))
    print(f"\n>> 逐图CSV: {CSVP}\n>> 日志: {LOGP}\n总耗时 {(time.time()-T0)/60:.1f} min")


if __name__ == "__main__":
    main()