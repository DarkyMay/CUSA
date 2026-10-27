#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""几何量测量层 + 相对严重度分级  (RDD dash-cam 手标真值 39张)

【2026-07 修订说明 —— 相对旧版的三处改动】
 1. 删除 MM_PER_PIXEL=5.0 的假设标定，以及据此产生的 JTG/ASTM 绝对等级列。
    理由：RDD dash-cam 无任何标定依据，假设标定已验证会导致近乎全判"重度"，
    属备忘录第八节明列的 ❌ 已否决方向。绝对mm分级仅在有真实标定的
    SUT-Crack 上做(EXIF光学 0.2219 mm/px)。
 2. 阈值不再本地硬编码，一律取自 pavement_codes.py（旧版 JTG 的"中3-10mm"、
    ASTM 的"38mm"均为错值，后者实为滑移裂缝阈值）。
 3. 形态分类由 branch_ratio>0.15 改为环数判据(欧拉示性数)。旧判据基于分叉
    "像素"密度，而分叉像素数会把单个交叉重复计 2~3 次，且对掩码边缘毛糙
    极其敏感——SUT-Crack 上实测视觉线状裂缝也能有上百分叉像素。

本脚本产出的是【相对分级】，这是已验证可靠的部分；绝对规范映射见闭环脚本。
"""
import sys, os, csv, datetime, glob
from collections import Counter
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pavement_codes as pc

OUT = "/root/autodl-tmp/proj_bridge"
os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logp = os.path.join(OUT, "results", f"geometry_layer_{ts}.txt")
csvp = os.path.join(OUT, "results", f"geometry_measurements_{ts}.csv")


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()


sys.stdout = Tee(logp)

paths = sorted(glob.glob(OUT + "/rdd_clean/rdd_clean_masks_edited/*.png") +
               glob.glob(OUT + "/rdd_tolabel/masks_edited/*.png"))
print("=" * 78)
print(f"几何量测量层 | dash-cam 手标真值 n={len(paths)}")
print("本脚本只做【相对严重度分级】。绝对mm/规范映射需真实标定，见 SUT-Crack 闭环脚本。")
print("=" * 78)

rows = []
for p in paths:
    m = np.array(Image.open(p).convert("L")) > 127
    g = pc.measure(m)                      # 不传 mm_per_px —— 无标定，只出像素量
    if g is None:
        continue
    g["file"] = os.path.basename(p)
    rows.append(g)

if not rows:
    print("!! 未读到任何掩码，检查路径"); sys.exit(1)

# ---- 相对分级（缝宽均值分位数） ----
levels, t33, t66 = pc.relative_levels([r["w_mean"] for r in rows])
for r, lv in zip(rows, levels):
    r["rel_level"] = lv

print(f"\n相对分级阈值(缝宽均值,像素): 轻<{t33:.2f}  中[{t33:.2f},{t66:.2f})  重>={t66:.2f}\n")
print(f"{'文件':<24}{'类型':>5}{'环数':>5}{'缝宽均px':>10}{'缝宽中位px':>11}"
      f"{'面积':>8}{'密度%':>7}{'相对':>5}")
for r in sorted(rows, key=lambda x: x["w_mean"]):
    print(f"{r['file']:<24}{r['ctype']:>5}{r['cycles']:>5}{r['w_mean']:>10.2f}"
          f"{r['w_median']:>11.2f}{r['area_px']:>8}{r['density_pct']:>7.2f}{r['rel_level']:>5}")

print("\n" + "=" * 78)
print("分布汇总")
print("=" * 78)
print("  相对等级 :", dict(Counter(r["rel_level"] for r in rows)))
print("  形态类型 :", dict(Counter(r["ctype"] for r in rows)))
print(f"  环数分布 : 中位 {np.median([r['cycles'] for r in rows]):.0f}  "
      f"最大 {max(r['cycles'] for r in rows)}")
print(f"  缝宽均值(px): {np.mean([r['w_mean'] for r in rows]):.2f} "
      f"± {np.std([r['w_mean'] for r in rows]):.2f}  "
      f"极差 {max(r['w_mean'] for r in rows)-min(r['w_mean'] for r in rows):.2f}")

print("\n[规范映射说明]")
print("  本数据集无像素->mm标定，故不输出 JTG/ASTM 绝对等级。")
print(f"  参考：JTG 5210-2018 线状裂缝仅两级 轻度<=3mm / 重度>3mm；")
print(f"        ASTM D6433 纵横向裂缝 L<10mm / M 10~75mm / H>75mm；")
print(f"        3~10mm 为极端背离带(JTG判最高档、ASTM判最低档)。")

fields = ["file", "ctype", "cycles", "n_holes", "n_comp", "area_px", "density_pct",
          "n_skel", "L_edge", "n_end", "n_branch_px",
          "w_median", "w_mean", "w_p90", "w_p95", "w_max", "rel_level"]
with open(csvp, "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=fields)
    wr.writeheader()
    for r in rows:
        wr.writerow({k: r.get(k) for k in fields})

print(f"\n>> CSV : {csvp}")
print(f">> 日志: {logp}")