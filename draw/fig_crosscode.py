#!/usr/bin/env python3
"""
fig_crosscode.py

生成图: 126 条线状裂缝的 GT 缝宽分布与两国阈值。
JTG 在 3mm 以上判重, ASTM 在 10mm 以下判轻, 落在两者之间的裂缝
同时获得中国最高档与美国最低档。

数据来自 closeloop_v2_20260726_170420.csv 的 gt_w_median 列。
输出与本脚本同目录。
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig_crosscode.png")
CSV = "/root/autodl-tmp/proj_bridge/results/closeloop_v2_20260726_170420.csv"

JTG, ASTM = 3.0, 10.0
XMAX = 20.0
RED, BAND = "#c00000", "#e8a33d"

df = pd.read_csv(CSV)
lin = df[df["gt_ctype"] == "线状"].drop_duplicates("stem")
w = lin["gt_w_median"].values
n = len(w)
in_band = int(((w > JTG) & (w <= ASTM)).sum())
tail = int((w > XMAX).sum())

fig, ax = plt.subplots(figsize=(6.8, 3.6))

bins = np.arange(0, XMAX + 0.5, 0.5)
counts, edges = np.histogram(np.clip(w, 0, XMAX - 1e-6), bins=bins)
centers = (edges[:-1] + edges[1:]) / 2
colors = ["#e8a33d" if JTG < c <= ASTM else "#9a9a9a" for c in centers]
ax.bar(centers, counts, width=0.46, color=colors, edgecolor="white",
       linewidth=.4, alpha=.85)
ymax = ax.get_ylim()[1]
ax.set_ylim(0, ymax * 1.18)
top = ymax * 1.18

# 阈值虚线, 标签置于线的正上方
for t, lab in [(JTG, "JTG\n3 mm"), (ASTM, "ASTM\n10 mm")]:
    ax.axvline(t, color=RED, lw=1.3, ls="--", ymax=.80)
    ax.text(t, top * .82, lab, ha="center", va="bottom",
            fontsize=11, color=RED, linespacing=1.3)


# 说明文字移到右上角空白处
ax.text(XMAX * .985, top * .74,
        "severe under JTG\nand low under ASTM\n"
        f"{in_band}/{n} = {100 * in_band / n:.1f} %",
        ha="right", va="top", fontsize=11, color="#8a6414",
        linespacing=1.55)

if tail:
    lab = "crack" if tail == 1 else "cracks"
    ax.text(XMAX * .985, top * .30,
            f"{tail} {lab} above {XMAX:.0f} mm",
            ha="right", va="center", fontsize=10, color="#666666")

ax.set_xlabel("crack width, median reduction (mm)", fontsize=12)
ax.set_ylabel("cracks", fontsize=12)
ax.set_xlim(0, XMAX)
ax.tick_params(labelsize=10.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f">> {OUT}")
print(f"   n={n}  in band: {in_band} ({100 * in_band / n:.1f}%)  "
      f"above {XMAX:.0f}mm: {tail}")