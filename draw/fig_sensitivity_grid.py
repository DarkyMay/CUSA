#!/usr/bin/env python3
"""
fig_sensitivity_grid.py

生成图: 分级敏感性热力图。
同一批 126 条线状裂缝, 仅因掩码来源与缝宽口径不同, ASTM L 级占比
从 0.0% 摆到 100.0%; 末行为参照掩码, 显示即便分割无误差, 口径仍能
使占比从 85.7% 降到 3.2%。

数据来自 results/closeloop_v2_20260726_170420.csv 的汇总。
输出与本脚本同目录。
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig_sensitivity_grid.png")

REDUCTIONS = ["median", "mean", "$P_{95}$", "max"]

# ASTM: 缝宽 <= 10mm 判为 L 级的样本占比 (%)
ROWS = [
    ("zero-shot, full",  [60.3, 56.3,  5.6,  0.0]),
    ("zero-shot, crop",  [84.1, 84.1, 37.3,  9.5]),
    ("zero-shot, box",   [47.6, 46.8,  7.1,  0.0]),
    ("fine-tuned, full", [100.0, 99.2, 81.7, 37.3]),
    ("fine-tuned, crop", [100.0, 99.2, 88.9, 54.0]),
    ("fine-tuned, box",  [99.2, 99.2, 79.4, 36.5]),
    ("reference mask",   [85.7, 86.5, 27.8,  3.2]),
]

labels = [r for r, _ in ROWS]
data = np.array([v for _, v in ROWS])

fig, ax = plt.subplots(figsize=(6.2, 3.6))
im = ax.imshow(data, cmap="viridis", vmin=0, vmax=100, aspect="auto")

ax.set_xticks(range(len(REDUCTIONS)), REDUCTIONS, fontsize=11)
ax.set_yticks(range(len(labels)), labels, fontsize=11)
ax.set_xlabel("width statistic", fontsize=12)
ax.tick_params(length=0)

# 参照掩码与六个真实配置之间的分隔
ax.axhline(5.5, color="white", lw=3)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=11,
                color="white" if v < 55 else "#111111")

for s in ax.spines.values():
    s.set_visible(False)

cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.03)
cb.set_label("cracks graded low under ASTM (%)", fontsize=11)
cb.ax.tick_params(labelsize=10)

fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f">> {OUT}")