#!/usr/bin/env python3
"""
fig_rank_disagreement.py  (2026-09 版)

六个配置按 Dice（左）和中位缝宽 MAE（右）排序，同一配置用线相连。
标签与数值同一行、与节点垂直对齐；fine-tuned 三个配置用空心灰点并注明斜率 b（宽度不响应真实宽度）。
数据来源: Table widthscale（closeloop_v2_20260726_170420.csv）
输出与本脚本同目录。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig_rank_disagreement.png")

# name, Dice, MAE(mm), slope b, fine-tuned?
CFG = [
    ("zero-shot, full",  0.759, 3.21, 0.78, False),
    ("zero-shot, crop",  0.751, 1.95, 0.69, False),
    ("zero-shot, box",   0.744, 3.86, 0.75, False),
    ("fine-tuned, full", 0.614, 2.59, 0.16, True),
    ("fine-tuned, box",  0.579, 2.40, 0.28, True),
    ("fine-tuned, crop", 0.555, 4.07, 0.15, True),
]
BLUE, ORANGE, GREY = "#1f77b4", "#d95f02", "#9a9a9a"

by_dice = sorted(CFG, key=lambda c: -c[1])
by_mae = sorted(CFG, key=lambda c: c[2])
rank_d = {c[0]: i for i, c in enumerate(by_dice)}
rank_m = {c[0]: i for i, c in enumerate(by_mae)}
n = len(CFG)

fig, ax = plt.subplots(figsize=(7.2, 4.2))
xl, xr = 0.0, 1.0
for name, dice, mae, b, ft in CFG:
    y0, y1 = n - 1 - rank_d[name], n - 1 - rank_m[name]
    if ft:
        col, lw, ls = GREY, 1.6, "--"
    else:
        col = BLUE if y1 > y0 else (ORANGE if y1 < y0 else GREY)
        lw, ls = 2.2, "-"
    ax.plot([xl, xr], [y0, y1], color=col, lw=lw, ls=ls, zorder=1)
    mk = dict(marker="o", ms=8, mfc="white", mec=col if not ft else GREY, mew=1.8, zorder=3, ls="none")
    ax.plot([xl], [y0], **mk); ax.plot([xr], [y1], **mk)
    # 左标签：名称 + Dice，同一行，右对齐到节点左侧
    ax.text(xl - 0.07, y0, f"{name}  {dice:.3f}", ha="right", va="center", fontsize=10.5,
            color="#333333" if not ft else GREY)
    # 右标签：MAE + 名称（fine-tuned 附斜率），左对齐到节点右侧
    tag = f"{mae:.2f} mm  {name}" + (f"  (b = {b:.2f})" if ft else "")
    ax.text(xr + 0.07, y1, tag, ha="left", va="center", fontsize=10.5,
            color="#333333" if not ft else GREY)

ax.text(xl, -1.0, "ranked by Dice", ha="center", va="top", fontsize=12)
ax.text(xr, -1.0, "ranked by width error", ha="center", va="top", fontsize=12)
ax.set_xlim(-0.95, 2.05)
ax.set_ylim(-1.4, n - 0.4)
ax.axis("off")
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f">> {OUT}")