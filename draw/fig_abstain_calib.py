#!/usr/bin/env python3
"""
fig_abstain_calib.py

生成图: 弃权曲线 + 标定成本(Brier 与 ECE 双 y 轴)。
数据取自 results/prob_grading_v2_0728_2124.txt 的 V2 与 V3 两节。
输出与本脚本同目录。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "fig_abstain_calib.png")

BLUE, ORANGE, RED, GREY = "#1f77b4", "#ff7f0e", "#c00000", "#666666"

# V2 弃权收益: (弃权比例%, 保留准确率)
ABSTAIN = [(0, .8240), (10, .8555), (20, .8817), (30, .9033), (50, .9392)]
# V3 标定成本: (标定张数, Brier, ECE)
CALIB = [(5, .1520, .1380), (10, .1385, .1147), (15, .1370, .1129),
         (20, .1349, .1108), (30, .1354, .1132), (40, .1339, .1185),
         (60, .1333, .1264)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.3))

# ---- 左: 弃权曲线 ----
x, y = zip(*ABSTAIN)
ax1.plot(x, [v * 100 for v in y], "o-", color=BLUE, lw=1.8, ms=6,
         label="accuracy on retained cases")
ax1.axhline(82.40, color="#999999", lw=1.0, ls=":",
            label="no abstention (82.4 %)")
ax1.axvline(30, color=RED, lw=1.1, ls="--")
ax1.annotate("30 %\n90.3 %", (30, 90.33), xytext=(33, 86.8),
             fontsize=11, color=RED, linespacing=1.4)
ax1.set_xlabel("cases withheld for review (%)", fontsize=12)
ax1.set_ylabel("accuracy on retained cases (%)", fontsize=12)
ax1.set_xlim(-2, 54)
ax1.set_ylim(80, 96)
ax1.tick_params(labelsize=10.5)
ax1.legend(fontsize=9.5, loc="upper left", frameon=False,
           handlelength=1.8, borderaxespad=.2)

# ---- 右: 标定成本 ----
n, brier, ece = zip(*CALIB)
l1, = ax2.plot(n, brier, "o-", color=BLUE, lw=1.8, ms=6, label="Brier score")
ax2.set_xlabel("calibration images", fontsize=12)
ax2.set_ylabel("Brier score", fontsize=12, color=BLUE)
ax2.tick_params(axis="y", labelcolor=BLUE, labelsize=10.5)
ax2.tick_params(axis="x", labelsize=10.5)
ax2.set_ylim(.128, .158)

ax3 = ax2.twinx()
l2, = ax3.plot(n, ece, "s--", color=ORANGE, lw=1.6, ms=5.5,
               label="expected calibration error")
ax3.set_ylabel("expected calibration error", fontsize=12, color=ORANGE)
ax3.tick_params(axis="y", labelcolor=ORANGE, labelsize=10.5)
ax3.set_ylim(.105, .147)

ax2.axvline(15, color=RED, lw=1.1, ls="--")
ax2.annotate("15 images", (15, .1425), xytext=(17, .1425),
             fontsize=11, color=RED, va="center")
ax2.legend(handles=[l1, l2], fontsize=9.5, loc="upper right",
           frameon=False, handlelength=1.8, borderaxespad=.3)

for a in (ax1, ax2, ax3):
    a.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(f">> {OUT}")