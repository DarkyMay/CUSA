#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e10_endtoend_comparison.py —— 端到端对比：现有"掩膜源 + 归约 + 硬阈值"管线 vs CUSA（无 GPU，秒级）

评价口径
  · 标准：ASTM D6433，T = 10 mm；参考等级 = 参考掩膜在标准规定归约（P95）下 > 10 mm
  · 所有管线在与 E1 相同的 200 次切分（42 标定 / 84 测试，SEED=20260728）的测试集上取均值，
    硬阈值管线不需要标定，但在相同测试子集上评估以保证可比
  · 指标：硬分级准确率、Brier（0/1 预测）、按 |w−T| 弃权 30% 后的准确率（不需标注即可做的拒答）
  · 附加列 "same-q"：以参考掩膜在该管线自身归约下的等级为参考的准确率（把归约差异与掩膜误差分开）
  · CUSA 行：ZS full + P95 + log 校准（复现 E1），及按 |P−0.5| 弃权 30%
输出：results/e10_endtoend_<ts>.txt
"""
import os, sys, csv, glob, datetime
import numpy as np
from scipy.stats import norm

OUT = "/root/autodl-tmp/proj_bridge"
CL = os.path.join(OUT, "results", "closeloop_v2_20260726_170420.csv")
E6 = sorted(glob.glob(os.path.join(OUT, "results", "e6_yolo_perimage_*.csv")))
SEED, NREP, T = 20260728, 200, 10.0
ABST = 0.30
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"e10_endtoend_{ts}.txt")


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()


def fl(x):
    try: return float(x)
    except Exception: return np.nan


def main():
    sys.stdout = Tee(LOGP)
    print(f"e10_endtoend_comparison.py  {ts}  ASTM T={T} mm  splits={NREP} (42/84)  seed={SEED}")
    rows = list(csv.DictReader(open(CL, encoding="utf-8")))
    lin = [r for r in rows if r["gt_ctype"] == "线状"]
    # 与 E1 相同的行序：zeroshot/text_full 线状行的 CSV 顺序
    base = [r for r in lin if r["variant"] == "zeroshot" and r["cond"] == "text_full" and r["pred_w_median"]]
    stems = [r["stem"] for r in base]; n = len(stems); idx = {s: i for i, s in enumerate(stems)}
    G = {q: np.array([fl(r["gt_" + q]) for r in base]) for q in ("w_median", "w_mean", "w_p95", "w_max")}
    y_ref = (G["w_p95"] > T).astype(int)                      # 参考等级（标准规定归约）
    print(f"  n={n}  参考越阈比例 {y_ref.mean():.1%}")

    def sam3_pred(variant, cond, q):
        p = np.full(n, np.nan)
        for r in lin:
            if r["variant"] == variant and r["cond"] == cond and r["stem"] in idx and r["pred_" + q]:
                p[idx[r["stem"]]] = fl(r["pred_" + q])
        return p

    def yolo_pred(source, imgsz, q):
        p = np.full(n, np.nan)
        if not E6: return p
        for r in csv.DictReader(open(E6[-1], encoding="utf-8")):
            if r["source"] == source and int(r["imgsz"]) == imgsz and r["stem"] in idx:
                p[idx[r["stem"]]] = fl(r["pred_" + q])
        return p

    PIPES = [
        ("SAM3 ZS full, max",        sam3_pred("zeroshot", "text_full", "w_max"),  "w_max"),
        ("SAM3 ZS full, P95",        sam3_pred("zeroshot", "text_full", "w_p95"),  "w_p95"),
        ("SAM3 ZS full, median",     sam3_pred("zeroshot", "text_full", "w_median"), "w_median"),
        ("SAM3 ZS crop, P95",        sam3_pred("zeroshot", "roi_crop", "w_p95"),   "w_p95"),
        ("SAM3 FT full, P95",        sam3_pred("ft_v3", "text_full", "w_p95"),     "w_p95"),
        ("SAM3 FT full, median",     sam3_pred("ft_v3", "text_full", "w_median"),  "w_median"),
        ("YOLO-Rob @640, max",       yolo_pred("yolo_rob", 640, "w_max"),          "w_max"),
        ("YOLO-Rob @640, P95",       yolo_pred("yolo_rob", 640, "w_p95"),          "w_p95"),
        ("YOLO-Rob @2016, P95",      yolo_pred("yolo_rob", 2016, "w_p95"),         "w_p95"),
        ("YOLO-CV12K @640, P95",     yolo_pred("yolo_cv12k", 640, "w_p95"),        "w_p95"),
    ]
    rng = np.random.default_rng(SEED)
    splits = [rng.permutation(n) for _ in range(NREP)]
    tests = [s[42:] for s in splits]

    print(f"\n  {'Pipeline (mask source, reduction; hard threshold)':<48}{'n_empty':>8}{'Acc':>7}{'Brier':>7}{'Acc@30%ref':>11}{'Acc same-q':>11}")
    latex = []
    for name, p, q in PIPES:
        if np.all(np.isnan(p)): print(f"  {name:<48}   (no data)"); continue
        nempty = int(np.isnan(p).sum()); pp = np.where(np.isnan(p), 0.0, p)     # 空掩膜 -> 宽度 0 -> 判低
        yhat = (pp > T).astype(int); y_same = (G[q] > T).astype(int)
        acc, bri, accr, accq = [], [], [], []
        for t in tests:
            acc.append(np.mean(yhat[t] == y_ref[t])); bri.append(np.mean((yhat[t] - y_ref[t]) ** 2))
            d = np.abs(pp[t] - T); keep = d > np.quantile(d, ABST)
            accr.append(np.mean((yhat[t] == y_ref[t])[keep])); accq.append(np.mean(yhat[t] == y_same[t]))
        print(f"  {name:<48}{nempty:>8}{np.mean(acc):>7.3f}{np.mean(bri):>7.3f}{np.mean(accr):>11.3f}{np.mean(accq):>11.3f}")
        latex.append(f"{name} & {np.mean(acc)*100:.1f} & {np.mean(bri):.3f} & {np.mean(accr)*100:.1f} & {np.mean(accq)*100:.1f} \\\\")

    # ---- CUSA: ZS full + P95 + log 校准（复现 E1 的 mul 模型）----
    w = sam3_pred("zeroshot", "text_full", "w_p95"); g = G["w_p95"]
    acc, bri, accr, accr_p = [], [], [], []
    for s in splits:
        cal, t = s[:42], s[42:]
        e = np.log(w[cal]) - np.log(g[cal]); b, sd = e.mean(), max(e.std(ddof=1), 1e-6)
        P = norm.cdf((np.log(w[t]) - b - np.log(T)) / sd); yhat = (P > 0.5).astype(int); yt = y_ref[t]
        acc.append(np.mean(yhat == yt)); bri.append(np.mean((P - yt) ** 2))
        conf = np.abs(P - 0.5); keep = conf > np.quantile(conf, ABST)
        accr.append(np.mean((yhat == yt)[keep]))
    print(f"  {'CUSA: SAM3 ZS full, P95, log calibration':<48}{0:>8}{np.mean(acc):>7.3f}{np.mean(bri):>7.3f}{np.mean(accr):>11.3f}{'---':>11}")
    latex.append(f"\\textbf{{CUSA}} (same masks, log calibration) & \\textbf{{{np.mean(acc)*100:.1f}}} & \\textbf{{{np.mean(bri):.3f}}} & \\textbf{{{np.mean(accr)*100:.1f}}} & --- \\\\")

    print("\n  说明：Acc/Brier 以参考掩膜 P95>10 mm 为真值；Acc@30%ref 为按 |w−T|（CUSA 为 |P−0.5|）弃权 30% 后的准确率；")
    print("        Acc same-q 以参考掩膜在该管线自身归约下的等级为真值，衡量掩膜误差单独的影响。")
    print("        硬阈值管线的 Brier 即 1−Acc；CUSA 的 Brier 来自概率。")
    print("\n  LaTeX rows:")
    for l in latex: print("    " + l)
    print(f"\n>> 日志: {LOGP}")


if __name__ == "__main__":
    main()