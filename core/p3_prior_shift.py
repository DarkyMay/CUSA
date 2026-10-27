#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p3_prior_shift.py —— 低流行率下的先验偏移修正（CPU，只读 closeloop CSV）

  模型：论文两参数 log 模型 M2，P(x) = Φ((log w − b − log T)/σ)，在标定集（正例率 π0）上校准。
  目标人群流行率 π' ≠ π0 时，按 Saerens et al. (2002) 修正后验：
        odds'(x) = odds(x) · [π'/(1−π')] / [π0/(1−π0)]
  π0 取该次切分的标定集正例率（部署时可得），π' 取目标流行率（自然 / 0.10 / 0.30 / 0.50 / 0.90）。
  测试集用重要性权重重加权到 π'（与 a_csv A1、p_width_dependent P1c 一致），200 次 42/84 切分，seed 20260728。
  输出：各流行率下 原始 M2 / 先验修正 M2 的全体准确率、Brier，c_α 表（拒答率/保留准确率），以及 majority-class 与
        “全判低/全判高”基线。另给 T=8、12 mm 的自然流行率对照。

用法：cd /root/autodl-tmp/proj_bridge && python3 p3_prior_shift.py
输出：results/p3_prior_<ts>.txt + results/p3_prior_<ts>.csv
"""
import os, sys, csv, datetime, time
import numpy as np
from scipy.stats import norm

OUT = "/root/autodl-tmp/proj_bridge"
CLOSELOOP = os.path.join(OUT, "results", "closeloop_v2_20260726_170420.csv")
SEED = 20260728; NSPLIT = 200; NCAL = 42; T_ASTM = 10.0
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"p3_prior_{ts}.txt")
CSVP = os.path.join(OUT, "results", f"p3_prior_{ts}.csv")
CALPHAS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()


def load_closeloop():
    rows = {}
    with open(CLOSELOOP, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["variant"] == "zeroshot" and r["cond"] == "text_full" and r["gt_ctype"] == "线状":
                rows[str(r["stem"])] = {k: float(r[k]) for k in ["gt_w_median", "gt_w_p95", "pred_w_median", "pred_w_p95"]}
    stems = sorted(rows, key=int)
    return stems, {k: np.array([rows[s][k] for s in stems]) for k in rows[stems[0]]}


def prior_shift(P, pi0, pi1):
    """Saerens 2002：按先验比修正后验概率"""
    r = (pi1 / (1 - pi1)) / (pi0 / (1 - pi0))
    return np.clip(P * r / (P * r + (1 - P)), 1e-6, 1 - 1e-6)


def run(wp, wr, T, prevalences, label):
    rng = np.random.RandomState(SEED)
    n = len(wp); y_all = (wr > T).astype(float)
    lw = np.log(np.maximum(wp, 1e-3)); lr = np.log(np.maximum(wr, 1e-3))
    M = ("raw", "shift")
    acc_all = {(pv, m): [] for pv in prevalences for m in M + ("majority", "all_low", "all_high", "pt")}
    bri_all = {(pv, m): [] for pv in prevalences for m in M}
    ca_ref = {(pv, c, m): [] for pv in prevalences for c in CALPHAS for m in M}
    ca_acc = {k: [] for k in ca_ref}
    pi0s = []
    for _ in range(NSPLIT):
        perm = rng.permutation(n); cal, te = perm[:NCAL], perm[NCAL:]
        res = lw[cal] - lr[cal]; b = res.mean(); s = res.std(ddof=0) + 1e-6
        pi0 = float(np.clip(y_all[cal].mean(), 1e-3, 1 - 1e-3)); pi0s.append(pi0)
        P_raw = np.clip(norm.cdf((lw[te] - b - np.log(T)) / s), 1e-6, 1 - 1e-6); y = y_all[te]
        g = (wp[te] > T).astype(float)
        for pv in prevalences:
            pi1 = y.mean() if pv is None else pv
            wgt = np.ones(len(te))
            if pv is not None:
                p0 = float(np.clip(y.mean(), 1e-3, 1 - 1e-3)); wgt = np.where(y > 0.5, pv / p0, (1 - pv) / (1 - p0))
            P_sh = prior_shift(P_raw, pi0, pi1)   # 目标流行率已知、标定集流行率估计
            acc_all[(pv, "majority")].append(max(pi1, 1 - pi1)); acc_all[(pv, "all_low")].append(1 - pi1); acc_all[(pv, "all_high")].append(pi1)
            acc_all[(pv, "pt")].append(np.average((g == y).astype(float), weights=wgt))
            for m, P in (("raw", P_raw), ("shift", P_sh)):
                conf = np.maximum(P, 1 - P); correct = ((P > 0.5) == (y > 0.5)).astype(float)
                acc_all[(pv, m)].append(np.average(correct, weights=wgt))
                bri_all[(pv, m)].append(np.average((P - y) ** 2, weights=wgt))
                for c in CALPHAS:
                    keep = conf >= c
                    ca_ref[(pv, c, m)].append(1 - np.average(keep.astype(float), weights=wgt))
                    ca_acc[(pv, c, m)].append(np.average(correct[keep], weights=wgt[keep]) if keep.sum() > 0 else np.nan)
    rows = []
    print(f"\n[{label}] T={T:.0f} mm  标定集正例率 π0 = {np.mean(pi0s):.3f} ± {np.std(pi0s):.3f}")
    for pv in prevalences:
        name = "自然 %.0f%%" % (100 * y_all.mean()) if pv is None else "%.0f%%" % (100 * pv)
        A = {m: float(np.mean(acc_all[(pv, m)])) for m in M + ("majority", "all_low", "all_high", "pt")}
        B = {m: float(np.mean(bri_all[(pv, m)])) for m in M}
        print(f"\n  流行率 {name}:  全体准确率  原始 M2 {100*A['raw']:.1f}%  先验修正 M2 {100*A['shift']:.1f}%  硬阈值 {100*A['pt']:.1f}%  "
              f"majority {100*A['majority']:.1f}% (全判低 {100*A['all_low']:.1f}% / 全判高 {100*A['all_high']:.1f}%)  |  Brier 原始 {B['raw']:.3f}  修正 {B['shift']:.3f}")
        print(f"    {'c_α':>5s} | {'原始 拒答%':>10s}{'保留acc%':>10s} | {'修正 拒答%':>10s}{'保留acc%':>10s}")
        for c in CALPHAS:
            r0, a0 = np.mean(ca_ref[(pv, c, "raw")]), np.nanmean(ca_acc[(pv, c, "raw")])
            r1, a1 = np.mean(ca_ref[(pv, c, "shift")]), np.nanmean(ca_acc[(pv, c, "shift")])
            print(f"    {c:5.2f} | {100*r0:10.1f}{100*a0:10.1f} | {100*r1:10.1f}{100*a1:10.1f}")
            rows.append(dict(label=label, T=T, prevalence=("natural" if pv is None else pv), c_alpha=c, ref_raw=r0, acc_raw=a0, ref_shift=r1, acc_shift=a1,
                             acc_all_raw=A["raw"], acc_all_shift=A["shift"], acc_pt=A["pt"], acc_majority=A["majority"], brier_raw=B["raw"], brier_shift=B["shift"], pi0=float(np.mean(pi0s))))
    return rows


def main():
    sys.stdout = Tee(LOGP); T0 = time.time()
    print(f"p3_prior_shift.py  {ts}  nsplit={NSPLIT} ncal={NCAL} seed={SEED}\nlog -> {LOGP}\n")
    stems, D = load_closeloop(); print(f"closeloop zs_full 线状 {len(stems)} 张")
    print("\n" + "=" * 110 + "\n[P3] 先验偏移修正：odds' = odds · [π'/(1−π')]/[π0/(1−π0)]，π0 = 标定集正例率，π' = 目标流行率\n" + "=" * 110)
    rows = []
    rows += run(D["pred_w_p95"], D["gt_w_p95"], T_ASTM, (None, 0.10, 0.30, 0.50, 0.90), "P95")
    rows += run(D["pred_w_p95"], D["gt_w_p95"], 8.0, (None, 0.10, 0.50), "P95")
    rows += run(D["pred_w_p95"], D["gt_w_p95"], 12.0, (None, 0.10, 0.50), "P95")
    rows += run(D["pred_w_median"], D["gt_w_median"], T_ASTM, (None, 0.10, 0.50, 0.90), "median")
    print("\n  判读：先验修正后 10% 列的全体准确率若 ≥ majority（90%）且 Brier 下降、拒答后准确率继续上升，则低流行率问题是先验偏移而非 (b,σ) 的问题；"
          "自然流行率列修正前后应几乎不变（π' ≈ π0）。")
    with open(CSVP, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nCSV {CSVP}\n总耗时 {(time.time()-T0)/60:.1f} min")


if __name__ == "__main__":
    main()