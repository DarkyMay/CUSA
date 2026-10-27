#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e1_calibration_reanalysis.py —— 标定式概率分级的复算（回应审稿 M7/M8，承接 width_mechanism 的"加性优于乘性"）

与 probe_probabilistic_grading.py (2026-07-28) 的关系：
  · 同一数据（closeloop_v2 逐图 pred/gt），同一切分口径（校准 n//3=42 / 测试 84，200 次），但
    ★ 所有模型与基线共用同一批 200 次切分（旧脚本各段各自抽样，数字不可互比）
  · 误差模型：
      add   加性         ĝ = w − b + ε                          P = Φ((w−b−T)/σ)                （论文式 4/6）
      mul   对数乘性     log ĝ = log w − b + ε                  P = Φ((log w − b − log T)/σ)    （论文式 5，旧代码未实现）
      quad  平方相加     ĝ² = w² − f² + ε（模糊地板）            P = Φ((w²−f²−T²)/σ)            （width_mechanism 支持）
      ratio 旧代码的比例模型（仅为复现 0.1308）
      auto  每次切分按校准集对数似然在 add/mul/quad 中选
  · 公平基线（审稿 M7）：点估计硬分级、先验常数、Platt(逻辑回归 y~w−T)、isotonic(y~w)
  · 拒答：加入"未校准 |w−T| 排序"对照，检验拒答收益是否与校准无关
  · ECE 零分布检验：每次切分都做（旧脚本仅单次），报告 p 值分布
  · 校准集规模曲线：对 add/mul/quad 各自给出（旧脚本只算加性，导致与主结果不可比）
输出：results/e1_calib_<ts>.txt
"""
import os, sys, csv, glob, datetime
import numpy as np
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

OUT = "/root/autodl-tmp/proj_bridge"
SEED = 20260728
NREP = 200
NNULL = 400                     # 每次切分的零分布模拟次数（200×400 次 ECE，约 1 分钟）
CONFIG = ("zeroshot", "text_full")
CASES = [("ASTM", 10.0, "w_p95", "主口径：ASTM 文本→P95"),
         ("ASTM", 10.0, "w_median", "副口径：同一阈值换中位，检验结论对归约的稳健性")]
SIZES = (5, 10, 15, 20, 30, 42, 60)
ABST = (0.0, 0.1, 0.2, 0.3, 0.5)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"e1_calib_{ts}.txt")


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()


def load():
    p = os.path.join(OUT, "results", "closeloop_v2_20260726_170420.csv")
    rows = [r for r in csv.DictReader(open(p, encoding="utf-8"))
            if r["gt_ctype"] == "线状" and r["variant"] == CONFIG[0] and r["cond"] == CONFIG[1]
            and r.get("pred_w_median")]
    print(f"读取 {os.path.basename(p)}  配置 {CONFIG}  n={len(rows)}")
    return rows


def ece(p, y, nb=10):
    e, n = 0.0, len(p)
    for i in range(nb):
        lo, hi = i / nb, (i + 1) / nb
        m = (p >= lo) & ((p < hi) if i < nb - 1 else (p <= hi))
        if m.sum(): e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return e


def nll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


# ---------------- 误差模型：fit(w_cal, g_cal) -> params ; prob(w, T, params) ; loglik(w_cal,g_cal,params) ----------------
def fit_add(w, g):
    e = w - g; return dict(b=e.mean(), s=max(e.std(ddof=1), 1e-6))
def prob_add(w, T, P): return norm.cdf((w - P["b"] - T) / P["s"])
def ll_add(w, g, P): return norm.logpdf(g, w - P["b"], P["s"]).sum()

def fit_mul(w, g):
    e = np.log(w) - np.log(g); return dict(b=e.mean(), s=max(e.std(ddof=1), 1e-6))
def prob_mul(w, T, P): return norm.cdf((np.log(w) - P["b"] - np.log(T)) / P["s"])
def ll_mul(w, g, P):  # 密度换元到 g 空间：p(g) = N(log g; log w − b, s)/g
    return (norm.logpdf(np.log(g), np.log(w) - P["b"], P["s"]) - np.log(g)).sum()

def fit_quad(w, g):
    e = w ** 2 - g ** 2; return dict(f2=e.mean(), s=max(e.std(ddof=1), 1e-6))
def prob_quad(w, T, P): return norm.cdf((w ** 2 - P["f2"] - T ** 2) / P["s"])
def ll_quad(w, g, P):  # p(g) = N(g²; w²−f², s)·2g
    return (norm.logpdf(g ** 2, w ** 2 - P["f2"], P["s"]) + np.log(2 * g)).sum()

def fit_ratio(w, g):  # 旧代码
    r = w / np.maximum(g, 1e-6); return dict(b=r.mean(), s=max(r.std(ddof=1), 1e-6))
def prob_ratio(w, T, P):
    return norm.cdf((w / P["b"] - T) / np.maximum(w * P["s"] / P["b"] ** 2, 1e-6))

MODELS = {"add": (fit_add, prob_add, ll_add), "mul": (fit_mul, prob_mul, ll_mul),
          "quad": (fit_quad, prob_quad, ll_quad)}


def run_case(code, T, stat, note, rows, rng):
    w = np.array([float(r["pred_" + stat]) for r in rows])
    g = np.array([float(r["gt_" + stat]) for r in rows])
    y = (g > T).astype(int); n = len(w); ncal = max(20, n // 3)
    print("\n" + "=" * 100)
    print(f"[{code} {T}mm | {stat}]  {note}   n={n}  越阈比例 {y.mean():.1%}  校准 {ncal}/测试 {n-ncal}  重复 {NREP}")
    print("=" * 100)
    splits = [rng.permutation(n) for _ in range(NREP)]

    names = ["add", "mul", "quad", "ratio(old)", "auto", "platt", "isotonic", "point", "prior"]
    M = {k: dict(brier=[], ece=[], nll=[], acc=[], pval=[], abst={q: [] for q in ABST}) for k in names}
    M["dist|w-T|"] = dict(abst={q: [] for q in ABST})
    picks = []; params = {k: [] for k in MODELS}
    for idx in splits:
        cal, tst = idx[:ncal], idx[ncal:]
        wc, gc, wt, yt = w[cal], g[cal], w[tst], y[tst]
        probs = {}
        # 参数模型
        lls = {}
        for k, (fit, prob, ll) in MODELS.items():
            P = fit(wc, gc); params[k].append(P); lls[k] = ll(wc, gc, P); probs[k] = prob(wt, T, P)
        best = max(lls, key=lls.get); picks.append(best); probs["auto"] = probs[best]
        probs["ratio(old)"] = prob_ratio(wt, T, fit_ratio(wc, gc))
        # 基线
        yc = (gc > T).astype(int)
        if 0 < yc.sum() < len(yc):
            lr = LogisticRegression(C=1e6, max_iter=1000).fit((wc - T)[:, None], yc)
            probs["platt"] = lr.predict_proba((wt - T)[:, None])[:, 1]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(wc, yc)
            probs["isotonic"] = np.clip(iso.predict(wt), 0, 1)
        else:
            probs["platt"] = probs["isotonic"] = np.full(len(wt), yc.mean())
        probs["point"] = (wt > T).astype(float)
        probs["prior"] = np.full(len(wt), yc.mean())
        for k, p in probs.items():
            M[k]["brier"].append(np.mean((p - yt) ** 2)); M[k]["ece"].append(ece(p, yt))
            M[k]["nll"].append(nll(p, yt)); M[k]["acc"].append(np.mean((p > 0.5) == yt))
            # ECE 零分布检验（每次切分）
            if k in ("add", "mul", "quad", "auto", "platt", "isotonic"):
                e_obs = ece(p, yt)
                null = np.array([ece(p, (rng.random(len(p)) < p).astype(int)) for _ in range(NNULL)])
                M[k]["pval"].append(float(np.mean(null >= e_obs)))
            # 拒答曲线：按 |p−0.5| 由低到高弃权
            conf = np.abs(p - 0.5)
            for q in ABST:
                keep = conf > np.quantile(conf, q) if q > 0 else np.ones(len(conf), bool)
                if keep.sum() >= 5: M[k]["abst"][q].append(np.mean((p[keep] > 0.5) == yt[keep]))
        # 未校准距离排序的拒答对照：按 |w−T| 由小到大弃权，硬分级 w>T
        d = np.abs(wt - T)
        for q in ABST:
            keep = d > np.quantile(d, q) if q > 0 else np.ones(len(d), bool)
            if keep.sum() >= 5: M["dist|w-T|"]["abst"][q].append(np.mean(((wt > T) == yt)[keep]))

    # -------- 报告 --------
    cnt = {k: picks.count(k) for k in MODELS}
    print(f"\n[1] 误差模型的校准集对数似然选择：{cnt}  （auto = 逐次切分取最优）")
    print(f"    参数（200 次切分均值±sd）：")
    for k in MODELS:
        keys = list(params[k][0].keys())
        print(f"      {k:<5}" + "  ".join(f"{kk}={np.mean([P[kk] for P in params[k]]):.3f}±{np.std([P[kk] for P in params[k]]):.3f}" for kk in keys)
              + (f"   → 等效地板 f={np.sqrt(max(np.mean([P['f2'] for P in params[k]]),0)):.2f} mm" if k == "quad" else ""))

    print(f"\n[2] 测试集指标（200 次切分均值；点估计 = 现行做法 w>T；prior = 校准集越阈比例常数）")
    print(f"    {'模型':<12}{'Brier':>8}{'NLL':>8}{'ECE均':>8}{'准确率':>8}{'ECE检验 p 中位':>14}{'p<0.05 占比':>12}")
    for k in names:
        R = M[k]
        pv = R["pval"]
        print(f"    {k:<12}{np.mean(R['brier']):>8.4f}{np.mean(R['nll']):>8.4f}{np.mean(R['ece']):>8.4f}{np.mean(R['acc']):>8.4f}"
              + (f"{np.median(pv):>14.3f}{np.mean(np.array(pv)<0.05):>12.1%}" if pv else f"{'—':>14}{'—':>12}"))
    print("    判读：Brier/NLL 越低越好；p<0.05 占比若接近 5% 即与完美校准无差别；"
          "platt/isotonic 是审稿人会要的'同样用 42 张标注、同样输出概率'的公平基线。")

    print(f"\n[3] 拒答曲线：保留样本上的准确率（按各自置信度由低到高弃权；dist|w-T| = 未校准距离排序 + 硬分级）")
    print(f"    {'弃权比例':<10}" + "".join(f"{k:>12}" for k in ["add", "mul", "quad", "auto", "platt", "dist|w-T|"]))
    for q in ABST:
        print(f"    {q:<10.0%}" + "".join(f"{np.mean(M[k]['abst'][q]):>12.4f}" if M[k]["abst"][q] else f"{'—':>12}"
                                          for k in ["add", "mul", "quad", "auto", "platt", "dist|w-T|"]))
    print("    判读：dist|w-T| = 不用任何标注的做法（硬分级 w>T，按 |w−T| 弃权）。")
    print("          add 的硬分级与弃权排序等价于 |w−b−T|：σ 不影响二者，add 对点估计的全部准确率/拒答收益都来自")
    print("          偏差常数 b（42 张图估出）；σ 只决定概率的形状（Brier/校准）。quad/mul 若高于 add，说明其排序含额外信息。")

    print(f"\n[4] 校准集规模曲线（每个模型各自；旧脚本只算加性，与主结果不可比）")
    print(f"    {'张数':<6}" + "".join(f"{k+' Brier':>12}{k+' ECE':>10}{k+' p中位':>10}" for k in MODELS))
    for m in SIZES:
        if m >= n - 20: break
        line = f"    {m:<6}"
        for k, (fit, prob, ll) in MODELS.items():
            bs, es, ps = [], [], []
            for idx in splits:
                cal, tst = idx[:m], idx[m:]
                P = fit(w[cal], g[cal]); p = prob(w[tst], T, P); yt = y[tst]
                bs.append(np.mean((p - yt) ** 2)); es.append(ece(p, yt))
                e_obs = es[-1]
                null = np.array([ece(p, (rng.random(len(p)) < p).astype(int)) for _ in range(60)])
                ps.append(np.mean(null >= e_obs))
            line += f"{np.mean(bs):>12.4f}{np.mean(es):>10.4f}{np.median(ps):>10.3f}"
        print(line)
    print("    判读：ECE 均值随测试集变小而机械上升（分箱效应），不能作校准判据；看 p 中位与 Brier。")

    # 额外：add 模型 200 次切分的 pooled 可靠性图数据（10 箱）
    print(f"\n[5] 可靠性表（add 与 quad，200 次切分汇总，10 箱）：置信区间中点 / 实际越阈率 / 样本数")
    for k in ("add", "quad"):
        fit, prob, _ = MODELS[k]; P_all, Y_all = [], []
        for idx in splits:
            cal, tst = idx[:ncal], idx[ncal:]
            P_all.append(prob(w[tst], T, fit(w[cal], g[cal]))); Y_all.append(y[tst])
        P_all = np.concatenate(P_all); Y_all = np.concatenate(Y_all)
        cells = []
        for i in range(10):
            m_ = (P_all >= i / 10) & ((P_all < (i + 1) / 10) if i < 9 else (P_all <= 1))
            if m_.sum(): cells.append(f"{(i+.5)/10:.2f}/{Y_all[m_].mean():.2f}/{m_.sum()}")
        print(f"    {k:<5} " + "  ".join(cells))


def main():
    sys.stdout = Tee(LOGP)
    print(f"e1_calibration_reanalysis.py  {ts}   SEED={SEED}  NREP={NREP}  NNULL={NNULL}")
    rows = load()
    for code, T, stat, note in CASES:
        rng = np.random.default_rng(SEED)
        run_case(code, T, stat, note, rows, rng)
    print(f"\n>> 日志: {LOGP}")


if __name__ == "__main__":
    main()