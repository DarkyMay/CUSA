#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""p_width_dependent.py —— 返修二轮补充实验（CPU，只读已有 CSV）

  P1  宽度依赖的偏差模型：在同样 200 次 42/84 切分（seed 20260728）上比较
        M2: log ŵ = log w − b + ε               （论文两参数 log 模型）
        M3: log ŵ = α + β·log w + ε              （三参数；α、β 由标定集 OLS 得到，σ 为残差 sd）
      输出：(a) 阈值扫描 6–20 mm（P95）与 4–12 mm（median）：Brier_pt / Brier_M2 / Brier_M3 / acc / ECE 失败率 / 参数
            (b) 42 张标定集上的模型选择：AIC/BIC 选 M3 的切分比例；β 的均值±sd
            (c) 拒答阈值 c_α × 流行率（自然/10/30/50/90%）：M2 与 M3 的拒答率/保留准确率，并列 majority-class 基线
  P2  在 lr 1e-5 epoch-1 微调 checkpoint 的宽度上跑 CUSA 标定（median 归约，T=10 mm；该 checkpoint 只有中值宽度）
      与零样本（同一 CSV 的 zeroshot 行）并列：bias / ρ / slope、M2 与 M3 的 Brier/acc/ECE

用法：
  cd /root/autodl-tmp/proj_bridge && python3 p_width_dependent.py
  可选：--e1csv results/e1_mitig_widths_20261021_234043.csv（默认）  --nsim 400  --quick
输出：results/p_widthdep_<ts>.txt + results/p_widthdep_<ts>.csv
"""
import os, sys, csv, glob, argparse, datetime, time
import numpy as np
from scipy.stats import spearmanr, norm

OUT = "/root/autodl-tmp/proj_bridge"
CLOSELOOP = os.path.join(OUT, "results", "closeloop_v2_20260726_170420.csv")
E1CSV_DEFAULT = os.path.join(OUT, "results", "e1_mitig_widths_20261021_234043.csv")
SEED = 20260728; NSPLIT = 200; NCAL = 42; T_ASTM = 10.0
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"p_widthdep_{ts}.txt")
CSVP = os.path.join(OUT, "results", f"p_widthdep_{ts}.csv")


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()


# ---------------------------------------------------------------- 数据
def load_closeloop():
    rows = {}
    with open(CLOSELOOP, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["variant"] == "zeroshot" and r["cond"] == "text_full" and r["gt_ctype"] == "线状":
                rows[str(r["stem"])] = {k: float(r[k]) for k in ["gt_w_median", "gt_w_p95", "gt_w_mean", "pred_w_median", "pred_w_p95", "pred_w_mean"]}
    stems = sorted(rows, key=int)
    return stems, {k: np.array([rows[s][k] for s in stems]) for k in rows[stems[0]]}


def load_e1(path):
    """tag -> {stem: (gt_mm, pred_mm)}，只取 ds == 'SUT'"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["ds"] != "SUT": continue
            out.setdefault(r["tag"], {})[str(r["stem"])] = (float(r["gt"]), float(r["pred"]))
    return out


# ---------------------------------------------------------------- 模型
def fit_m2(lw, lr):
    res = lw - lr; return dict(b=res.mean(), s=res.std(ddof=0) + 1e-6)

def fit_m3(lw, lr):
    beta, alpha = np.polyfit(lw, lr, 1)          # lr ≈ alpha + beta·lw
    res = lr - (alpha + beta * lw)
    return dict(alpha=alpha, beta=beta, s=res.std(ddof=0) + 1e-6)

def prob_m2(lw, p, T): return np.clip(norm.cdf((lw - p["b"] - np.log(T)) / p["s"]), 1e-6, 1 - 1e-6)
def prob_m3(lw, p, T): return np.clip(norm.cdf((p["alpha"] + p["beta"] * lw - np.log(T)) / p["s"]), 1e-6, 1 - 1e-6)

def loglik_m2(lw, lr, p):
    res = lr - (lw - p["b"]); return float(np.sum(norm.logpdf(res, 0, p["s"])))
def loglik_m3(lw, lr, p):
    res = lr - (p["alpha"] + p["beta"] * lw); return float(np.sum(norm.logpdf(res, 0, p["s"])))


def ece(p, yy, nb=10):
    e = 0.0; edges = np.linspace(0, 1, nb + 1)
    for i in range(nb):
        mk = (p >= edges[i]) & (p < edges[i + 1] + (1e-9 if i == nb - 1 else 0))
        if mk.any(): e += mk.mean() * abs(p[mk].mean() - yy[mk].mean())
    return e


def calib_compare(w_pred, w_ref, T, nsplit=NSPLIT, ncal=NCAL, seed=SEED, nsim=400, refer=0.30, prevalences=(None,), calphas=None):
    """同一切分序列上 M2 vs M3 vs 点估计；返回均值字典 + 按流行率的 c_α 表"""
    rng = np.random.RandomState(seed)          # 切分 + M2 零假设：与 b_estimators_gating.calib_eval 消耗完全一致 → 切分序列与 Table 9 相同
    rng3 = np.random.RandomState(seed + 1)    # M3 零假设用独立 RNG，不打乱主 RNG 的状态
    n = len(w_pred); y_all = (w_ref > T).astype(float)
    lw = np.log(np.maximum(w_pred, 1e-3)); lr = np.log(np.maximum(w_ref, 1e-3))
    keys = ["brier_pt", "acc_pt", "ref_pt", "brier2", "acc2", "ref2", "ece2", "nll2", "brier3", "acc3", "ref3", "ece3", "nll3",
            "b", "s2", "alpha", "beta", "s3", "sel3_aic", "sel3_bic"]
    acc = {k: [] for k in keys}
    ca_rows = []   # (prevalence, c, model, ref, acc)
    calphas = calphas or [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    ca_acc = {(pv, c, m): [] for pv in prevalences for c in calphas for m in ("M2", "M3")}
    ca_ref = {k: [] for k in ca_acc}
    ca_all = {(pv, m): [] for pv in prevalences for m in ("M2", "M3", "majority")}
    for _ in range(nsplit):
        perm = rng.permutation(n); cal, te = perm[:ncal], perm[ncal:]
        p2 = fit_m2(lw[cal], lr[cal]); p3 = fit_m3(lw[cal], lr[cal])
        ll2 = loglik_m2(lw[cal], lr[cal], p2); ll3 = loglik_m3(lw[cal], lr[cal], p3)
        acc["sel3_aic"].append(float((2 * 3 - 2 * ll3) < (2 * 2 - 2 * ll2)))
        acc["sel3_bic"].append(float((3 * np.log(ncal) - 2 * ll3) < (2 * np.log(ncal) - 2 * ll2)))
        y = y_all[te]
        P2 = prob_m2(lw[te], p2, T); P3 = prob_m3(lw[te], p3, T)
        g = (w_pred[te] > T).astype(float)
        acc["brier_pt"].append(np.mean((g - y) ** 2)); acc["acc_pt"].append(np.mean(g == y))
        keep2 = np.argsort(-np.abs(w_pred[te] - T))[: int(round(len(te) * (1 - refer)))]
        acc["ref_pt"].append(np.mean(g[keep2] == y[keep2]))
        for tag, P in (("2", P2), ("3", P3)):
            acc["brier" + tag].append(np.mean((P - y) ** 2))
            acc["nll" + tag].append(-np.mean(y * np.log(P) + (1 - y) * np.log(1 - P)))
            acc["acc" + tag].append(np.mean((P > 0.5) == (y > 0.5)))
            keep = np.argsort(-np.abs(P - 0.5))[: int(round(len(te) * (1 - refer)))]
            acc["ref" + tag].append(np.mean((P[keep] > 0.5) == (y[keep] > 0.5)))
            e_obs = ece(P, y)
            r_ = rng if tag == "2" else rng3
            e_null = np.array([ece(P, (r_.rand(len(P)) < P).astype(float)) for _ in range(nsim)])
            acc["ece" + tag].append(float(e_obs > np.percentile(e_null, 95)))
        acc["b"].append(p2["b"]); acc["s2"].append(p2["s"]); acc["alpha"].append(p3["alpha"]); acc["beta"].append(p3["beta"]); acc["s3"].append(p3["s"])
        # c_α × 流行率（重要性权重，与 a_csv A1 一致）
        for pv in prevalences:
            wgt = np.ones(len(te))
            if pv is not None:
                p0 = min(max(y.mean(), 1e-3), 1 - 1e-3)
                wgt = np.where(y > 0.5, pv / p0, (1 - pv) / (1 - p0))
            pv_eff = y.mean() if pv is None else pv
            ca_all[(pv, "majority")].append(max(pv_eff, 1 - pv_eff))
            for m, P in (("M2", P2), ("M3", P3)):
                conf = np.maximum(P, 1 - P); correct = ((P > 0.5) == (y > 0.5)).astype(float)
                ca_all[(pv, m)].append(np.average(correct, weights=wgt))
                for c in calphas:
                    keep = conf >= c
                    ca_ref[(pv, c, m)].append(1 - np.average(keep.astype(float), weights=wgt))
                    ca_acc[(pv, c, m)].append(np.average(correct[keep], weights=wgt[keep]) if keep.sum() > 0 else np.nan)
    out = {k: float(np.mean(v)) for k, v in acc.items()}
    for k in ("b", "s2", "alpha", "beta", "s3"): out[k + "_sd"] = float(np.std(acc[k]))
    out["pos_rate"] = float(y_all.mean()); out["n"] = n
    ca = {}
    for pv in prevalences:
        ca[pv] = dict(all={m: float(np.mean(ca_all[(pv, m)])) for m in ("M2", "M3", "majority")},
                      rows=[(c, float(np.mean(ca_ref[(pv, c, "M2")])), float(np.nanmean(ca_acc[(pv, c, "M2")])),
                             float(np.mean(ca_ref[(pv, c, "M3")])), float(np.nanmean(ca_acc[(pv, c, "M3")]))) for c in calphas])
    return out, ca


def widthstats(w_pred, w_ref):
    ok = w_ref > 0
    bias = 100 * (np.mean(w_pred[ok]) / np.mean(w_ref[ok]) - 1); rho = spearmanr(w_ref, w_pred).correlation
    sl, ic = np.polyfit(w_ref[ok], w_pred[ok], 1); mae = np.mean(np.abs(w_pred - w_ref))
    return bias, rho, sl, ic, mae


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1csv", type=str, default=E1CSV_DEFAULT)
    ap.add_argument("--nsim", type=int, default=400)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    nsplit = 20 if args.quick else NSPLIT; nsim = 50 if args.quick else args.nsim
    sys.stdout = Tee(LOGP); T0 = time.time()
    print(f"p_width_dependent.py  {ts}  nsplit={nsplit} nsim={nsim} seed={SEED}\nlog -> {LOGP}\n")
    stems, D = load_closeloop(); n = len(stems)
    print(f"closeloop zs_full 线状 {n} 张")
    crows = []

    def report(label, o):
        print(f"  {label:28s} pos={100*o['pos_rate']:5.1f}% | pt: Brier {o['brier_pt']:.3f} acc {100*o['acc_pt']:5.1f} ref30 {100*o['ref_pt']:5.1f} "
              f"| M2: Brier {o['brier2']:.3f} acc {100*o['acc2']:5.1f} ref30 {100*o['ref2']:5.1f} ECEf {100*o['ece2']:5.1f}% "
              f"| M3: Brier {o['brier3']:.3f} acc {100*o['acc3']:5.1f} ref30 {100*o['ref3']:5.1f} ECEf {100*o['ece3']:5.1f}% "
              f"| b {o['b']:.3f}±{o['b_sd']:.3f} σ2 {o['s2']:.3f} | α {o['alpha']:.3f} β {o['beta']:.3f}±{o['beta_sd']:.3f} σ3 {o['s3']:.3f} | AIC选M3 {100*o['sel3_aic']:4.0f}% BIC选M3 {100*o['sel3_bic']:4.0f}%")

    # ---------------- P1(a) 阈值扫描
    print("\n" + "=" * 110 + "\n[P1a] 阈值扫描：M2（两参数 log）vs M3（三参数 log-linear）vs 点估计；zs_full, EDT\n" + "=" * 110)
    for red, Ts in (("p95", [6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 20]), ("median", [4, 5, 6, 7, 8, 9, 10, 12])):
        wp, wr = D[f"pred_w_{red}"], D[f"gt_w_{red}"]
        for T in Ts:
            o, _ = calib_compare(wp, wr, float(T), nsplit=nsplit, nsim=nsim)
            report(f"{red:7s} T={T:4.1f}", o)
            crows.append(dict(stage="P1a", source="zs_full", red=red, T=T, **o))
    print("\n  判读：若 M3 在 11–14 mm 把 ECE 失败率压到 ~5% 且 Brier 不劣于 M2，则宽度依赖的 b 是有证据的修正；AIC/BIC 选 M3 的比例说明 42 张是否支持第三个参数。")

    # ---------------- P1(b) 宽度依赖的直接证据（全样本）
    print("\n" + "=" * 110 + "\n[P1b] 残差 log(pred/gt) 对 GT 宽度的依赖（全 126 张，不切分）\n" + "=" * 110)
    for red in ("median", "p95"):
        wp, wr = D[f"pred_w_{red}"], D[f"gt_w_{red}"]
        lw, lr = np.log(wp), np.log(wr); res = lw - lr
        beta, alpha = np.polyfit(lw, lr, 1)
        q = np.quantile(wr, [0.25, 0.5, 0.75])
        bq = [res[wr <= q[0]].mean(), res[(wr > q[0]) & (wr <= q[1])].mean(), res[(wr > q[1]) & (wr <= q[2])].mean(), res[wr > q[2]].mean()]
        print(f"  {red:7s} ρ(res, gt)={spearmanr(res, wr).correlation:+.3f}  四分位 b: {bq[0]:.2f} / {bq[1]:.2f} / {bq[2]:.2f} / {bq[3]:.2f}  "
              f"全样本 M3: α={alpha:.3f} β={beta:.3f} σ3={np.std(lr-(alpha+beta*lw)):.3f}  M2: b={res.mean():.3f} σ2={res.std():.3f}")

    # ---------------- P1(c) c_α × 流行率（M2 vs M3 vs majority）
    print("\n" + "=" * 110 + "\n[P1c] 拒答阈值 × 流行率：M2 vs M3，并列 majority-class 基线；zs_full, P95, T=10\n" + "=" * 110)
    wp, wr = D["pred_w_p95"], D["gt_w_p95"]
    o, ca = calib_compare(wp, wr, T_ASTM, nsplit=nsplit, nsim=nsim, prevalences=(None, 0.10, 0.30, 0.50, 0.90))
    for pv in (None, 0.10, 0.30, 0.50, 0.90):
        a = ca[pv]["all"]
        print(f"\n  流行率 {'自然 %.0f%%' % (100*o['pos_rate']) if pv is None else '%.0f%%' % (100*pv)}:  全体准确率 M2 {100*a['M2']:.1f}%  M3 {100*a['M3']:.1f}%  majority-class {100*a['majority']:.1f}%")
        print(f"    {'c_α':>5s} | {'M2 拒答%':>9s}{'M2 保留acc%':>12s} | {'M3 拒答%':>9s}{'M3 保留acc%':>12s}")
        for c, r2, a2, r3, a3 in ca[pv]["rows"]:
            print(f"    {c:5.2f} | {100*r2:9.1f}{100*a2:12.1f} | {100*r3:9.1f}{100*a3:12.1f}")
            crows.append(dict(stage="P1c", source="zs_full", red="p95", T=T_ASTM, prevalence=("natural" if pv is None else pv), c_alpha=c,
                              ref_M2=r2, acc_M2=a2, ref_M3=r3, acc_M3=a3, acc_all_M2=a["M2"], acc_all_M3=a["M3"], acc_majority=a["majority"]))
    print("\n  判读：10% 列若 M3 全体准确率 ≥ majority（90%）且拒答后继续上升，则低流行率问题由宽度依赖的 b 解决；否则如实写为局限并给 majority 基线。")

    # ---------------- P2 微调 checkpoint 上的 CUSA
    print("\n" + "=" * 110 + f"\n[P2] 微调 checkpoint 的 CUSA 标定（median 归约，T=10）：{os.path.basename(args.e1csv)}\n" + "=" * 110)
    if not os.path.exists(args.e1csv):
        print(f"  !! 找不到 {args.e1csv}")
    else:
        E = load_e1(args.e1csv)
        tags = [t for t in E if t == "zeroshot" or t.endswith("_ep1") or t.endswith("_ep2") or t.endswith("_ep25")]
        tags = sorted(tags, key=lambda t: (t != "zeroshot", t))
        print(f"  CSV 中的 tag: {sorted(E)}；本次评估: {tags}")
        # 参考：closeloop 的 gt_w_median（e1 的 gt 列即此值）
        for tag in tags:
            d = E[tag]; ss = [s for s in stems if s in d]
            wr_ = np.array([d[s][0] for s in ss]); wp_ = np.array([d[s][1] for s in ss])
            bias, rho, sl, ic, mae = widthstats(wp_, wr_)
            o, _ = calib_compare(wp_, wr_, T_ASTM, nsplit=nsplit, nsim=nsim)
            print(f"  [{tag}] n={len(ss)} bias {bias:+6.1f}% ρ {rho:.3f} slope {sl:.2f} icpt {ic:.2f} MAE {mae:.2f}")
            report(f"    {tag}", o)
            crows.append(dict(stage="P2", source=tag, red="median", T=T_ASTM, bias=bias, rho=rho, slope=sl, icpt=ic, mae=mae, **o))
        print("\n  判读：与 zeroshot 行比较。若 lr1e-5 ep1 的 M2/M3 Brier 与 acc 接近零样本（b 吸收偏移、ρ 0.80 保排序），说明 slope 0.5 的'部分塌缩'仍可标定；"
              "若明显变差，说明斜率下降直接损害可标定性，支持论文'塌缩'结论。")

    # CSV
    keys = sorted({k for r in crows for k in r})
    with open(CSVP, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(crows)
    print(f"\nCSV {CSVP}\n总耗时 {(time.time()-T0)/60:.1f} min")


if __name__ == "__main__":
    main()
