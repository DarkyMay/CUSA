#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e2e3_sensitivity_rank.py —— 两个无 GPU 的补充分析（回应审稿 M3/M5/M6）

E2  敏感性随阈值位置变化
    · ASTM 阈值 T 从 2 到 25 mm 扫描；每个 T 下计算
        (a) 6 配置 × 4 归约 的"判低比例"极差   (b) 仅零样本 3 配置 × 4 归约   (c) 参考掩膜 4 归约
    · 标出 T 在 GT 宽度分布中的百分位 → 验证"敏感性在阈值靠近众数时最大、在尾部收窄"（论文 4.3.1 的声明）
    · 收窄网格：剔除 |中位缝宽偏差|>20% 的配置后极差还有多大（回应"稻草人"批评）
E3  Dice 排序 vs 宽度误差排序 的统计化
    · 2000 次图像 bootstrap：Dice 最优配置 ≠ MAE 最优配置 的频率；两排序 Spearman 的分布
    · 配置内逐图：Spearman(Dice_i, |err_i|)、按 Dice 分箱的宽度排序 —— 区分"跨配置偏差"与"配置内判别"
    · 对 width_mechanism 的 5 种解码重复 bootstrap（单配置内的脱钩）
输出 results/e2e3_<ts>.txt
"""
import os, sys, csv, glob, datetime
import numpy as np
from scipy.stats import spearmanr, kendalltau

OUT = "/root/autodl-tmp/proj_bridge"
CL = os.path.join(OUT, "results", "closeloop_v2_20260726_170420.csv")
WM = sorted(glob.glob(os.path.join(OUT, "results", "width_mechanism_perimage_*.csv")))
STATS = ["w_median", "w_mean", "w_p95", "w_max"]
CONFIGS = [("zeroshot", "text_full"), ("zeroshot", "roi_crop"), ("zeroshot", "box_prompt"),
           ("ft_v3", "text_full"), ("ft_v3", "roi_crop"), ("ft_v3", "box_prompt")]
NAME = {("zeroshot", "text_full"): "zs/full", ("zeroshot", "roi_crop"): "zs/crop", ("zeroshot", "box_prompt"): "zs/box",
        ("ft_v3", "text_full"): "ft/full", ("ft_v3", "roi_crop"): "ft/crop", ("ft_v3", "box_prompt"): "ft/box"}
NBOOT = 2000; SEED = 0
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"e2e3_{ts}.txt")


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()


def fl(x):
    try: return float(x)
    except Exception: return np.nan


def load():
    rows = [r for r in csv.DictReader(open(CL, encoding="utf-8")) if r["gt_ctype"] == "线状"]
    stems = sorted({r["stem"] for r in rows}, key=lambda s: int(s))
    # 每配置：stem -> row；只保留六配置都有预测的 stem
    by = {c: {r["stem"]: r for r in rows if (r["variant"], r["cond"]) == c and r["pred_w_median"]} for c in CONFIGS}
    common = [s for s in stems if all(s in by[c] for c in CONFIGS)]
    gt = {st: np.array([fl(by[CONFIGS[0]][s]["gt_" + st]) for s in common]) for st in STATS}
    pred = {c: {st: np.array([fl(by[c][s]["pred_" + st]) for s in common]) for st in STATS} for c in CONFIGS}
    dice = {c: np.array([fl(by[c][s]["dice"]) for s in common]) for c in CONFIGS}
    print(f"读取 {os.path.basename(CL)}  线状且六配置齐全 n={len(common)}")
    return common, gt, pred, dice


# =============================================================================== E2
def e2(gt, pred):
    print("\n" + "=" * 100)
    print("E2  敏感性随阈值位置变化（判低比例 = 缝宽 <= T 的样本占比）")
    print("=" * 100)
    g_ref = gt["w_median"]
    print(f"  GT 中位缝宽分布：P10 {np.percentile(g_ref,10):.2f}  P25 {np.percentile(g_ref,25):.2f}  中位 {np.median(g_ref):.2f}"
          f"  P75 {np.percentile(g_ref,75):.2f}  P90 {np.percentile(g_ref,90):.2f} mm；GT P95 缝宽中位 {np.median(gt['w_p95']):.2f} mm")
    Ts = np.arange(2.0, 25.01, 1.0)
    zs = [c for c in CONFIGS if c[0] == "zeroshot"]
    # 收窄集：|中位缝宽偏差| <= 20%
    bias = {c: (pred[c]["w_median"].mean() - gt["w_median"].mean()) / gt["w_median"].mean() for c in CONFIGS}
    narrow = [c for c in CONFIGS if abs(bias[c]) <= 0.20]
    print("  各配置中位缝宽相对偏差：" + "  ".join(f"{NAME[c]} {bias[c]:+.1%}" for c in CONFIGS))
    print(f"  |偏差|<=20% 的收窄集：{[NAME[c] for c in narrow]}")
    print(f"\n  {'T mm':>5}{'T在GT中位分布的百分位':>22}{'全部6×4 极差':>14}{'零样本3×4 极差':>15}{'收窄集×4 极差':>14}{'参考掩膜4归约 极差':>18}{'参考掩膜 median→max':>22}")
    curve = []
    for T in Ts:
        pct = np.mean(g_ref <= T) * 100
        allv = [np.mean(pred[c][st] <= T) for c in CONFIGS for st in STATS]
        zsv = [np.mean(pred[c][st] <= T) for c in zs for st in STATS]
        nv = [np.mean(pred[c][st] <= T) for c in narrow for st in STATS] if narrow else [np.nan]
        refv = [np.mean(gt[st] <= T) for st in STATS]
        curve.append((T, pct, max(allv) - min(allv), max(zsv) - min(zsv), (max(nv) - min(nv)) if narrow else np.nan,
                      max(refv) - min(refv), refv[0], refv[3]))
        print(f"  {T:>5.0f}{pct:>21.0f}%{curve[-1][2]:>14.1%}{curve[-1][3]:>15.1%}{curve[-1][4]:>14.1%}{curve[-1][5]:>18.1%}"
              f"{f'{refv[0]:.1%} → {refv[3]:.1%}':>22}")
    c = np.array(curve)
    iall, iref = np.argmax(c[:, 2]), np.argmax(c[:, 5])
    print(f"\n  全部配置极差最大处 T={c[iall,0]:.0f} mm（GT 中位分布百分位 {c[iall,1]:.0f}%，极差 {c[iall,2]:.1%}）")
    print(f"  参考掩膜极差最大处 T={c[iref,0]:.0f} mm（百分位 {c[iref,1]:.0f}%，极差 {c[iref,5]:.1%}）")
    print(f"  ASTM 10 mm 处：百分位 {c[Ts==10,1][0]:.0f}%，全部极差 {c[Ts==10,2][0]:.1%}，零样本极差 {c[Ts==10,3][0]:.1%}，"
          f"收窄集极差 {c[Ts==10,4][0]:.1%}，参考掩膜极差 {c[Ts==10,5][0]:.1%}")
    tail = c[(c[:, 1] > 95) | (c[:, 1] < 5)]
    if len(tail):
        print(f"  分布尾部（百分位 <5% 或 >95%）的全部极差：中位 {np.median(tail[:,2]):.1%}，最大 {tail[:,2].max():.1%}")
    print("  判读：若极差在百分位 30–70% 区间达到峰值、在尾部收窄，则'敏感性是阈值相对位置的性质'得到验证；")
    print("        收窄集极差是回应'稻草人'批评时应报告的数字；参考掩膜极差不依赖任何分割配置。")

    # 3 mm（JTG）处顺带报告
    print(f"\n  JTG 3 mm 处（判重比例 = >3mm）：参考掩膜 " + "  ".join(f"{st} {np.mean(gt[st]>3):.1%}" for st in STATS)
          + "；全部配置范围 " + f"{min(np.mean(pred[c][st]>3) for c in CONFIGS for st in STATS):.1%}–"
          f"{max(np.mean(pred[c][st]>3) for c in CONFIGS for st in STATS):.1%}")


# =============================================================================== E3
def rank_boot(dice_mat, err_mat, names, rng, nboot):
    """dice_mat, err_mat: (K configs, n images)。返回 Dice最优≠MAE最优 频率、排序Spearman 分布、Kendall"""
    K, n = dice_mat.shape
    dis, rhos, taus = 0, [], []
    best_pairs = {}
    for _ in range(nboot):
        i = rng.integers(0, n, n)
        d = dice_mat[:, i].mean(1); e = err_mat[:, i].mean(1)
        bd, be = int(np.argmax(d)), int(np.argmin(e))
        if bd != be: dis += 1
        best_pairs[(names[bd], names[be])] = best_pairs.get((names[bd], names[be]), 0) + 1
        rd = np.argsort(np.argsort(-d)); re = np.argsort(np.argsort(e))
        rhos.append(spearmanr(rd, re).correlation); taus.append(kendalltau(rd, re).correlation)
    return dis / nboot, np.array(rhos), np.array(taus), best_pairs


def e3(common, gt, pred, dice):
    print("\n" + "=" * 100)
    print("E3  Dice 排序 vs 宽度误差排序：统计化")
    print("=" * 100)
    rng = np.random.default_rng(SEED)
    names = [NAME[c] for c in CONFIGS]
    D = np.vstack([dice[c] for c in CONFIGS])
    for st in ("w_median", "w_p95"):
        E = np.vstack([np.abs(pred[c][st] - gt[st]) for c in CONFIGS])
        print(f"\n[E3a] 跨配置（口径 {st}）：点估计  " + "  ".join(f"{names[k]} Dice {D[k].mean():.3f}/MAE {E[k].mean():.2f}" for k in range(len(names))))
        dis, rhos, taus, pairs = rank_boot(D, E, names, rng, NBOOT)
        print(f"      bootstrap {NBOOT} 次：Dice最优 ≠ MAE最优 的频率 {dis:.1%}；两排序 Spearman 中位 {np.median(rhos):+.2f}"
              f" [2.5%,97.5%] = [{np.percentile(rhos,2.5):+.2f},{np.percentile(rhos,97.5):+.2f}]；Kendall τ 中位 {np.median(taus):+.2f}")
        top = sorted(pairs.items(), key=lambda kv: -kv[1])[:4]
        print("      最常见的 (Dice最优, MAE最优) 组合：" + "；".join(f"{a}→{b} {v/NBOOT:.0%}" for (a, b), v in top))
        # 只在零样本三配置内
        zi = [k for k, c in enumerate(CONFIGS) if c[0] == "zeroshot"]
        dis_z, rhos_z, _, pairs_z = rank_boot(D[zi], E[zi], [names[k] for k in zi], rng, NBOOT)
        print(f"      仅零样本三配置：不一致频率 {dis_z:.1%}；Spearman 中位 {np.median(rhos_z):+.2f}")

    print(f"\n[E3b] 配置内逐图：Dice 高的图，宽度是否测得更准？（Spearman(Dice_i, |err_i|) 为负 = 是）")
    print(f"  {'配置':<10}{'ρ(Dice,|err| median)':>22}{'ρ(Dice,|err| p95)':>20}{'Dice<0.5 n':>11}{'ρ(pred,gt)|Dice<0.5':>20}{'0.5-0.7 n':>10}{'ρ':>7}{'>=0.7 n':>9}{'ρ':>7}")
    for c in CONFIGS:
        d = dice[c]; em = np.abs(pred[c]["w_median"] - gt["w_median"]); ep = np.abs(pred[c]["w_p95"] - gt["w_p95"])
        line = f"  {NAME[c]:<10}{spearmanr(d, em).correlation:>22.3f}{spearmanr(d, ep).correlation:>20.3f}"
        for k_, (lo, hi) in enumerate([(0, .5), (.5, .7), (.7, 1.01)]):
            m = (d >= lo) & (d < hi)
            rho = spearmanr(pred[c]["w_p95"][m], gt["w_p95"][m]).correlation if m.sum() >= 5 else np.nan
            wn, wr = (11, 20) if k_ == 0 else ((10, 7) if k_ == 1 else (9, 7))
            line += f"{m.sum():>{wn}}{rho:>{wr}.3f}"
        print(line)
    print("  判读：配置内 ρ(Dice,|err|) 显著为负、且高 Dice 分箱排序更好 → Dice 在配置内是有效的可用性门控；")
    print("        与 E3a 的跨配置不一致并不矛盾：Dice 不携带系统偏差（截距）的信息，而偏差决定了跨配置的 MAE 排序。")

    # 五种解码
    if WM:
        rows = list(csv.DictReader(open(WM[-1], encoding="utf-8")))
        decs = sorted({r["decode"] for r in rows})
        stems = sorted({r["stem"] for r in rows}, key=int)
        by = {d: {r["stem"]: r for r in rows if r["decode"] == d} for d in decs}
        stems = [s for s in stems if all(s in by[d] and by[d][s]["pred_w_median"] not in ("", "nan") for d in decs)]
        Dd = np.vstack([[fl(by[d][s]["dice"]) for s in stems] for d in decs])
        for st in ("w_median", "w_p95"):
            Ed = np.vstack([[abs(fl(by[d][s]["pred_" + st]) - fl(by[d][s]["gt_" + st])) for s in stems] for d in decs])
            dis, rhos, taus, pairs = rank_boot(Dd, Ed, decs, rng, NBOOT)
            print(f"\n[E3c] 单配置(zs/full)内 5 种解码（{os.path.basename(WM[-1])}，n={len(stems)}，口径 {st}）")
            print("      点估计  " + "  ".join(f"{decs[k]} Dice {Dd[k].mean():.4f}/MAE {Ed[k].mean():.2f}" for k in range(len(decs))))
            print(f"      bootstrap：Dice最优 ≠ MAE最优 频率 {dis:.1%}；排序 Spearman 中位 {np.median(rhos):+.2f}；"
                  + "最常见组合 " + "；".join(f"{a}→{b} {v/NBOOT:.0%}" for (a, b), v in sorted(pairs.items(), key=lambda kv: -kv[1])[:3]))
    else:
        print("\n[E3c] 未找到 width_mechanism_perimage_*.csv，跳过解码 bootstrap")


def main():
    sys.stdout = Tee(LOGP)
    print(f"e2e3_sensitivity_rank.py  {ts}")
    common, gt, pred, dice = load()
    e2(gt, pred)
    e3(common, gt, pred, dice)
    print(f"\n>> 日志: {LOGP}")


if __name__ == "__main__":
    main()