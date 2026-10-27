#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e4_scale_sensitivity.py —— 像素→毫米尺度 s 的灵敏度（回应审稿 M13），纯解析 + 传播到 126 条裂缝

s0 = (d35 / N) * (h / f35)      d35=43.267 mm 全画幅对角, N=5040 px 图像对角, h=672 mm, f35=26.0 mm  → 0.2219 mm/px
焦距像素 f_px = N * f35 / d35 ≈ 3028 px；长边半视场 atan(2016/f_px)≈33.7°，短边半视场 atan(1512/f_px)≈26.5°

倾斜模型（平面路面，相机高 h，光轴偏离铅垂 θ，像点偏离光轴 φ，均在倾斜方向上）：
  地面坐标 x(φ) = h·tan(θ+φ)，像面坐标 u(φ) = f_px·tan φ
  沿倾斜方向的局部尺度  s_∥(φ) = dx/du = (h/f_px) · cos²φ / cos²(θ+φ)
  垂直倾斜方向的局部尺度 s_⊥(φ) = (h/f_px) · cos φ / cos(θ+φ)   （物距 h/cos(θ+φ) 除以 f_px/cos φ）
  θ=0 时两者都退化为 s0，与画面位置无关；θ>0 时 s 随 φ 变化 —— 画面内尺度梯度，常数 s 假设失效。
  裂缝宽度可能沿任一方向，故取 [min(s_⊥,s_∥), max(s_⊥,s_∥)] 在 φ∈[−φmax, φmax] 上的范围作为逐图不确定区间。

传播：若真实尺度在 [k_lo·s0, k_hi·s0] 内，则以 s0 测得的宽度 w 的真实值在 [k_lo·w, k_hi·w]；
      等级"可能翻转"⇔ 该区间跨过阈值。统计 GT P95（ASTM 10 mm）与 GT 中位（JTG 3 mm）上可能翻转的比例。
输出 results/e4_scale_<ts>.txt
"""
import os, sys, csv, datetime
import numpy as np

OUT = "/root/autodl-tmp/proj_bridge"
CL = os.path.join(OUT, "results", "closeloop_v2_20260726_170420.csv")
D35, N, H, F35 = 43.267, 5040.0, 672.0, 26.0
LONG, SHORT = 4032, 3024
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"e4_scale_{ts}.txt")


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()


def s_scalar(h=H, f35=F35, d35=D35, n=N): return d35 / n * h / f35


def tilt_range(theta_deg, phimax_deg, s0):
    """θ 倾斜下，φ∈[−φmax,φmax] 上 s_∥ 与 s_⊥ 的最小/最大值（相对 s0 的倍数）"""
    th = np.radians(theta_deg); ph = np.radians(np.linspace(-phimax_deg, phimax_deg, 2001))
    par = np.cos(ph) ** 2 / np.cos(th + ph) ** 2
    per = np.cos(ph) / np.cos(th + ph)
    return min(par.min(), per.min()), max(par.max(), per.max()), par, per


def flip_frac(w, T, klo, khi):
    """以 s0 测得 w，真实尺度倍数 ∈[klo,khi] → 真实宽度 ∈[klo·w, khi·w]，跨过 T 即可能翻转"""
    return np.mean((klo * w <= T) & (khi * w >= T))


def main():
    sys.stdout = Tee(LOGP)
    print(f"e4_scale_sensitivity.py  {ts}")
    rows = [r for r in csv.DictReader(open(CL, encoding="utf-8"))
            if r["gt_ctype"] == "线状" and r["variant"] == "zeroshot" and r["cond"] == "text_full"]
    g95 = np.array([float(r["gt_w_p95"]) for r in rows]); gmed = np.array([float(r["gt_w_median"]) for r in rows])
    s0 = s_scalar(); fpx = N * F35 / D35
    phi_long = np.degrees(np.arctan(LONG / 2 / fpx)); phi_short = np.degrees(np.arctan(SHORT / 2 / fpx))
    print(f"  s0 = {s0:.4f} mm/px   f_px = {fpx:.0f} px   半视场 长边 {phi_long:.1f}° / 短边 {phi_short:.1f}°   n={len(g95)}")
    print(f"  参考宽度：GT P95 中位 {np.median(g95):.2f} mm；GT 中位宽中位 {np.median(gmed):.2f} mm")

    # ---- [1] 标量参数灵敏度 ----
    print("\n[1] 标量参数（整幅图同一倍数）：s 的相对变化 与 可能翻转的等级比例")
    print(f"  {'扰动':<34}{'s mm/px':>16}{'Δs':>14}{'ASTM 10mm 可翻转':>18}{'JTG 3mm 可翻转':>16}")
    cases = [("h ±20 mm（手持高度误差 3%）", [s_scalar(h=H - 20), s_scalar(h=H + 20)]),
             ("h ±50 mm（7.4%）", [s_scalar(h=H - 50), s_scalar(h=H + 50)]),
             ("f35 26.0 → 25.5 / 26.5（EXIF 取整）", [s_scalar(f35=26.5), s_scalar(f35=25.5)]),
             ("光学 0.2219 vs 长度法 0.2175（论文 2%）", [0.2175, 0.2219])]
    for name, (slo, shi) in cases:
        klo, khi = slo / s0, shi / s0
        ds = f"{(klo-1):+.1%}/{(khi-1):+.1%}"
        print(f"  {name:<34}{f'{slo:.4f}–{shi:.4f}':>16}{ds:>14}"
              f"{flip_frac(g95, 10.0, klo, khi):>18.1%}{flip_frac(gmed, 3.0, klo, khi):>16.1%}")

    # ---- [2] 倾斜：画面内尺度梯度 ----
    print("\n[2] 相机倾斜 θ：画面内尺度倍数范围（相对 s0），以及可能翻转的等级比例")
    print("    （逐图范围取 φ∈[−φmax, φmax]，φmax 用长边半视场；'中心'为 φ=0 处的倍数）")
    print(f"  {'θ':>4}{'中心 s_∥/s0':>12}{'中心 s_⊥/s0':>12}{'全画面 min':>12}{'全画面 max':>12}{'跨度':>8}{'ASTM 可翻转':>13}{'JTG 可翻转':>12}{'短边轴 min–max':>18}")
    for th in (0, 1, 2, 5, 10, 15):
        klo, khi, par, per = tilt_range(th, phi_long, s0)
        klo_s, khi_s, _, _ = tilt_range(th, phi_short, s0)
        c_par = 1 / np.cos(np.radians(th)) ** 2; c_per = 1 / np.cos(np.radians(th))
        print(f"  {th:>3}°{c_par:>12.3f}{c_per:>12.3f}{klo:>12.3f}{khi:>12.3f}{khi-klo:>8.3f}"
              f"{flip_frac(g95, 10.0, klo, khi):>13.1%}{flip_frac(gmed, 3.0, klo, khi):>12.1%}{f'{klo_s:.3f}–{khi_s:.3f}':>18}")
    print("  判读：θ=0 时全画面倍数恒为 1（常数 s 假设成立）；θ>0 时同一幅图内尺度即随位置变化，")
    print("        且该梯度对'总长度/骨架像素数'的一致性检验是不可见的（沿画面积分互相抵消）。")
    print("        论文应把 s 的不确定性表述为一个逐图区间而非 2% 的常数，并说明未记录倾角是数据集的限制。")

    # ---- [3] 分布视角 ----
    print("\n[3] 距阈值的相对距离分布（决定尺度误差多大时等级才会动）")
    for name, w, T in (("ASTM 10 mm / GT P95", g95, 10.0), ("JTG 3 mm / GT 中位", gmed, 3.0)):
        rel = np.abs(np.log(w / T))
        print(f"  {name}: |log(w/T)| 分位  P10 {np.percentile(rel,10):.3f}  P25 {np.percentile(rel,25):.3f}  中位 {np.median(rel):.3f}；"
              f"  尺度误差 5% / 10% / 20% 内可翻转：{np.mean(rel<=np.log(1.05)):.1%} / {np.mean(rel<=np.log(1.10)):.1%} / {np.mean(rel<=np.log(1.20)):.1%}")
    print("  判读：这就是校准式概率评级要吸收的另一项不确定度来源；(b,σ) 是在同一拍摄几何下估的，隐含吸收了恒定部分，")
    print("        但吸收不了倾斜带来的画面内梯度。")
    print(f"\n>> 日志: {LOGP}")


if __name__ == "__main__":
    main()