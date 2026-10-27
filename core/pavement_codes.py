#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pavement_codes.py —— 规范阈值 + 裂缝几何量 的【单一真值来源】
所有脚本一律 import 本模块，禁止在别处硬编码阈值。
（本项目曾因阈值被复制进多个脚本而使用了错误值，务必保持此约定）

阈值核实记录 2026-07：
  · JTG 5210-2018 §5.2.3纵向裂缝 / §5.2.4横向裂缝
      轻度: 主要裂缝宽度 <= 3mm ； 重度: > 3mm 。【仅两级，无"中度"，无10mm阈值】
  · JTG 5210-2018 §5.2.1龟裂  (三级，但同时依赖"块度"，宽度分支不可单独判定)
  · JTG 5210-2018 §5.2.2块状裂缝 (两级，同样依赖块度)
  · ASTM D6433 病害(10) 纵横向裂缝 (据 USACE ERDC-CERL PAVER 沥青路面病害手册 2009)
      L: 未填缝宽 < 3/8in(10mm)，或任意宽度已填缝(填缝料良好)
      M: 未填缝宽 >=10mm 且 <3in(75mm)，或 <=75mm但周围有轻度随机裂缝，或已填缝周围有轻度随机裂缝
      H: 周围有中/重度随机裂缝，或未填缝宽 >75mm，或裂缝周围约100mm路面严重破碎
  · ASTM D6433 病害(17) 滑移裂缝: L<10 / M 10~38 / H>38mm
      —— 本项目早期误把这组当成纵横向裂缝使用，保留于此仅作对照，勿用于线状裂缝

掩码不可判定的定性分支（论文须界定范围）：
  ASTM 已填缝(灌缝维修)裂缝无论多宽判L —— 图像上仍呈深色线条，掩码无法区分【最强限制】
  ASTM 周围随机裂缝程度、周围100mm破碎程度
  JTG  龟裂/块状裂缝所需的"块度"，以及裂缝类型(纵/横/龟/块)的前置分类
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

# ============ 规范阈值 (mm) ============
JTG_LIN   = [("轻度", 0.0, 3.0), ("重度", 3.0, 1e9)]              # 纵/横向裂缝，两级
JTG_ALLI  = [("轻度", 0.0, 2.0), ("中度", 2.0, 5.0), ("重度", 5.0, 1e9)]   # 龟裂，需配合块度
JTG_BLOCK = [("轻度", 1.0, 2.0), ("重度", 2.0, 1e9)]              # 块状裂缝，需配合块度
ASTM_LT   = [("L", 0.0, 10.0), ("M", 10.0, 75.0), ("H", 75.0, 1e9)]       # 纵横向裂缝(10)
ASTM_SLIP = [("L", 0.0, 10.0), ("M", 10.0, 38.0), ("H", 38.0, 1e9)]       # 滑移裂缝(17)，勿误用

# 极端背离带：JTG判最高档(重度) 而 ASTM判最低档(L)
DIVERGENCE_BAND = (3.0, 10.0)

# ============ 口径 ============
# ASTM 手册规定：一条裂缝若沿全长严重程度不一且无法分段，按其存在的最高严重程度评定 -> 偏高分位
# JTG  线状写"主要裂缝宽度"、龟裂写"平均裂缝宽度"                                -> 偏中位/均值
CALIPER = {"JTG": "w_median", "ASTM": "w_p95"}

# ============ 标定常数 ============
# SUT-Crack: EXIF 全部为 samsung SM-G780F, f=5.4mm, f35=26.0mm; 数据集载明拍摄高度 672mm
#   mm/px = (43.267mm 全画幅对角 / 5040px 图像对角) x (672 / 26.0) = 0.2219
#   独立佐证: 真实裂缝长度/骨架像素数 经验估计 0.2175 (n=130)，两者相差 2.0%
# ⚠ 该常数仅适用于 SUT-Crack。其他数据集(RDD/DeepCrack/NHA12D/Crack500)无标定，
#   严禁假设标定后做绝对mm分级（已验证会导致近乎全判重度，误导性强）
SUTCRACK_MMPX = 0.2219

K8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def grade(width_mm, table):
    """按规范表把缝宽(mm)映射为等级名"""
    for name, lo, hi in table:
        if lo <= width_mm < hi:
            return name
    return table[-1][0] if width_mm >= table[-1][1] else table[0][0]


def in_divergence_band(width_mm):
    lo, hi = DIVERGENCE_BAND
    return lo < width_mm < hi


def skeleton_stats(mask):
    """
    骨架拓扑量。关键：用【环数】(欧拉示性数 E-V+C) 判网状，不要用分叉像素数——
    一个Y型交叉常由2~4个像素同时满足"邻居>=3"，分叉像素数会系统性放大2~3倍，
    与端点数不可比（本项目曾据此误判形态）。
    """
    S = skeletonize(mask)
    if S.sum() == 0:
        return None
    nb = ndi.convolve(S.astype(np.uint8), K8, mode='constant') * S
    n_end = int(((nb == 1) & S).sum())
    n_branch_px = int(((nb >= 3) & S).sum())
    V = int(S.sum())
    ortho = int((S[:, :-1] & S[:, 1:]).sum()) + int((S[:-1, :] & S[1:, :]).sum())
    d1 = S[:-1, :-1] & S[1:, 1:] & ~(S[:-1, 1:] | S[1:, :-1])
    d2 = S[:-1, 1:] & S[1:, :-1] & ~(S[:-1, :-1] | S[1:, 1:])
    E = ortho + int(d1.sum()) + int(d2.sum())
    _, C = ndi.label(S, structure=np.ones((3, 3), np.uint8))
    filled = ndi.binary_fill_holes(mask)
    _, n_holes = ndi.label(filled & ~mask)
    return dict(skel=S, n_skel=V, L_edge=float(ortho + np.sqrt(2.0) * (int(d1.sum()) + int(d2.sum()))),
                n_end=n_end, n_branch_px=n_branch_px, n_comp=C,
                cycles=int(E - V + C), n_holes=int(n_holes))


def enclosed_blocks(mask, st=None, mm_per_px=None, min_block_mm2=2500.0,
                    min_block_side_factor=12.0):
    """
    骨架闭合环所围合的区块 —— 同时解决两件事：
      1) 线状/网状判别（比环数可靠：掩码针眼孔洞也会产生环，但围合面积极小）
      2) 直接产出 JTG §5.2.1龟裂 / §5.2.2块状裂缝 所需的【块度】
    min_block_mm2 默认 2500 mm²(约50mm见方)，小于此视为掩码噪声而非路面块。
    无标定时按像素统计，min_block 退化为骨架长度平方的经验比例。
    """
    if st is None:
        st = skeleton_stats(mask)
    if st is None:
        return dict(n_blocks=0, block_sides_mm=[], block_side_median_mm=None,
                    block_side_max_mm=None, enclosed_frac=0.0)
    S = st["skel"]
    enc = ndi.binary_fill_holes(S) & ~S
    lab, n = ndi.label(enc)
    if n == 0:
        return dict(n_blocks=0, block_sides_mm=[], block_side_median_mm=None,
                    block_side_max_mm=None, enclosed_frac=0.0)
    areas = np.asarray(ndi.sum(enc, lab, range(1, n + 1)), float)

    # 门槛 = max(物理下限, 宽度相对下限)
    # 宽度相对下限是关键：真实路面块的边长为裂缝宽度的十几倍以上，该比值无量纲、与图像尺度无关。
    # ⚠ 旧版无标定分支写成 0.001*n_skel**2（随骨架长度平方增长），量纲错误，会把判据整个反转：
    #    骨架长的图门槛高到滤掉全部环、骨架短的图一个环就过关。已于 2026-07 修正。
    edt_med = float(np.median(ndi.distance_transform_edt(mask)[S] * 2.0))
    min_area_px = (min_block_side_factor * max(edt_med, 1.0)) ** 2
    if mm_per_px:
        min_area_px = max(min_area_px, min_block_mm2 / (mm_per_px ** 2))
    keep = areas[areas >= min_area_px]
    sides = np.sqrt(keep) * (mm_per_px if mm_per_px else 1.0)   # 有标定->mm, 否则->px
    return dict(n_blocks=int(len(keep)),
                block_sides_mm=[float(x) for x in sides],
                block_side_median_mm=(float(np.median(sides)) if len(sides) else None),
                block_side_max_mm=(float(sides.max()) if len(sides) else None),
                enclosed_frac=float(enc.sum() / max(1, mask.sum())))


def classify_morphology(st, blocks=None):
    """'线状' -> 适用 JTG §5.2.3/5.2.4 与 ASTM 病害(10) 的缝宽阈值
       '网状' -> 须走 JTG §5.2.1龟裂/§5.2.2块状(依赖块度)，且 ASTM 龟裂判据为纯定性，
                 无宽度阈值，本方法无法映射，应从绝对分级中剔除
    判据为【闭合环围合面积】而非环数：掩码针眼孔洞同样产生环，但围不出路面块。
    ⚠ 网状正样本目前仅1例(SUT-Crack #10)，用于全量时须把命中样本拼图人工复核。"""
    if st is None:
        return "空"
    if blocks is None:
        return "网状" if st["cycles"] >= 2 else "线状"
    return "网状" if blocks["n_blocks"] >= 1 else "线状"


def crack_width_px(mask, st=None, exclude_junctions=False, junction_radius_factor=1.0):
    """
    缝宽(像素) = 骨架点距离变换值 x 2。
    exclude_junctions: 剔除交叉点邻域再统计（经SUT-Crack全量验证，剔与不剔差异<2%，
                       默认关闭；保留开关以备粗糙掩码时使用）
    注意：用 distance_transform_edt 求"离分叉点的距离"，不要用大结构元 binary_dilation——
          后者是 O(k^2)，在12MP图上会慢一个数量级（本项目曾因此单次多耗约30分钟）。
    """
    if st is None:
        st = skeleton_stats(mask)
    if st is None:
        return None
    S = st["skel"]
    edt = ndi.distance_transform_edt(mask)
    sel = S
    if exclude_junctions:
        nb = ndi.convolve(S.astype(np.uint8), K8, mode='constant') * S
        br = (nb >= 3) & S
        if br.any():
            w_all_med = float(np.median(edt[S] * 2.0))
            r = max(2.0, w_all_med * junction_radius_factor)
            dist_to_br = ndi.distance_transform_edt(~br)   # O(n)，替代大核膨胀
            cand = S & (dist_to_br > r)
            if cand.sum() > 50:
                sel = cand
    w = edt[sel] * 2.0
    if w.size == 0:
        return None
    return dict(w_median=float(np.median(w)), w_mean=float(w.mean()),
                w_p90=float(np.percentile(w, 90)), w_p95=float(np.percentile(w, 95)),
                w_max=float(w.max()), n_pts=int(w.size))


def measure(mask, mm_per_px=None, exclude_junctions=False):
    """一站式：拓扑 + 形态 + 缝宽(px，若给标定则附mm) + 面积/密度"""
    if mask.sum() == 0:
        return None
    st = skeleton_stats(mask)
    if st is None:
        return None
    w = crack_width_px(mask, st, exclude_junctions=exclude_junctions)
    if w is None:
        return None
    blk = enclosed_blocks(mask, st, mm_per_px=mm_per_px)
    out = dict(area_px=int(mask.sum()), density_pct=float(mask.mean() * 100),
               n_skel=st["n_skel"], L_edge=st["L_edge"], n_end=st["n_end"],
               n_branch_px=st["n_branch_px"], n_comp=st["n_comp"],
               cycles=st["cycles"], n_holes=st["n_holes"],
               n_blocks=blk["n_blocks"], block_side_median=blk["block_side_median_mm"],
               block_side_max=blk["block_side_max_mm"], enclosed_frac=blk["enclosed_frac"],
               ctype=classify_morphology(st, blk))
    out.update(w)
    if mm_per_px:
        for k in ("w_median", "w_mean", "w_p90", "w_p95", "w_max"):
            out[k + "_mm"] = out[k] * mm_per_px
        out["mm_per_px"] = mm_per_px
    return out


def relative_levels(values, q=(33, 66), names=("轻", "中", "重")):
    """相对严重度分级（分位数）。本项目已验证：可靠性单调依赖掩码质量
       (Dice<0.1 -> Spearman 0.086 失效； Dice 0.6 -> 0.678； Dice>0.5子集 -> 0.854)"""
    v = np.asarray(values, float)
    t1, t2 = np.percentile(v, q[0]), np.percentile(v, q[1])
    return [names[0] if x < t1 else (names[1] if x < t2 else names[2]) for x in v], float(t1), float(t2)