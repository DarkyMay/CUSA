#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e5e7_timing_collapse.py —— E5 计算成本表 + E7 微调宽度塌缩曲线（GPU）

E5  在前 N 张 SUT-Crack 图上分阶段计时：processor / SAM3 前向 / 上采样+二值化 / 几何测量(骨架+EDT)；
    峰值显存；给出每张 12MP 图的端到端时延。回应审稿 M14。
E7  对 ckpts/v3/*.pt、ckpts/v2/*.pt（可选 lora）逐个 load_state_dict(strict=False)，在 126 条线状裂缝上
    计算 Dice、前景比、w_median / w_p95 的偏差%、MAE、Spearman、线性斜率 b 与截距 a、模糊地板 f。
    看斜率 b 是否随 epoch 单调下降（宽度塌缩），把 width_mechanism 里 ft_v3 的 b=0.16 变成一条曲线。
用法：python3 e5e7_timing_collapse.py --limit 12     # 冒烟
      python3 e5e7_timing_collapse.py                # 全量，约 40–50 min（13 ckpt + zeroshot，126 张）
输出：results/e5e7_<ts>.txt, results/e7_perimage_<ts>.csv
"""
import os, sys, csv, glob, re, time, argparse, datetime
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image, ImageOps
from scipy.stats import spearmanr
from transformers import Sam3Model, Sam3Processor

sys.path.insert(0, "/root/autodl-tmp/proj_bridge")
import pavement_codes as pc

Image.MAX_IMAGE_PIXELS = None
OUT = "/root/autodl-tmp/proj_bridge"
W = "/root/autodl-tmp/sam3_weights"
ROOT = "/root/autodl-tmp/SUT-Crack"
CL = os.path.join(OUT, "results", "closeloop_v2_20260726_170420.csv")
DEV = "cuda"; TEXT = "crack"; MMPX = pc.SUTCRACK_MMPX
STATS = ["w_median", "w_p95"]
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"e5e7_{ts}.txt")
CSVP = os.path.join(OUT, "results", f"e7_perimage_{ts}.csv")


def _resolve(*c):
    for x in c:
        if os.path.isdir(x): return x
    return c[0]
IMGD = _resolve(os.path.join(ROOT, "1-Segmentation", "Original_Image"), os.path.join(ROOT, "1-Segmentation", "Original Image"))
GTD = _resolve(os.path.join(ROOT, "1-Segmentation", "Ground_Truth"), os.path.join(ROOT, "1-Segmentation", "Ground Truth"))


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()


def numkey(s): m = re.findall(r'\d+', s); return int(m[0]) if m else 0
def fl(x):
    try: return float(x)
    except Exception: return np.nan
def dice_np(a, b): s = a.sum() + b.sum(); return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def geom(mask):
    if mask.sum() < 30: return None
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - 60), min(mask.shape[0], ys.max() + 61)
    x0, x1 = max(0, xs.min() - 60), min(mask.shape[1], xs.max() + 61)
    return pc.measure(mask[y0:y1, x0:x1], mm_per_px=MMPX)


def linfit(x, y):
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 5: return np.nan, np.nan, np.nan
    b, a = np.polyfit(x, y, 1); r2 = 1 - np.sum((y - a - b * x) ** 2) / np.sum((y - y.mean()) ** 2)
    return a, b, r2


@torch.no_grad()
def infer(model, processor, img, timing=None):
    t0 = time.perf_counter()
    inp = processor(images=img, text=TEXT, return_tensors="pt")
    inp = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}
    torch.cuda.synchronize(); t1 = time.perf_counter()
    with torch.autocast(DEV, dtype=torch.bfloat16):
        o = model(**inp)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    Wd, Hd = img.size
    m = (F.interpolate(o["semantic_seg"][0, 0][None, None].float(), size=(Hd, Wd), mode="bilinear")[0, 0] > 0).cpu().numpy()
    torch.cuda.synchronize(); t3 = time.perf_counter()
    if timing is not None:
        timing["processor"].append(t1 - t0); timing["forward"].append(t2 - t1); timing["upsample"].append(t3 - t2)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--n-timing", type=int, default=25)
    ap.add_argument("--with-lora", action="store_true", help="尝试加载 ckpts/lora/*.pt（通常键名不匹配会跳过）")
    args = ap.parse_args()
    sys.stdout = Tee(LOGP)
    T0 = time.time()
    print(f"e5e7_timing_collapse.py  {ts}  torch {torch.__version__}  GPU {torch.cuda.get_device_name(0)}")

    # ---- 数据：线状 stem 与 GT 宽度来自 closeloop csv，GT 掩膜用于 Dice ----
    cl = [r for r in csv.DictReader(open(CL, encoding="utf-8"))
          if r["gt_ctype"] == "线状" and r["variant"] == "zeroshot" and r["cond"] == "text_full"]
    gtw = {r["stem"]: r for r in cl}
    stems = sorted(gtw, key=numkey)
    if args.limit: stems = stems[:args.limit]
    print(f"  线状图像 {len(stems)} 张   IMGD={IMGD}")

    processor = Sam3Processor.from_pretrained(W)
    model = Sam3Model.from_pretrained(W).to(DEV); model.eval(); torch.set_grad_enabled(False)
    for p in model.parameters(): p.requires_grad_(False)
    base_sd = {k: v.clone() for k, v in model.state_dict().items()}
    nparam = sum(p.numel() for p in model.parameters())
    print(f"  SAM3 参数量 {nparam/1e6:.1f} M")

    def load_img(stem):
        img = ImageOps.exif_transpose(Image.open(os.path.join(IMGD, stem + ".jpg"))).convert("RGB")
        gt = np.array(Image.open(os.path.join(GTD, stem + ".png")).convert("L")) > 127
        return img, gt

    # ============================== E5 计时 ==============================
    print("\n" + "=" * 100); print("E5  计算成本（zeroshot text_full，12MP 输入，bf16 autocast）"); print("=" * 100)
    timing = {k: [] for k in ("read", "processor", "forward", "upsample", "geom_pred", "geom_gt_full", "geom_gt_crop")}
    torch.cuda.reset_peak_memory_stats()
    # 预热
    img, gt = load_img(stems[0]); infer(model, processor, img)
    for stem in stems[:args.n_timing]:
        t = time.perf_counter(); img, gt = load_img(stem); timing["read"].append(time.perf_counter() - t)
        m = infer(model, processor, img, timing)
        t = time.perf_counter(); geom(m); timing["geom_pred"].append(time.perf_counter() - t)
        t = time.perf_counter(); pc.measure(gt, mm_per_px=MMPX); timing["geom_gt_full"].append(time.perf_counter() - t)
        t = time.perf_counter(); geom(gt); timing["geom_gt_crop"].append(time.perf_counter() - t)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    print(f"  n={len(timing['forward'])}   峰值显存 {peak:.2f} GB")
    print(f"  {'阶段':<22}{'中位 s':>10}{'均值 s':>10}{'P90 s':>10}")
    for k, v in timing.items():
        if v: print(f"  {k:<22}{np.median(v):>10.3f}{np.mean(v):>10.3f}{np.percentile(v,90):>10.3f}")
    e2e = np.array(timing["processor"]) + np.array(timing["forward"]) + np.array(timing["upsample"]) + np.array(timing["geom_pred"])
    print(f"  端到端(不含读图) 中位 {np.median(e2e):.3f} s → {1/np.median(e2e):.2f} 图/s；SAM3 前向占 {np.sum(timing['forward'])/np.sum(e2e):.0%}")
    print("  说明：几何测量在 bbox 裁剪上做（geom_pred/geom_gt_crop）；geom_gt_full 为不裁剪的整幅 12MP 参考。")
    print("        校准/概率评级为闭式公式，成本可忽略。")

    # ============================== E7 塌缩曲线 ==============================
    print("\n" + "=" * 100); print("E7  微调 checkpoint 的宽度响应（pred = a + b·gt；b→0 即宽度塌缩）"); print("=" * 100)
    ck_list = [("zeroshot", 0, None)]
    for tag, pat in (("v3", "ckpts/v3/sam3_v3_ep*.pt"), ("v2", "ckpts/v2/sam3_md_v2_ep*.pt")):
        for p in sorted(glob.glob(os.path.join(OUT, pat)), key=lambda x: numkey(os.path.basename(x).split("ep")[-1])):
            ep = int(re.findall(r'ep(\d+)', os.path.basename(p))[0]); ck_list.append((tag, ep, p))
    for tag, name in (("v3", "sam3_ft_v3_textonly_best.pt"), ("v2", "sam3_ft_multidomain_v2_best.pt")):
        p = os.path.join(OUT, "ckpts", name)
        if os.path.exists(p): ck_list.append((tag + "_best", -1, p))
    if args.with_lora:
        for p in sorted(glob.glob(os.path.join(OUT, "ckpts/lora/lora_r16_ep*.pt")), key=lambda x: numkey(os.path.basename(x).split("ep")[-1])):
            ck_list.append(("lora", int(re.findall(r'ep(\d+)', os.path.basename(p))[0]), p))
    print(f"  待评估：{[(t, e) for t, e, _ in ck_list]}")

    # GT 掩膜缓存（Dice 用）与 GT 宽度
    GT = {}
    for stem in stems:
        _, gt = load_img(stem); GT[stem] = gt
    g_med = np.array([fl(gtw[s]["gt_w_median"]) for s in stems]); g_p95 = np.array([fl(gtw[s]["gt_w_p95"]) for s in stems])

    rows = []; summary = []
    for tag, ep, path in ck_list:
        model.load_state_dict(base_sd, strict=True)
        if path:
            sd = torch.load(path, map_location=DEV)
            if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
            if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict): sd = sd["model"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            nmatch = len(sd) - len(unexpected)
            if unexpected:
                print(f"  !! {tag} ep{ep}: {len(unexpected)} 个键不匹配（示例 {list(unexpected)[:2]}），跳过"); continue
            print(f"  {tag} ep{ep}: 载入 {nmatch} 个张量", flush=True)
        model.eval()
        t0 = time.time()
        d, fg, pm, pp = [], [], [], []
        for i, stem in enumerate(stems):
            img, _ = load_img(stem); gt = GT[stem]
            m = infer(model, processor, img); g = geom(m)
            d.append(dice_np(m, gt)); fg.append(m.mean() / gt.mean() if gt.mean() > 0 else np.nan)
            pm.append(g["w_median_mm"] if g else np.nan); pp.append(g["w_p95_mm"] if g else np.nan)
            rows.append(dict(tag=tag, ep=ep, stem=stem, dice=d[-1], fg_ratio=fg[-1],
                             gt_w_median=g_med[i], pred_w_median=pm[-1], gt_w_p95=g_p95[i], pred_w_p95=pp[-1]))
        d, fg, pm, pp = map(np.array, (d, fg, pm, pp))
        rec = dict(tag=tag, ep=ep, dice=np.nanmean(d), fg=np.nanmean(fg), n_empty=int(np.isnan(pm).sum()))
        for st, g, p in (("w_median", g_med, pm), ("w_p95", g_p95, pp)):
            ok = np.isfinite(p)
            a, b, r2 = linfit(g[ok], p[ok]); fa, fk, _ = linfit(g[ok] ** 2, p[ok] ** 2)
            rec[st] = dict(bias=(p[ok].mean() - g[ok].mean()) / g[ok].mean(), mae=np.mean(np.abs(p[ok] - g[ok])),
                           rho=spearmanr(p[ok], g[ok]).correlation, a=a, b=b, r2=r2,
                           f=np.sqrt(max(fa, 0)) if np.isfinite(fa) else np.nan, k=fk)
        summary.append(rec)
        print(f"    {tag} ep{ep}: Dice {rec['dice']:.4f} fg {rec['fg']:.2f} | median: bias {rec['w_median']['bias']:+.1%} "
              f"b={rec['w_median']['b']:.2f} ρ={rec['w_median']['rho']:.3f} | 用时 {(time.time()-t0)/60:.1f} min  累计 {(time.time()-T0)/60:.0f} min", flush=True)

    with open(CSVP, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    print("\n[E7 汇总]  n=%d 线状裂缝；斜率 b 接近 1 = 宽度响应完整，接近 0 = 塌缩；a = 截距 mm；f = 模糊地板 mm" % len(stems))
    print(f"  {'变体':<9}{'ep':>4}{'Dice':>8}{'前景比':>7}{'空掩膜':>7}"
          + "".join(f"{s+' bias':>12}{'MAE':>7}{'ρ':>7}{'a':>7}{'b':>6}{'R2':>6}{'f':>6}" for s in STATS))
    for rec in summary:
        line = f"  {rec['tag']:<9}{rec['ep']:>4}{rec['dice']:>8.4f}{rec['fg']:>7.2f}{rec['n_empty']:>7}"
        for st in STATS:
            r = rec[st]
            line += f"{r['bias']:>+12.1%}{r['mae']:>7.2f}{r['rho']:>7.3f}{r['a']:>7.2f}{r['b']:>6.2f}{r['r2']:>6.2f}{r['f']:>6.2f}"
        print(line)
    print("\n  判读：")
    print("   · 若 v3/v2 的 b 随 epoch 单调下降、a 趋于常数，则微调使宽度对真实宽度失去响应（塌缩），是与零样本地板不同的失效；")
    print("   · 若 ep1 就已塌缩，则塌缩来自训练目标/数据（512×512 训练掩膜的宽度分布），不是过拟合；")
    print("   · 前景比 <1 且 b 小：欠分割 + 塌缩同时存在；Dice 的变化幅度远小于 b 的变化幅度即再次说明 Dice 不反映宽度保真。")
    print(f"\n>> 逐图CSV: {CSVP}\n>> 日志: {LOGP}\n总耗时 {(time.time()-T0)/60:.1f} min")


if __name__ == "__main__":
    main()