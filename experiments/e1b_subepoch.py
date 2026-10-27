#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e1b_subepoch.py —— R2-6 补充：塌缩发生在第一个 epoch 内部，看亚 epoch 早停 + 低学习率 + 面积项能否保住斜率

复用 e1_collapse_mitigation.py 的数据/模型/损失函数（import），自己写训练循环以便按图像数评测。
变体: A      v3 原样 (lr 1e-4)
      C      + 面积保持损失 (lr 1e-4)
      A_lo   v3 原样, lr 1e-5
      C_lo   + 面积保持损失, lr 1e-5
每个变体 × 3 seed × 1 epoch，在看过 0/60/120/240/400/620 张后各评一次：
  四集 Dice、held-out 均值、NHA 前景比 r、SUT 126 线状的 斜率/截距/ρ/MAE（中值）+ 逐图 median & p95 宽度落盘
输出: results/e1b_subepoch_<ts>.txt, e1b_eval_<ts>.csv, e1b_widths_<ts>.csv
用法: python3 e1b_subepoch.py [--quick] [--variants A,C,A_lo,C_lo] [--seeds 1234,2025,7]
"""
import os, sys, csv, time, random, argparse, datetime, traceback
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr

sys.path.insert(0, "/root/autodl-tmp/proj_bridge")
import e1_collapse_mitigation as E
from transformers import Sam3Model, Sam3Processor

OUT = E.OUT; DEV = E.DEV
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"e1b_subepoch_{ts}.txt")
CSV_EVAL = os.path.join(OUT, "results", f"e1b_eval_{ts}.csv")
CSV_W = os.path.join(OUT, "results", f"e1b_widths_{ts}.csv")
CHECKPOINTS = [0, 60, 120, 240, 400, 620]
SPEC = {"A": (False, 1e-4), "C": (True, 1e-4), "A_lo": (False, 1e-5), "C_lo": (True, 1e-5)}


def sut_widths(mask):
    """中值与 P95（mm）。骨架点处 2*EDT，与 pc.measure 的 w_median / w_p95 逐图精确一致（b 脚本 B0 已验证）；
    注意 pc.measure 的 w_* 键是像素值，故这里统一在像素上算再乘 MMPX。"""
    r = E.width_profile_px(mask)
    return (0.0, 0.0) if r is None else (r[0] * E.MMPX, r[2] * E.MMPX)


@torch.no_grad()
def eval_point(model, processor, tag, nseen, seed, quick, sut_ref, sut_p95, sut_linear, erows, wrows):
    lim = 6 if quick else 0
    rec = dict(tag=tag, variant=tag.split("@")[0], seed=seed, n_seen=nseen)
    for t in ["dataset_400", "DeepCrack", "NHA12D", "Crack500"]:
        d, r = E.eval_dice(model, processor, t, lim); rec[f"dice_{t}"] = d; rec[f"r_{t}"] = r
    rec["heldout_mean"] = (rec["dice_NHA12D"] + rec["dice_Crack500"]) / 2
    xs = []; ys = []; xs95 = []; ys95 = []
    pairs = E.eval_pairs("SUT"); pairs = pairs[:lim] if lim else pairs
    model.eval()
    for ip, mp, thr in pairs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        if stem not in sut_linear: continue
        try:
            img = E.open_img(ip)
            with torch.autocast(DEV, dtype=torch.bfloat16):
                sm = E.sem_mask(model, processor, img)
            wmed, w95 = sut_widths(sm)
        except Exception:
            torch.cuda.empty_cache(); continue
        xs.append(sut_ref[stem]); ys.append(wmed); xs95.append(sut_p95[stem]); ys95.append(w95)
        wrows.append(dict(tag=tag, variant=rec["variant"], seed=seed, n_seen=nseen, stem=stem, gt_med=sut_ref[stem], pred_med=wmed, gt_p95=sut_p95[stem], pred_p95=w95))
    xs = np.array(xs); ys = np.array(ys)
    if len(xs) >= 5:
        b, a = np.polyfit(xs, ys, 1)
        rec.update(slope_SUT=float(b), icpt_SUT=float(a), rho_SUT=float(spearmanr(xs, ys).correlation), mae_SUT=float(np.mean(np.abs(ys - xs))), n_SUT=len(xs))
        xs95 = np.array(xs95); ys95 = np.array(ys95); ok = np.isfinite(xs95) & np.isfinite(ys95)
        b95, a95 = np.polyfit(xs95[ok], ys95[ok], 1)
        rec.update(slope95_SUT=float(b95), icpt95_SUT=float(a95), rho95_SUT=float(spearmanr(xs95[ok], ys95[ok]).correlation))
    else:
        rec.update(slope_SUT=np.nan, icpt_SUT=np.nan, rho_SUT=np.nan, mae_SUT=np.nan, n_SUT=len(xs), slope95_SUT=np.nan, icpt95_SUT=np.nan, rho95_SUT=np.nan)
    erows.append(rec)
    print(f"  [{tag} seed={seed} n={nseen:3d}] Dice ds400={rec['dice_dataset_400']:.4f} NHA={rec['dice_NHA12D']:.4f} C500={rec['dice_Crack500']:.4f} held={rec['heldout_mean']:.4f} "
          f"| r_NHA={rec['r_NHA12D']:.2f} | SUT slope={rec['slope_SUT']:.2f} a={rec['icpt_SUT']:.2f} ρ={rec['rho_SUT']:.2f} MAE={rec['mae_SUT']:.2f} | p95 slope={rec['slope95_SUT']:.2f} a={rec['icpt95_SUT']:.2f}", flush=True)


def run_variant(vname, seed, model, processor, base_sd, quick, sut_ref, sut_p95, sut_linear, erows, wrows):
    use_area, lr = SPEC[vname]
    E.seed_all(seed)
    model.load_state_dict(base_sd)
    for p in model.parameters(): p.requires_grad_(False)
    for n, p in model.named_parameters():
        if any(m in n for m in E.MODS): p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    train = E.train_pairs(); random.shuffle(train)
    cps = [0, 3, 6] if quick else CHECKPOINTS
    if quick: train = train[:6]
    tag = f"{vname}@lr{lr:g}"
    print(f"\n==== {tag} seed={seed} area_loss={use_area} 训练 {len(train)} 张，评测点 {cps}", flush=True)
    seen = 0; losses = []; t0 = time.time()
    if not any(r["tag"] == "zeroshot" for r in erows):   # 零样本只评一次
        eval_point(model, processor, "zeroshot", 0, seed, quick, sut_ref, sut_p95, sut_linear, erows, wrows)
    for ip, mp, thr in train:
        model.train()
        try:
            img = Image.open(ip).convert("RGB")
            gt = np.array(Image.open(mp).convert("L")) > thr
            img, gt = E.augment_base(img, gt)
            gt_t = torch.from_numpy(np.ascontiguousarray(gt)).float().to(DEV)
            out = model(**E.make_inputs(processor, img))
            pm = out["pred_masks"][0]; lg = out["pred_logits"][0]; sem = out["semantic_seg"][0, 0]
            loss = E.W_SEM * (E.dice_loss(sem, gt_t) + E.bce_mask(sem, gt_t))
            if use_area: loss = loss + E.W_AREA * E.area_loss(sem, gt_t)
            if pm.shape[0] > 0:
                with torch.no_grad():
                    gs = F.interpolate(gt_t[None, None], size=pm.shape[-2:], mode="nearest")[0, 0]
                    pmb = (pm > 0).float(); inter = (pmb * gs).sum((1, 2))
                    iou = inter / (pmb.sum((1, 2)) + gs.sum() - inter + 1e-6); match = int(iou.argmax())
                loss = loss + E.W_MASK * (E.dice_loss(pm[match], gt_t) + E.bce_mask(pm[match], gt_t))
                st = torch.zeros_like(lg); st[match] = 1.0
                loss = loss + E.W_SCORE * F.binary_cross_entropy_with_logits(lg.float(), st)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        except Exception:
            torch.cuda.empty_cache()
        seen += 1
        if seen in cps:
            print(f"  {tag} seed={seed} 已看 {seen} 张 loss(最近)={np.mean(losses[-60:]):.3f} {time.time()-t0:.0f}s", flush=True)
            eval_point(model, processor, tag, seen, seed, quick, sut_ref, sut_p95, sut_linear, erows, wrows)
    for p in model.parameters(): p.requires_grad_(False)
    torch.cuda.empty_cache()


def summarize(erows):
    print("\n" + "=" * 100 + "\n汇总：各变体在各评测点的 SUT 斜率 / held-out Dice（seed 均值 ± sd）\n" + "=" * 100)
    tags = []
    for r in erows:
        if r["tag"] not in tags: tags.append(r["tag"])
    print(f"  {'tag':14s}{'n_seen':>7s}  {'slope':>12s}  {'icpt':>12s}  {'rho':>12s}  {'MAE':>12s}  {'held':>14s}  {'ds400':>14s}  {'r_NHA':>12s}  seeds")
    for tag in tags:
        for n in sorted(set(r["n_seen"] for r in erows if r["tag"] == tag)):
            rr = [r for r in erows if r["tag"] == tag and r["n_seen"] == n]
            def ms(k): v = np.array([r[k] for r in rr], float); return f"{np.nanmean(v):.2f}±{np.nanstd(v):.2f}"
            def ms4(k): v = np.array([r[k] for r in rr], float); return f"{np.nanmean(v):.4f}±{np.nanstd(v):.4f}"
            print(f"  {tag:14s}{n:7d}  {ms('slope_SUT'):>12s}  {ms('icpt_SUT'):>12s}  {ms('rho_SUT'):>12s}  {ms('mae_SUT'):>12s}  {ms4('heldout_mean'):>14s}  {ms4('dice_dataset_400'):>14s}  {ms('r_NHA12D'):>12s}  {len(rr)}")
    print("\n判读: 斜率是否随看过的图像数单调下降？面积项 / 低 lr 是否把下降推迟？在 held-out 不低于零样本(0.578)的前提下，最佳斜率出现在哪个点？")


SEEDS0 = None


def main():
    global SEEDS0
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--variants", type=str, default="A,C,A_lo,C_lo")
    ap.add_argument("--seeds", type=str, default="1234,2025,7")
    args = ap.parse_args()
    sys.stdout = E.Tee(LOGP); T0 = time.time()
    seeds = [int(s) for s in args.seeds.split(",")]; SEEDS0 = seeds[0]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    print(f"e1b_subepoch.py  {ts}  quick={args.quick}  variants={variants}  seeds={seeds}  {torch.cuda.get_device_name(0)}\nlog -> {LOGP}\n")

    processor = Sam3Processor.from_pretrained(E.W)
    model = Sam3Model.from_pretrained(E.W).to(DEV).eval()
    for p in model.parameters(): p.requires_grad_(False)
    base_sd = {k: v.clone() for k, v in model.state_dict().items()}
    sut_ref, sut_linear = E.load_sut_ref()
    sut_p95 = {}
    with open(E.CLOSELOOP, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["variant"] == "zeroshot" and r["cond"] == "text_full": sut_p95[str(r["stem"])] = float(r["gt_w_p95"])
    print(f"SUT 参考 {len(sut_ref)} 张, 线状 {len(sut_linear)} 张")

    erows = []; wrows = []
    for seed in seeds:
        for v in variants:
            try: run_variant(v, seed, model, processor, base_sd, args.quick, sut_ref, sut_p95, sut_linear, erows, wrows)
            except Exception: traceback.print_exc()
    if erows:
        keys = sorted(set(k for r in erows for k in r), key=lambda k: (k not in ("tag", "variant", "seed", "n_seen"), k))
        with open(CSV_EVAL, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(erows)
    if wrows:
        with open(CSV_W, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(wrows[0].keys())); w.writeheader(); w.writerows(wrows)
    try: summarize(erows)
    except Exception: traceback.print_exc()
    print(f"\n评测CSV {CSV_EVAL}\n逐图宽度CSV {CSV_W}\n总耗时 {(time.time()-T0)/60:.1f} min")


if __name__ == "__main__":
    main()