#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d_prompt_sweep.py —— R3-3：文本提示词的系统扫描 + 提示词集成（SAM3 零样本，语义头，logit 0）

15 个提示词（先验列出，不按结果挑选）在 SUT-Crack 130 / NHA12D 80 / Crack500 100 上：
  Dice（各集）、SUT 126 线状的宽度偏差 / ρ / 斜率（中值 EDT 宽，与论文同口径）
集成（logit 平均后 >0）：
  ens_core  = {crack, cracks, pavement crack, road crack}      —— 论文正文提到的同义近邻
  ens_all   = 全部 15 个
  ens_vote  = 全部 15 个的多数票
输出 results/d_prompt_<ts>.txt + d_prompt_<ts>.csv（逐提示词×数据集）+ d_prompt_sut_<ts>.csv（逐图宽度）
用法 python3 d_prompt_sweep.py [--limit N]
"""
import os, sys, csv, glob, re, time, argparse, datetime, traceback
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage.morphology import skeletonize
from transformers import Sam3Model, Sam3Processor

sys.path.insert(0, "/root/autodl-tmp/proj_bridge")
import e1_collapse_mitigation as E

Image.MAX_IMAGE_PIXELS = None
OUT = E.OUT; W = E.W; MMPX = E.MMPX
PROMPTS = ["crack", "cracks", "pavement crack", "road crack", "asphalt crack", "concrete crack", "surface crack",
           "crack in the road", "linear crack", "thin crack", "fracture", "fissure", "pavement distress", "damage", "crack line"]
CORE = ["crack", "cracks", "pavement crack", "road crack"]
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"d_prompt_{ts}.txt")
CSV_M = os.path.join(OUT, "results", f"d_prompt_{ts}.csv")
CSV_S = os.path.join(OUT, "results", f"d_prompt_sut_{ts}.csv")


@torch.no_grad()
def sem_logit(model, processor, img, text):
    inp = processor(images=img, text=text, return_tensors="pt")
    inp = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = model(**inp)
    return o["semantic_seg"][0, 0].float()


def to_mask(logit, size_hw):
    return F.interpolate(logit[None, None], size=size_hw, mode="bilinear")[0, 0].cpu().numpy() > 0


def width_med_mm(mask):
    r = E.width_profile_px(mask)
    return 0.0 if r is None else r[0] * MMPX


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); args = ap.parse_args()
    sys.stdout = E.Tee(LOGP); T0 = time.time()
    print(f"d_prompt_sweep.py  {ts}  prompts={len(PROMPTS)}  {torch.cuda.get_device_name(0)}\nlog -> {LOGP}\n")
    processor = Sam3Processor.from_pretrained(W); model = Sam3Model.from_pretrained(W).cuda().eval()
    torch.set_grad_enabled(False)
    sut_ref, sut_linear = E.load_sut_ref()
    sets = {"SUT": E.eval_pairs("SUT"), "NHA12D": E.eval_pairs("NHA12D"), "Crack500": E.eval_pairs("Crack500")}
    if args.limit: sets = {k: v[:args.limit] for k, v in sets.items()}
    names = PROMPTS + ["ens_core", "ens_all", "ens_vote"]
    dice = {n: {k: [] for k in sets} for n in names}
    srows = []
    for dsname, pairs in sets.items():
        print(f"\n[{dsname}] {len(pairs)} 张", flush=True)
        for i, (ip, mp, thr) in enumerate(pairs):
            stem = os.path.splitext(os.path.basename(ip))[0]
            try:
                img = E.open_img(ip) if dsname == "SUT" else Image.open(ip).convert("RGB")
                gt = np.array(Image.open(mp).convert("L")) > thr
                logits = {p: sem_logit(model, processor, img, p) for p in PROMPTS}
                masks = {p: to_mask(l, gt.shape) for p, l in logits.items()}
                masks["ens_core"] = to_mask(torch.stack([logits[p] for p in CORE]).mean(0), gt.shape)
                masks["ens_all"] = to_mask(torch.stack([logits[p] for p in PROMPTS]).mean(0), gt.shape)
                votes = np.stack([masks[p] for p in PROMPTS]).sum(0)
                masks["ens_vote"] = votes > len(PROMPTS) / 2
                for n in names: dice[n][dsname].append(E.dice_np(masks[n], gt))
                if dsname == "SUT" and stem in sut_linear:
                    row = dict(stem=stem, gt_med=sut_ref[stem])
                    for n in names: row[n] = width_med_mm(masks[n])
                    srows.append(row)
            except Exception:
                torch.cuda.empty_cache(); traceback.print_exc()
            if i % 25 == 0: print(f"  {i+1}/{len(pairs)} {time.time()-T0:.0f}s", flush=True)

    # 汇总
    print("\n" + "=" * 100 + "\n提示词 × 数据集：Dice；SUT 126 线状：中值宽偏差% / ρ / 斜率 / 截距 / MAE\n" + "=" * 100)
    print(f"  {'prompt':20s}{'SUT':>7s}{'NHA12D':>8s}{'C500':>7s}{'held':>7s} | {'bias%':>7s}{'rho':>7s}{'slope':>7s}{'icpt':>7s}{'MAE':>6s}{'n_empty':>8s}")
    mrows = []
    gtv = np.array([r["gt_med"] for r in srows])
    for n in names:
        d = {k: float(np.mean(v)) if v else np.nan for k, v in dice[n].items()}
        pv = np.array([r[n] for r in srows]); ok = pv > 0
        if ok.sum() >= 5:
            b, a = np.polyfit(gtv[ok], pv[ok], 1); rho = spearmanr(gtv[ok], pv[ok]).correlation
            bias = 100 * (pv[ok].mean() / gtv[ok].mean() - 1); mae = np.mean(np.abs(pv[ok] - gtv[ok]))
        else: b = a = rho = bias = mae = np.nan
        held = (d["NHA12D"] + d["Crack500"]) / 2
        mrows.append(dict(prompt=n, dice_SUT=d["SUT"], dice_NHA12D=d["NHA12D"], dice_Crack500=d["Crack500"], heldout=held,
                          bias_pct=bias, rho=rho, slope=b, icpt=a, mae=mae, n_empty=int((~ok).sum())))
        print(f"  {n:20s}{d['SUT']:7.3f}{d['NHA12D']:8.3f}{d['Crack500']:7.3f}{held:7.3f} | {bias:+7.1f}{rho:7.3f}{b:7.2f}{a:7.2f}{mae:6.2f}{int((~ok).sum()):8d}")
    base = [r for r in mrows if r["prompt"] == "crack"][0]
    better = [r["prompt"] for r in mrows if r["prompt"] in PROMPTS and r["heldout"] > base["heldout"]]
    print(f"\n  论文所用 'crack' 的 held-out = {base['heldout']:.3f}；15 个提示词中 held-out 更高的: {better if better else '无'}")
    print(f"  提示词间 held-out 极差 = {max(r['heldout'] for r in mrows if r['prompt'] in PROMPTS) - min(r['heldout'] for r in mrows if r['prompt'] in PROMPTS):.3f}；"
          f"SUT 宽度偏差极差 = {max(r['bias_pct'] for r in mrows if r['prompt'] in PROMPTS) - min(r['bias_pct'] for r in mrows if r['prompt'] in PROMPTS):.1f} 个百分点")
    print("  判读：若多数同义提示词的宽度偏差都为 +30% 以上且 ρ 相近，说明偏差是输出粒度而非提示词选择造成；集成若不改变偏差，则提示词优化不是缓解路径。")
    with open(CSV_M, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(mrows[0].keys())); w.writeheader(); w.writerows(mrows)
    with open(CSV_S, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(srows[0].keys())); w.writeheader(); w.writerows(srows)
    print(f"\nCSV {CSV_M}\n逐图 {CSV_S}\n总耗时 {(time.time()-T0)/60:.1f} min")


if __name__ == "__main__":
    main()