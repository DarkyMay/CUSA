#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dump_masks.py —— 把 SUT-Crack 上 SAM3 各变体的预测掩膜落盘，供返修实验复用。

复用 closeloop_sutcrack_v2.py 的 seg()/predict() 逻辑（semantic 头、logit 0、
bilinear 回原图），保证掩膜与论文数字一致。

输出:
  results/masks/<variant>/<stem>.png        二值掩膜 (0/255)，原图尺寸
  results/masks/zs_full_logit/<stem>.npy    ZS text_full 的原始 288x288 logit (float16)
  results/masks/manifest.csv                每图每变体的 dice / 前景像素数，用来和 closeloop CSV 对账

变体: zs_full  zs_crop  zs_box  ft_full   （与论文 Table widthscale 前四行对应）

用法:  python3 dump_masks.py            全量
       python3 dump_masks.py --limit 5  冒烟
"""
import os, sys, glob, re, csv, time, argparse
import xml.etree.ElementTree as ET
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image, ImageOps
from transformers import Sam3Model, Sam3Processor

Image.MAX_IMAGE_PIXELS = None
OUT = "/root/autodl-tmp/proj_bridge"
W = "/root/autodl-tmp/sam3_weights"
CK_V3 = os.path.join(OUT, "ckpts", "sam3_ft_v3_textonly_best.pt")
ROOT = "/root/autodl-tmp/SUT-Crack"
XMLD = os.path.join(ROOT, "2-Object Detection")
IMGD = os.path.join(ROOT, "1-Segmentation", "Original Image")
GTD = os.path.join(ROOT, "1-Segmentation", "Ground Truth")
MASKD = os.path.join(OUT, "results", "masks")
DEV = "cuda"; TEXT = "crack"

# 目录名可能已改成下划线，两种都试
for cand in [IMGD, IMGD.replace(" ", "_")]:
    if os.path.isdir(cand): IMGD = cand; break
for cand in [GTD, GTD.replace(" ", "_")]:
    if os.path.isdir(cand): GTD = cand; break
for cand in [XMLD, XMLD.replace(" ", "_")]:
    if os.path.isdir(cand): XMLD = cand; break


def numkey(p):
    m = re.findall(r'\d+', os.path.basename(p)); return int(m[0]) if m else 0


def parse_xml(p):
    out = []
    for o in ET.parse(p).getroot().findall('object'):
        b = o.find('bndbox')
        out.append([float(b.findtext('xmin')), float(b.findtext('ymin')),
                    float(b.findtext('xmax')), float(b.findtext('ymax'))])
    return out


def find_file(d, stem, exts):
    for e in exts:
        for s in [stem, stem.replace(" ", "_"), stem.replace("_", " ")]:
            p = os.path.join(d, s + e)
            if os.path.exists(p): return p
    return None


def dice_np(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


@torch.no_grad()
def seg_logit(model, processor, img, box_px=None):
    """返回 (原图尺寸二值掩膜, 原始 logit 图 [R,R])"""
    if box_px is None:
        inp = processor(images=img, text=TEXT, return_tensors="pt")
    else:
        inp = processor(images=img, text=TEXT, input_boxes=[[list(box_px)]],
                        input_boxes_labels=[[1]], return_tensors="pt")
    inp = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}
    with torch.autocast(DEV, dtype=torch.bfloat16):
        o = model(**inp)
    raw = o["semantic_seg"][0, 0].float()
    Wd, Hd = img.size
    m = F.interpolate(raw[None, None], size=(Hd, Wd), mode="bilinear")[0, 0].cpu().numpy() > 0
    return m, raw.cpu().numpy()


def predict(model, processor, img, boxes, cond):
    Wd, Hd = img.size
    if cond == "text_full":
        return seg_logit(model, processor, img)
    if cond == "box_prompt":
        m = np.zeros((Hd, Wd), bool)
        for b in boxes:
            m |= seg_logit(model, processor, img, box_px=b)[0]
        return m, None
    m = np.zeros((Hd, Wd), bool)
    for b in boxes:
        bw, bh = b[2] - b[0], b[3] - b[1]
        x0 = int(max(0, b[0] - 0.1 * bw)); y0 = int(max(0, b[1] - 0.1 * bh))
        x1 = int(min(Wd, b[2] + 0.1 * bw)); y1 = int(min(Hd, b[3] + 0.1 * bh))
        if x1 - x0 < 32 or y1 - y0 < 32: continue
        m[y0:y1, x0:x1] |= seg_logit(model, processor, img.crop((x0, y0, x1, y1)))[0]
    return m, None


def load_ckpt(model, path, base_sd):
    model.load_state_dict(base_sd)
    sd = torch.load(path, map_location="cpu")
    for k in ["model", "state_dict", "model_state_dict"]:
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict): sd = sd[k]; break
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    r = model.load_state_dict(sd, strict=False)
    print(f"  载入 {os.path.basename(path)}: 覆盖 {len(sd)} 项, missing {len(r.missing_keys)}, unexpected {len(r.unexpected_keys)}")
    if len(sd) < 50: print("  !! 覆盖项很少，检查 ckpt 格式");


def save_mask(d, stem, m):
    os.makedirs(d, exist_ok=True)
    Image.fromarray((m.astype(np.uint8) * 255)).save(os.path.join(d, stem + ".png"), optimize=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip_ft", action="store_true", help="没有 v3 ckpt 时跳过 ft_full")
    args = ap.parse_args()
    t0 = time.time()
    print(f"IMGD={IMGD}\nGTD={GTD}\nXMLD={XMLD}\n落盘到 {MASKD}")

    processor = Sam3Processor.from_pretrained(W)
    model = Sam3Model.from_pretrained(W).to(DEV).eval()
    torch.set_grad_enabled(False)
    base_sd = {k: v.clone() for k, v in model.state_dict().items()}

    xmls = sorted(glob.glob(os.path.join(XMLD, "*.xml")), key=numkey)
    if args.limit: xmls = xmls[:args.limit]
    print(f"图像 {len(xmls)} 张")

    # 变体: (名称, 需要的权重, cond)
    plan = [("zs_full", "zs", "text_full"), ("zs_crop", "zs", "roi_crop"),
            ("zs_box", "zs", "box_prompt"), ("ft_full", "ft", "text_full")]
    if args.skip_ft or not os.path.exists(CK_V3):
        print(f"!! 跳过 ft_full（ckpt {'不存在' if not os.path.exists(CK_V3) else '按参数跳过'}）")
        plan = [p for p in plan if p[1] == "zs"]

    rows = []
    cur = None
    for vname, wt, cond in plan:
        if wt != cur:
            if wt == "zs": model.load_state_dict(base_sd)
            else: load_ckpt(model, CK_V3, base_sd)
            cur = wt
        print(f"\n[{vname}] cond={cond} weights={wt}", flush=True)
        for i, xp in enumerate(xmls):
            stem = os.path.splitext(os.path.basename(xp))[0]
            ip = find_file(IMGD, stem, [".jpg", ".JPG", ".jpeg", ".png"])
            gp = find_file(GTD, stem, [".png", ".jpg"])
            if ip is None: print(f"  缺图 {stem}"); continue
            img = ImageOps.exif_transpose(Image.open(ip)).convert("RGB")   # 与 closeloop 一致：先按 EXIF 转正
            boxes = parse_xml(xp)
            m, raw = predict(model, processor, img, boxes, cond)
            save_mask(os.path.join(MASKD, vname), stem.replace(" ", "_"), m)
            if vname == "zs_full" and raw is not None:
                ld = os.path.join(MASKD, "zs_full_logit"); os.makedirs(ld, exist_ok=True)
                np.save(os.path.join(ld, stem.replace(" ", "_") + ".npy"), raw.astype(np.float16))
            d = np.nan
            shape_ok = ""
            if gp is not None:
                gt = np.array(Image.open(gp).convert("L")) > 127
                if gt.shape == m.shape: d = dice_np(m, gt); shape_ok = "ok"
                else: shape_ok = f"MISMATCH gt{gt.shape} pred{m.shape}"
            rows.append({"variant": vname, "stem": stem, "H": m.shape[0], "W": m.shape[1],
                         "fg_px": int(m.sum()), "dice": round(float(d), 4), "shape": shape_ok,
                         "logit_shape": "x".join(map(str, raw.shape)) if raw is not None else ""})
            if i % 20 == 0:
                print(f"  {i:4d} {stem:12s} fg={int(m.sum()):8d} dice={d:.3f}  {time.time()-t0:.0f}s", flush=True)
        torch.cuda.empty_cache()

    mp = os.path.join(MASKD, "manifest.csv")
    with open(mp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tp = os.path.join(OUT, "results", f"dump_masks_{ts}.txt")
    lines = [f"dump_masks  {ts}", f"IMGD={IMGD}", f"GTD={GTD}", f"XMLD={XMLD}", f"masks -> {MASKD}",
             f"manifest -> {mp}", f"images={len(xmls)}  ft_ckpt={CK_V3 if any(p[1]=='ft' for p in plan) else 'skipped'}", ""]
    lines.append("variant   n_ok  n_mismatch  mean_Dice   paper")
    paper = {"zs_full": 0.759, "zs_crop": 0.751, "zs_box": 0.744, "ft_full": 0.614}
    for vname, _, _ in plan:
        rr = [r for r in rows if r["variant"] == vname]
        ds = [r["dice"] for r in rr if not np.isnan(r["dice"])]
        nm = sum(1 for r in rr if r["shape"].startswith("MISMATCH"))
        lines.append(f"{vname:8s}  {len(ds):4d}  {nm:10d}  {np.mean(ds) if ds else float('nan'):9.3f}   {paper[vname]:.3f}")
    mm = [r for r in rows if r["shape"].startswith("MISMATCH")]
    if mm:
        lines += ["", "shape mismatches (first 10):"] + [f"  {r['variant']} {r['stem']} {r['shape']}" for r in mm[:10]]
    lines += ["", f"elapsed {time.time()-t0:.0f}s"]
    with open(tp, "w") as f: f.write("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"\n结果 -> {tp}")


if __name__ == "__main__":
    main()