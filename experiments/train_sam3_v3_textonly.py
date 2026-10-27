#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAM3 多域微调 v3 —— 纯文本协议（训练与评测协议终于一致）

【为什么必须重训】
  v2 及此前所有训练，每张图都传了 input_boxes=[[归一化cxcywh]]。
  probe_box_effect 证实：processor 期望像素 xyxy，归一化值产生 x1<x0 的非法框
  并被直接丢弃。实测 legacy(0.5721) ≈ textonly(0.5775) ≫ tinybox(0.5265)，
  证明该框既非小框也非整图框，而是【被忽略】。
  于是训练协议无法写进论文（"我们传了一个非法框"不是方法描述），
  且 ep10 在 textonly 下 NHA12D 从 0.5195 掉到 0.4651 —— 训练时有box token、
  测试时没有，属协议不匹配。

【相对 v2 的改动】
  1. ★训练与评测一律纯文本，不传任何 input_boxes。方法可描述为
     "文本提示 'crack'，无任何视觉 prompt"。
  2. 无框时模型可能对某些图返回 0 个实例，此时跳过 instance/score 两项损失，
     只用 semantic 损失，不再让脚本崩掉或静默跳图。
  3. 评测点加密到 ep1,2,3,5,10,15,20,25 —— 用于检验"轻微微调能否在适配域内的
     同时保住预训练先验"（v2 观察到 NHA12D 在最早的评测点最高，疑似灾难性遗忘）。
  4. 删除 RDD XML 与 GT 派生框相关代码（已无用，减少出错面）。

【保持不变】同数据620张、同增强、25 epoch、LR 1e-4、三模块可训练(15.06M)、
  checkpoint 选择仅用 dataset_400、NHA12D 全部图、semantic 单头。

【预期与目的】零训练独立跨域已达 0.5775，而 v2 微调最好只有 0.5756。
  本次重训的目的不是刷高，而是让"微调无助于跨域"这个结论在
  完全一致的协议下成立、可发表。若结果仍是无助，那就如实写。

预计: 训练 25×~140s ≈ 58min + 8次评测×~35s ≈ 5min，合计约 1-1.2h
"""
import sys, os, datetime, random, time, glob, csv
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image, ImageEnhance
from transformers import Sam3Model, Sam3Processor

OUT = "/root/autodl-tmp/proj_bridge"
for d in ["ckpts", "ckpts/v3", "logs", "results"]:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()
    def close(s): s.f.close()


tee = Tee(os.path.join(OUT, "logs", f"train_sam3_v3_textonly_{ts}.log")); sys.stdout = tee

W = "/root/autodl-tmp/sam3_weights"
DS400 = "/root/autodl-tmp/dataset_400_clean"
DEEPCRACK = "/root/autodl-tmp/DeepCrack-master/dataset"
NHA = "/root/autodl-tmp/NHA12D/NHA12D_dataset"
SUBS = ["Asphalt_pavement_FD", "Asphalt_pavement_VD", "Concrete_pavement_FD", "Concrete_pavement_VD"]
DEV = "cuda"; EPOCHS = 25; LR = 1e-4; W_MASK = 1.0; W_SCORE = 2.0; W_SEM = 1.0
TEXT = "crack"; SEED = 1234
EVAL_EPS = [1, 2, 3, 5, 10, 15, 20, 25]
BEST = os.path.join(OUT, "ckpts", "sam3_ft_v3_textonly_best.pt")
CURVE = os.path.join(OUT, "results", f"train_v3_curve_{ts}.csv")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
TESTSETS = ["dataset_400", "DeepCrack", "NHA12D", "Crack500"]


def dice_np(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def build_train():
    items = []
    idir = os.path.join(DS400, "images/train"); mdir = os.path.join(DS400, "masks/train")
    for f in sorted(os.listdir(idir)):
        if f.endswith(".jpg"):
            items.append((os.path.join(idir, f),
                          os.path.join(mdir, os.path.splitext(f)[0] + ".png")))
    for img in sorted(glob.glob(os.path.join(DEEPCRACK, "train_img", "*.jpg"))):
        mp = os.path.join(DEEPCRACK, "train_lab",
                          os.path.splitext(os.path.basename(img))[0] + ".png")
        if os.path.exists(mp): items.append((img, mp))
    return items


def augment(img, gt):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT); gt = gt[:, ::-1].copy()
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    return img, gt


def make_inputs(processor, img):
    """★改动1：纯文本，不传任何 input_boxes"""
    inp = processor(images=img, text=TEXT, return_tensors="pt")
    return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}


def dice_loss(logit, target):
    p = torch.sigmoid(logit.float())
    if p.shape != target.shape:
        p = F.interpolate(p[None, None], size=target.shape, mode="bilinear", align_corners=False)[0, 0]
    inter = (p * target).sum(); s = p.sum() + target.sum()
    return 1 - (2 * inter + 1) / (s + 1)


def bce_mask(logit, target):
    if logit.shape != target.shape:
        logit = F.interpolate(logit[None, None].float(), size=target.shape,
                              mode="bilinear", align_corners=False)[0, 0]
    return F.binary_cross_entropy_with_logits(logit.float(), target)


def eval_pairs(name):
    if name == "dataset_400":
        d = DS400
        return [(os.path.join(d, "images/test", f),
                 os.path.join(d, "masks/test", os.path.splitext(f)[0] + ".png"), 127)
                for f in sorted(os.listdir(os.path.join(d, "images/test"))) if f.endswith(".jpg")]
    if name == "DeepCrack":
        out = []
        for img in sorted(glob.glob(os.path.join(DEEPCRACK, "test_img", "*.jpg")))[:60]:
            m = os.path.join(DEEPCRACK, "test_lab", os.path.splitext(os.path.basename(img))[0] + ".png")
            if os.path.exists(m): out.append((img, m, 127))
        return out
    if name == "NHA12D":
        out = []
        for sub in SUBS:
            rd = os.path.join(NHA, sub, "Raw_images"); md = os.path.join(NHA, sub, "Masks")
            if not os.path.isdir(rd): continue
            for img in sorted(glob.glob(os.path.join(rd, "*.jpg"))):
                m = os.path.join(md, os.path.splitext(os.path.basename(img))[0] + ".png")
                if os.path.exists(m): out.append((img, m, 0))
        return out
    if name == "Crack500":
        out = []
        with open("/root/autodl-tmp/CRACK500/test.txt") as f:
            for line in f:
                p = line.split()
                if len(p) >= 2:
                    ip = os.path.join("/root/autodl-tmp/CRACK500", p[0])
                    mp = os.path.join("/root/autodl-tmp/CRACK500", p[1])
                    if os.path.exists(ip) and os.path.exists(mp): out.append((ip, mp, 127))
        return out[:100]


@torch.no_grad()
def evaluate(model, processor, dsname):
    model.eval(); acc = {"semantic": [], "instance": []}; nfail = 0
    for ip, mp, thr in eval_pairs(dsname):
        img = Image.open(ip).convert("RGB")
        gt = np.array(Image.open(mp).convert("L")) > thr
        try:
            out = model(**make_inputs(processor, img))
            sm = F.interpolate(out["semantic_seg"][0, 0][None, None].float(),
                               size=gt.shape, mode="bilinear")[0, 0].cpu().numpy() > 0
            acc["semantic"].append(dice_np(sm, gt))
            pm = out["pred_masks"][0].float()
            if pm.shape[0] > 0:
                idx = int(out["pred_logits"][0].float().argmax())
                mi = F.interpolate(pm[idx][None, None], size=gt.shape,
                                   mode="bilinear")[0, 0].cpu().numpy() > 0
                acc["instance"].append(dice_np(mi, gt))
            else:
                acc["instance"].append(0.0)
        except Exception:
            torch.cuda.empty_cache(); nfail += 1
            for k in acc: acc[k].append(0.0)
    return {k: float(np.mean(v)) for k, v in acc.items()}, nfail


# ==== 主流程 ====
print("=" * 96)
print("SAM3 多域微调 v3 —— 纯文本协议（训练与评测一致）")
print(f"seed={SEED}  epochs={EPOCHS}  评测点={EVAL_EPS}  (基线为40ep，预算对基线更宽松)")
print("参照: 零训练纯文本 独立跨域 0.5775 | v2微调最好 0.5756 | 最佳基线均 0.5165")
print("=" * 96)

model = Sam3Model.from_pretrained(W).to(DEV)
processor = Sam3Processor.from_pretrained(W)
mods = ["mask_decoder", "detr_decoder", "dot_product_scoring"]
for p in model.parameters(): p.requires_grad_(False)
for n, p in model.named_parameters():
    if any(m in n for m in mods): p.requires_grad_(True)
print(f"可训练 {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M "
      f"/ 总 {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)

train = build_train()
print(f"训练集 {len(train)} 张")
for t in TESTSETS: print(f"  测试集 {t}: {len(eval_pairs(t))} 张")

rows = []
print("\nep00 (零训练):")
for t in TESTSETS:
    s, nf = evaluate(model, processor, t)
    print(f"  {t:<14} semantic={s['semantic']:.4f} instance={s['instance']:.4f} 失败={nf}")
    rows.append(dict(ep=0, ds=t, **s, nfail=nf))

best = -1; best_ep = -1; noinst_total = 0; nskip_total = 0
for ep in range(1, EPOCHS + 1):
    model.train(); random.shuffle(train); t0 = time.time()
    losses = []; nskip = 0; noinst = 0
    for ip, mp in train:
        img = Image.open(ip).convert("RGB")
        gt = np.array(Image.open(mp).convert("L")) > 127
        img, gt = augment(img, gt)
        gt_t = torch.from_numpy(gt).float().to(DEV)
        try:
            out = model(**make_inputs(processor, img))
            pm = out["pred_masks"][0]; lg = out["pred_logits"][0]; sem = out["semantic_seg"][0, 0]
            loss = W_SEM * (dice_loss(sem, gt_t) + bce_mask(sem, gt_t))
            # ★改动2：无框时可能没有实例，此时只用 semantic 损失
            if pm.shape[0] > 0:
                with torch.no_grad():
                    gs = F.interpolate(gt_t[None, None], size=pm.shape[-2:], mode="nearest")[0, 0]
                    pmb = (pm > 0).float(); inter = (pmb * gs).sum((1, 2))
                    iou = inter / (pmb.sum((1, 2)) + gs.sum() - inter + 1e-6)
                    match = int(iou.argmax())
                loss = loss + W_MASK * (dice_loss(pm[match], gt_t) + bce_mask(pm[match], gt_t))
                st = torch.zeros_like(lg); st[match] = 1.0
                loss = loss + W_SCORE * F.binary_cross_entropy_with_logits(lg.float(), st)
            else:
                noinst += 1
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        except Exception:
            torch.cuda.empty_cache(); nskip += 1; continue
    noinst_total += noinst
    nskip_total += nskip

    if ep in EVAL_EPS:
        sc = {}
        for t in TESTSETS:
            s, nf = evaluate(model, processor, t)
            sc[t] = s; rows.append(dict(ep=ep, ds=t, **s, nfail=nf))
        indep = (sc["NHA12D"]["semantic"] + sc["Crack500"]["semantic"]) / 2
        print(f"ep{ep:02d} loss={np.mean(losses):.3f} skip={nskip} 无实例={noinst} | "
              f"ds400={sc['dataset_400']['semantic']:.4f} DeepCrack={sc['DeepCrack']['semantic']:.4f} "
              f"NHA12D={sc['NHA12D']['semantic']:.4f} Crack500={sc['Crack500']['semantic']:.4f} "
              f"| 独立均={indep:.4f}(仅观测) {time.time()-t0:.0f}s")
        snap = {n: model.state_dict()[n] for n in model.state_dict() if any(m in n for m in mods)}
        torch.save(snap, os.path.join(OUT, "ckpts", "v3", f"sam3_v3_ep{ep}.pt"))
        key = sc["dataset_400"]["semantic"]          # 选择仅用域内，与基线对齐
        if key > best:
            best = key; best_ep = ep
            torch.save(snap, BEST)
            print(f"     >> 存best (ep{ep}, ds400={key:.4f})")
    else:
        print(f"ep{ep:02d} loss={np.mean(losses):.3f} skip={nskip} 无实例={noinst} {time.time()-t0:.0f}s")

with open(CURVE, "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

print("\n" + "=" * 96)
print(f"完成。best = ep{best_ep} (dataset_400 semantic={best:.4f})")
print(f"权重 {BEST}   逐epoch快照 {OUT}/ckpts/v3/")
print(f"训练期异常跳过累计 {nskip_total}，无实例样本累计 {noinst_total}")
print(f"曲线CSV {CURVE}")
print("\n⚠ NHA12D / Crack500 全程未参与选择，为真正 held-out。")
print("⚠ 判读: 若各epoch独立跨域均 < 0.5775(零训练)，则'微调无助于跨域'在一致协议下成立，")
print("  这是本项目要如实报告的结论，不是失败。")
sys.stdout = tee.o; tee.close()