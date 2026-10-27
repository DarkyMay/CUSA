#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_sam3_lora.py —— LoRA 微调对照（防守性实验，非寻找方法）

【为什么必须做】现有"微调无助于跨域"的结论基于【直接放开 mask_decoder/detr_decoder/
  dot_product_scoring 三个完整模块(15.06M)】的配方。CrackSAM 用 adapter/LoRA 得到
  相反结论。不做 LoRA 对照就无法排除"是我们的配方差"，会被直接反驳。

【为什么手写而非用 peft】
  探针显示 peft 的后缀匹配会把 LoRA 注入全模型：q_proj 全模型 93 层/61.21M，
  而目标模块内仅 19 层/1.25M；fc1 全模型 71 层/264.00M，目标模块内仅 6 层/3.16M。
  那样 LoRA 可触及 454M 视觉编码器与 353M 文本编码器，与"只动解码器"的对照不对等。
  手写可精确锁定到三个目标模块内部。

【与 v3 的唯一差异】
  · 可训练参数：LoRA 低秩增量（rank 16，约 0.92M）替代直接放开三模块（15.06M）
  · 学习率：5e-4（LoRA 常规范围），而非 v3 的 1e-4
    —— 有意给 LoRA 适配其自身的超参数。本实验的目的是让对立假设尽可能赢；
       若它仍未赢，"微调无助于跨域"才站得住。论文须如实写明此处未做等超参对照。
【完全相同】数据620张、增强、25 epoch、纯文本协议、评测点、
  checkpoint 仅按 dataset_400 选、NHA12D 全部图、semantic 单头 τ=0

用法: python3 train_sam3_lora.py [--rank 16] [--lr 5e-4]
预计: 约 1-1.3h
"""
import sys, os, math, datetime, random, time, glob, csv, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image, ImageEnhance
from transformers import Sam3Model, Sam3Processor

OUT = "/root/autodl-tmp/proj_bridge"
for d in ["ckpts", "ckpts/lora", "logs", "results"]:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()
    def close(s): s.f.close()


W = "/root/autodl-tmp/sam3_weights"
DS400 = "/root/autodl-tmp/dataset_400_clean"
DEEPCRACK = "/root/autodl-tmp/DeepCrack-master/dataset"
NHA = "/root/autodl-tmp/NHA12D/NHA12D_dataset"
SUBS = ["Asphalt_pavement_FD", "Asphalt_pavement_VD", "Concrete_pavement_FD", "Concrete_pavement_VD"]
DEV = "cuda"; EPOCHS = 25; TEXT = "crack"; SEED = 1234
W_MASK, W_SCORE, W_SEM = 1.0, 2.0, 1.0
EVAL_EPS = [1, 2, 3, 5, 10, 15, 20, 25]
TARGET_PREFIXES = ["mask_decoder", "detr_decoder", "dot_product_scoring"]
TARGET_SUFFIXES = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
TESTSETS = ["dataset_400", "DeepCrack", "NHA12D", "Crack500"]


# ---------------- LoRA ----------------
class LoRALinear(nn.Module):
    """y = W x + (alpha/r) * B A x ；W 冻结，只训 A/B"""
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # 零初始化 -> 初始等价于原模型
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.base(x)
        lora = self.drop(x) @ self.A.transpose(0, 1) @ self.B.transpose(0, 1)
        return out + lora * self.scaling


def inject_lora(model, r, alpha, dropout):
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not any(name.startswith(p + ".") for p in TARGET_PREFIXES):
            continue
        if name.split(".")[-1] in TARGET_SUFFIXES:
            targets.append(name)
    for name in targets:
        parent_name, leaf = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, leaf, LoRALinear(getattr(parent, leaf), r, alpha, dropout))
    return targets


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
    model.eval(); acc = []
    for ip, mp, thr in eval_pairs(dsname):
        img = Image.open(ip).convert("RGB")
        gt = np.array(Image.open(mp).convert("L")) > thr
        try:
            out = model(**make_inputs(processor, img))
            sm = F.interpolate(out["semantic_seg"][0, 0][None, None].float(),
                               size=gt.shape, mode="bilinear")[0, 0].cpu().numpy() > 0
            acc.append(dice_np(sm, gt))
        except Exception:
            torch.cuda.empty_cache(); acc.append(0.0)
    return float(np.mean(acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--dropout", type=float, default=0.05)
    args = ap.parse_args()

    tee = Tee(os.path.join(OUT, "logs", f"train_sam3_lora_r{args.rank}_{ts}.log"))
    sys.stdout = tee
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    print("=" * 96)
    print(f"SAM3 LoRA 微调对照 | rank={args.rank} alpha={args.alpha} lr={args.lr} dropout={args.dropout}")
    print(f"注入范围: {TARGET_PREFIXES} 内的 {TARGET_SUFFIXES}")
    print("参照: 零训练纯文本 独立跨域 0.5779 | v3全模块微调最好 0.5818 | 最佳基线均 0.5165")
    print("=" * 96)

    model = Sam3Model.from_pretrained(W).to(DEV)
    processor = Sam3Processor.from_pretrained(W)
    tot_before = sum(p.numel() for p in model.parameters())
    targets = inject_lora(model, args.rank, args.alpha, args.dropout)
    model = model.to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    for n, p in model.named_parameters():
        if n.endswith(".A") or n.endswith(".B"):
            p.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n注入 LoRA 层数 {len(targets)}（示例: {targets[:3]}）")
    print(f"可训练参数 {trainable/1e6:.3f}M  = 全模块配方 15.06M 的 {100*trainable/15.06e6:.1f}%")
    print(f"总参数 {tot_before/1e6:.1f}M -> {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    if not (0.3e6 < trainable < 4e6):
        print("!! 可训练参数不在 0.3~4M 窗口内，对照可能不成立，请调整 rank")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    train = build_train()
    print(f"训练集 {len(train)} 张")
    for t in TESTSETS: print(f"  测试集 {t}: {len(eval_pairs(t))} 张")

    rows = []
    print("\nep00 (LoRA 零初始化，应与零训练一致):")
    for t in TESTSETS:
        v = evaluate(model, processor, t)
        print(f"  {t:<14} semantic={v:.4f}")
        rows.append(dict(ep=0, ds=t, semantic=v))
    print("  ^ 若与零训练数值不符，说明注入破坏了前向，须排查")

    best, best_ep, nskip_total = -1, -1, 0
    for ep in range(1, EPOCHS + 1):
        model.train(); random.shuffle(train); t0 = time.time()
        losses, nskip, noinst = [], 0, 0
        for ip, mp in train:
            img = Image.open(ip).convert("RGB")
            gt = np.array(Image.open(mp).convert("L")) > 127
            img, gt = augment(img, gt)
            gt_t = torch.from_numpy(gt).float().to(DEV)
            try:
                out = model(**make_inputs(processor, img))
                pm = out["pred_masks"][0]; lg = out["pred_logits"][0]; sem = out["semantic_seg"][0, 0]
                loss = W_SEM * (dice_loss(sem, gt_t) + bce_mask(sem, gt_t))
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
        nskip_total += nskip

        if ep in EVAL_EPS:
            sc = {t: evaluate(model, processor, t) for t in TESTSETS}
            for t in TESTSETS: rows.append(dict(ep=ep, ds=t, semantic=sc[t]))
            indep = (sc["NHA12D"] + sc["Crack500"]) / 2
            print(f"ep{ep:02d} loss={np.mean(losses):.3f} skip={nskip} 无实例={noinst} | "
                  f"ds400={sc['dataset_400']:.4f} DeepCrack={sc['DeepCrack']:.4f} "
                  f"NHA12D={sc['NHA12D']:.4f} Crack500={sc['Crack500']:.4f} "
                  f"| 独立均={indep:.4f}(仅观测) {time.time()-t0:.0f}s")
            snap = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
            torch.save(snap, os.path.join(OUT, "ckpts", "lora", f"lora_r{args.rank}_ep{ep}.pt"))
            if sc["dataset_400"] > best:
                best, best_ep = sc["dataset_400"], ep
                torch.save(snap, os.path.join(OUT, "ckpts", f"sam3_lora_r{args.rank}_best.pt"))
                print(f"     >> 存best (ep{ep}, ds400={best:.4f})")
        else:
            print(f"ep{ep:02d} loss={np.mean(losses):.3f} skip={nskip} 无实例={noinst} {time.time()-t0:.0f}s")

    cp = os.path.join(OUT, "results", f"train_lora_r{args.rank}_curve_{ts}.csv")
    with open(cp, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    print("\n" + "=" * 96)
    print(f"完成。best = ep{best_ep} (dataset_400 = {best:.4f})")
    ind = [(r["ep"], r["semantic"]) for r in rows if r["ds"] == "NHA12D"]
    print(f"训练期异常跳过累计 {nskip_total}")
    print(f"曲线CSV {cp}")
    print("\n⚠ 判读: 按域内准则正选出的 checkpoint，其独立跨域均值")
    print("   > 0.5779(零训练) -> LoRA 确有跨域收益，'微调无助于跨域'须限定为'全模块微调'")
    print("   <= 0.5779        -> 结论扩展到参数高效微调，对 CrackSAM 的反驳有了防守")
    sys.stdout = tee.o; tee.close()


if __name__ == "__main__":
    main()