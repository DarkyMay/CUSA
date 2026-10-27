#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e1_collapse_mitigation.py —— 回应 R2-6（微调宽度塌缩的缓解）+ 顺带落盘 YOLO 掩膜

一次性跑完五个阶段，任一阶段异常跳过继续，全部结果写 txt/csv：
  S0  YOLO 掩膜落盘        yolo_rob@640 / yolo_rob@2016 / yolo_cv12k@640  -> results/masks/<name>/
  E0a 训练集 vs SUT 的 GT 裂缝宽度（以 288 输出格为单位）分布；逐图量出训练图宽度，供尺度感知训练使用
  E0b 现有 v3 快照（ep1..25）在 ds400-test / NHA12D / SUT 上的宽度斜率曲线
      —— 检验"宽度感知的 checkpoint 选择"是否无需 SUT 即可做
  E1  四个训练变体 × 10 epoch（评测点 1,2,3,5,10）：
      A 对照(v3原样)  B 尺度感知训练(逐图缩放到 1–4 格)  C 面积保持损失  D B+C
      每个评测点：四集 Dice + ds400-test/NHA12D/SUT 的宽度斜率/ρ/MAE + 前景比 r
  汇总 两种选择准则（held-out Dice vs 宽度感知）下各变体选中的 epoch 及其 SUT 斜率

SUT-Crack 全程不参与训练与选择。
用法:  python3 e1_collapse_mitigation.py            全量（约 3-3.5 h）
       python3 e1_collapse_mitigation.py --quick    冒烟（每集 ≤6 张, 1 epoch）
       --skip_yolo / --skip_e0b / --variants A,C    可选
"""
import os, sys, glob, re, csv, time, json, random, argparse, datetime, traceback
import xml.etree.ElementTree as ET
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image, ImageOps, ImageEnhance
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from skimage.morphology import skeletonize
from transformers import Sam3Model, Sam3Processor

sys.path.insert(0, "/root/autodl-tmp/proj_bridge")
try:
    import pavement_codes as pc; HAVE_PC = True
except Exception:
    HAVE_PC = False

Image.MAX_IMAGE_PIXELS = None
OUT = "/root/autodl-tmp/proj_bridge"
W = "/root/autodl-tmp/sam3_weights"
DS400 = "/root/autodl-tmp/dataset_400_clean"
DEEPCRACK = "/root/autodl-tmp/DeepCrack-master/dataset"
NHA = "/root/autodl-tmp/NHA12D/NHA12D_dataset"
SUBS = ["Asphalt_pavement_FD", "Asphalt_pavement_VD", "Concrete_pavement_FD", "Concrete_pavement_VD"]
SUT = "/root/autodl-tmp/SUT-Crack"
SUT_IMG = os.path.join(SUT, "1-Segmentation", "Original_Image")
SUT_GT = os.path.join(SUT, "1-Segmentation", "Ground_Truth")
CLOSELOOP = os.path.join(OUT, "results", "closeloop_v2_20260726_170420.csv")
V3DIR = os.path.join(OUT, "ckpts", "v3")
MASKD = os.path.join(OUT, "results", "masks")
YOLO_W = [("yolo_rob", "/root/autodl-tmp/crack-yolo-seg-best.pt", [640, 2016]),
          ("yolo_cv12k", "/root/autodl-tmp/CrackVision12K-yolo-seg-best.pt", [640])]
DEV = "cuda"; TEXT = "crack"; SEED = 1234
MMPX = pc.SUTCRACK_MMPX if HAVE_PC else 0.2219
R_OUT = 288                        # semantic 头输出格数
LR = 1e-4; W_MASK = 1.0; W_SCORE = 2.0; W_SEM = 1.0; W_AREA = 1.0
EPOCHS = 10; EVAL_EPS = [1, 2, 3, 5, 10]
MODS = ["mask_decoder", "detr_decoder", "dot_product_scoring"]

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"e1_mitig_{ts}.txt")
CSV_EVAL = os.path.join(OUT, "results", f"e1_mitig_eval_{ts}.csv")
CSV_W = os.path.join(OUT, "results", f"e1_mitig_widths_{ts}.csv")
CKD = os.path.join(OUT, "ckpts", "e1"); os.makedirs(CKD, exist_ok=True)


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def dice_np(a, b):
    a = a.astype(bool); b = b.astype(bool); s = a.sum() + b.sum()
    return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def numkey(p):
    m = re.findall(r'\d+', os.path.basename(p)); return int(m[0]) if m else 0


def epkey(p):
    """快照文件名里的 epoch 号：sam3_v3_ep3.pt -> 3（numkey 会误取 'sam3' 里的 3）"""
    m = re.search(r'ep(\d+)', os.path.basename(p)); return int(m.group(1)) if m else -1


# ---------------------------------------------------------------- 宽度测量
def width_profile_px(mask):
    """论文定义：骨架点处 2*EDT。裁到 bbox 加速。返回 (中值, 均值, p95, 最大) 像素；掩膜太小返回 None"""
    if mask.sum() < 30: return None
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - 40), min(mask.shape[0], ys.max() + 41)
    x0, x1 = max(0, xs.min() - 40), min(mask.shape[1], xs.max() + 41)
    m = mask[y0:y1, x0:x1]
    sk = skeletonize(m)
    if sk.sum() < 5: return None
    d = ndi.distance_transform_edt(m)
    w = 2.0 * d[sk]
    return float(np.median(w)), float(np.mean(w)), float(np.percentile(w, 95)), float(np.max(w))


def width_median_px(mask):
    """优先用 pavement_codes.measure（与论文同一实现），失败则用本地实现"""
    if HAVE_PC:
        try:
            if mask.sum() < 30: return None
            ys, xs = np.where(mask)
            y0, y1 = max(0, ys.min() - 60), min(mask.shape[0], ys.max() + 61)
            x0, x1 = max(0, xs.min() - 60), min(mask.shape[1], xs.max() + 61)
            g = pc.measure(mask[y0:y1, x0:x1], mm_per_px=1.0)
            if g and "w_median" in g: return float(g["w_median"])
        except Exception:
            pass
    r = width_profile_px(mask)
    return None if r is None else r[0]


def cell_px(H, Wd):
    """SAM3 把整图重采样到 1008^2 再出 288^2，一格在原图上沿法向约 sqrt(W*H)/288 像素"""
    return float(np.sqrt(H * Wd) / R_OUT)


# ---------------------------------------------------------------- 数据
def train_pairs():
    items = []
    idir = os.path.join(DS400, "images/train"); mdir = os.path.join(DS400, "masks/train")
    for f in sorted(os.listdir(idir)):
        if f.endswith(".jpg"):
            items.append((os.path.join(idir, f), os.path.join(mdir, os.path.splitext(f)[0] + ".png"), 127))
    for img in sorted(glob.glob(os.path.join(DEEPCRACK, "train_img", "*.jpg"))):
        mp = os.path.join(DEEPCRACK, "train_lab", os.path.splitext(os.path.basename(img))[0] + ".png")
        if os.path.exists(mp): items.append((img, mp, 127))
    return items


def eval_pairs(name):
    if name == "dataset_400":
        d = DS400
        return [(os.path.join(d, "images/test", f), os.path.join(d, "masks/test", os.path.splitext(f)[0] + ".png"), 127)
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
                    ip = os.path.join("/root/autodl-tmp/CRACK500", p[0]); mp = os.path.join("/root/autodl-tmp/CRACK500", p[1])
                    if os.path.exists(ip) and os.path.exists(mp): out.append((ip, mp, 127))
        return out[:100]
    if name == "SUT":
        out = []
        for ip in sorted(glob.glob(os.path.join(SUT_IMG, "*.jpg")), key=numkey):
            stem = os.path.splitext(os.path.basename(ip))[0]
            mp = os.path.join(SUT_GT, stem + ".png")
            if os.path.exists(mp): out.append((ip, mp, 127))
        return out


def load_sut_ref():
    """closeloop CSV 的 gt_w_median(mm) 与线状标签；与 Table collapse 同一参考"""
    ref = {}; linear = set()
    with open(CLOSELOOP, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["variant"] == "zeroshot" and r["cond"] == "text_full":
                ref[str(r["stem"])] = float(r["gt_w_median"])
                if r["gt_ctype"] == "线状": linear.add(str(r["stem"]))
    return ref, linear


def open_img(ip):
    return ImageOps.exif_transpose(Image.open(ip)).convert("RGB")


# ---------------------------------------------------------------- 模型
def make_inputs(processor, img):
    inp = processor(images=img, text=TEXT, return_tensors="pt")
    return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in inp.items()}


@torch.no_grad()
def sem_mask(model, processor, img):
    out = model(**make_inputs(processor, img))
    Wd, Hd = img.size
    return F.interpolate(out["semantic_seg"][0, 0][None, None].float(), size=(Hd, Wd), mode="bilinear")[0, 0].cpu().numpy() > 0


def load_snapshot(model, base_sd, path):
    model.load_state_dict(base_sd)
    if path is None: return
    sd = torch.load(path, map_location="cpu")
    for k in ["model", "state_dict"]:
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict): sd = sd[k]; break
    model.load_state_dict(sd, strict=False)


def dice_loss(logit, target):
    p = torch.sigmoid(logit.float())
    if p.shape != target.shape:
        p = F.interpolate(p[None, None], size=target.shape, mode="bilinear", align_corners=False)[0, 0]
    inter = (p * target).sum(); s = p.sum() + target.sum()
    return 1 - (2 * inter + 1) / (s + 1)


def bce_mask(logit, target):
    if logit.shape != target.shape:
        logit = F.interpolate(logit[None, None].float(), size=target.shape, mode="bilinear", align_corners=False)[0, 0]
    return F.binary_cross_entropy_with_logits(logit.float(), target)


def area_loss(sem_logit, target):
    """面积保持项：|log(Σσ(sem) / ΣGT)|，在 288 格上算；直接压住论文发现的 r 收缩"""
    gs = F.interpolate(target[None, None], size=sem_logit.shape[-2:], mode="area")[0, 0]
    ps = torch.sigmoid(sem_logit.float()).sum(); gsum = gs.sum()
    return torch.abs(torch.log((ps + 1.0) / (gsum + 1.0)))


# ---------------------------------------------------------------- 评测
@torch.no_grad()
def eval_dice(model, processor, dsname, limit=0):
    model.eval(); ds = []; fg = []
    pairs = eval_pairs(dsname); pairs = pairs[:limit] if limit else pairs
    for ip, mp, thr in pairs:
        try:
            img = Image.open(ip).convert("RGB")
            gt = np.array(Image.open(mp).convert("L")) > thr
            with torch.autocast(DEV, dtype=torch.bfloat16):
                sm = sem_mask(model, processor, img)
            ds.append(dice_np(sm, gt)); fg.append((sm.sum() + 1) / (gt.sum() + 1))
        except Exception:
            torch.cuda.empty_cache(); ds.append(0.0)
    return float(np.mean(ds)), float(np.median(fg)) if fg else np.nan


@torch.no_grad()
def eval_width(model, processor, dsname, limit=0, sut_ref=None, sut_linear=None, tag="", wrows=None):
    """逐图 预测中值宽 vs GT 中值宽（像素；SUT 用 mm 与 CSV 参考）；返回 slope, intercept, rho, MAE, n"""
    model.eval(); xs = []; ys = []
    pairs = eval_pairs(dsname); pairs = pairs[:limit] if limit else pairs
    for ip, mp, thr in pairs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        try:
            if dsname == "SUT":
                if sut_linear is not None and stem not in sut_linear: continue
                img = open_img(ip); x = sut_ref[stem]
            else:
                img = Image.open(ip).convert("RGB")
                gt = np.array(Image.open(mp).convert("L")) > thr
                x = width_median_px(gt)
                if x is None: continue
            with torch.autocast(DEV, dtype=torch.bfloat16):
                sm = sem_mask(model, processor, img)
            y = width_median_px(sm)
            if y is None: y = 0.0
            if dsname == "SUT": y *= MMPX
            xs.append(x); ys.append(y)
            if wrows is not None: wrows.append(dict(tag=tag, ds=dsname, stem=stem, gt=x, pred=y))
        except Exception:
            torch.cuda.empty_cache(); continue
    xs = np.array(xs); ys = np.array(ys)
    if len(xs) < 5: return dict(slope=np.nan, icpt=np.nan, rho=np.nan, mae=np.nan, n=len(xs))
    b, a = np.polyfit(xs, ys, 1); rho = spearmanr(xs, ys).correlation
    return dict(slope=float(b), icpt=float(a), rho=float(rho), mae=float(np.mean(np.abs(ys - xs))), n=len(xs))


def full_eval(model, processor, tag, quick, sut_ref, sut_linear, erows, wrows):
    lim = 6 if quick else 0
    rec = dict(tag=tag)
    for t in ["dataset_400", "DeepCrack", "NHA12D", "Crack500"]:
        d, r = eval_dice(model, processor, t, lim); rec[f"dice_{t}"] = d; rec[f"r_{t}"] = r
    rec["heldout_mean"] = (rec["dice_NHA12D"] + rec["dice_Crack500"]) / 2
    for t in ["dataset_400", "NHA12D", "SUT"]:
        w = eval_width(model, processor, t, lim, sut_ref, sut_linear, tag, wrows)
        for k, v in w.items(): rec[f"{k}_{t}"] = v
    erows.append(rec)
    print(f"  [{tag}] Dice ds400={rec['dice_dataset_400']:.4f} DC={rec['dice_DeepCrack']:.4f} NHA={rec['dice_NHA12D']:.4f} "
          f"C500={rec['dice_Crack500']:.4f} held={rec['heldout_mean']:.4f} | r_NHA={rec['r_NHA12D']:.2f} | "
          f"slope ds400={rec['slope_dataset_400']:.2f} NHA={rec['slope_NHA12D']:.2f} "
          f"SUT={rec['slope_SUT']:.2f}(ρ={rec['rho_SUT']:.2f} MAE={rec['mae_SUT']:.2f} a={rec['icpt_SUT']:.2f} n={rec['n_SUT']})", flush=True)
    return rec


# ---------------------------------------------------------------- S0 YOLO
def stage_yolo(quick):
    from ultralytics import YOLO
    pairs = eval_pairs("SUT"); pairs = pairs[:6] if quick else pairs
    rows = []
    for name, wpath, sizes in YOLO_W:
        if not os.path.exists(wpath): print(f"  缺权重 {wpath}"); continue
        model = YOLO(wpath)
        for sz in sizes:
            od = os.path.join(MASKD, f"{name}_{sz}"); os.makedirs(od, exist_ok=True)
            ds = []; t0 = time.time()
            for ip, mp, thr in pairs:
                stem = os.path.splitext(os.path.basename(ip))[0]
                img = open_img(ip); Wd, Hd = img.size
                try:
                    res = model.predict(img, imgsz=sz, conf=0.25, retina_masks=True, verbose=False, device=0, half=True)[0]
                    if res.masks is None or res.masks.data.shape[0] == 0:
                        m = np.zeros((Hd, Wd), bool)
                    else:
                        m = res.masks.data.any(0).cpu().numpy().astype(bool)
                        if m.shape != (Hd, Wd):
                            m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((Wd, Hd), Image.NEAREST)) > 127
                except Exception as e:
                    print(f"    !! {name}@{sz} {stem}: {e}"); m = np.zeros((Hd, Wd), bool)
                Image.fromarray(m.astype(np.uint8) * 255).save(os.path.join(od, stem + ".png"), optimize=True)
                gt = np.array(Image.open(mp).convert("L")) > thr
                ds.append(dice_np(m, gt))
            rows.append(dict(source=f"{name}_{sz}", n=len(ds), dice=float(np.mean(ds)), n_empty=int(sum(1 for d in ds if d == 0))))
            print(f"  {name}@{sz}: n={len(ds)} Dice={np.mean(ds):.4f} -> {od}  ({time.time()-t0:.0f}s)", flush=True)
        del model; torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------- E0a
TARGET_CELLS = (1.0, 4.0)   # 尺度感知训练的目标宽度范围（格），由规范阈值量级 ≈ 1–3 格决定，不使用 SUT 统计量


def stage_e0a(quick):
    """GT 宽度以输出格为单位：w_px / cell_px。训练集(620) vs SUT(130)。
    返回 train_cells: {image_path: 该图 GT 中值宽(格)}，供逐图尺度感知抖动使用"""
    sets = {"train_ds400": [p for p in train_pairs() if DS400 in p[0]],
            "train_DeepCrack": [p for p in train_pairs() if DEEPCRACK in p[0]],
            "SUT": eval_pairs("SUT")}
    train_cells = {}; pooled = []
    for name, pairs in sets.items():
        if quick: pairs = pairs[:12]
        cells = []; px = []
        for ip, mp, thr in pairs:
            gt = np.array(Image.open(mp).convert("L")) > thr
            w = width_median_px(gt)
            if w is None: continue
            c = w / cell_px(*gt.shape); cells.append(c); px.append(w)
            if name.startswith("train"): train_cells[ip] = c; pooled.append(c)
        cells = np.array(cells); px = np.array(px)
        print(f"  {name:16s} n={len(cells):3d}  GT中值宽 = {np.median(px):6.1f} px = {np.median(cells):5.2f} 格 "
              f"(IQR {np.percentile(cells,25):.2f}-{np.percentile(cells,75):.2f}, P10-P90 {np.percentile(cells,10):.2f}-{np.percentile(cells,90):.2f})")
    pooled = np.array(pooled)
    print(f"  训练集逐图合并 n={len(pooled)}: 中值 {np.median(pooled):.2f} 格, IQR {np.percentile(pooled,25):.2f}-{np.percentile(pooled,75):.2f}")
    print(f"  => 尺度感知抖动：每张训练图缩放到目标宽度 ~ logU{TARGET_CELLS} 格，缩放因子 = 目标/该图宽度，截断到 [0.25, 4]")
    return train_cells


# ---------------------------------------------------------------- 尺度抖动
def scale_jitter(img, gt, cells_i):
    """逐图尺度感知：把该图裂缝缩放到目标宽度（格）。cells_i 为该图 GT 中值宽（格），未知则取 2"""
    target = float(np.exp(random.uniform(np.log(TARGET_CELLS[0]), np.log(TARGET_CELLS[1]))))
    f = float(np.clip(target / (cells_i if cells_i and cells_i > 0 else 2.0), 0.25, 4.0))
    Wd, Hd = img.size
    if f >= 1.0:   # 放大：随机裁剪 1/f 区域（尽量包含裂缝），processor 再把它重采样到 1008 -> 裂缝像素宽 ×f
        cw, ch = max(64, int(Wd / f)), max(64, int(Hd / f))
        ys, xs = np.where(gt)
        if len(ys) > 0:
            k = random.randrange(len(ys)); cy, cx = ys[k], xs[k]
            x0 = int(np.clip(cx - cw // 2 + random.randint(-cw // 4, cw // 4), 0, Wd - cw))
            y0 = int(np.clip(cy - ch // 2 + random.randint(-ch // 4, ch // 4), 0, Hd - ch))
        else:
            x0 = random.randint(0, Wd - cw); y0 = random.randint(0, Hd - ch)
        return img.crop((x0, y0, x0 + cw, y0 + ch)), gt[y0:y0 + ch, x0:x0 + cw].copy()
    # 缩小：整图降采样到 f 倍，再反射补边回原尺寸 -> 裂缝像素宽 ×f
    nw, nh = max(64, int(Wd * f)), max(64, int(Hd * f))
    im = np.array(img.resize((nw, nh), Image.BILINEAR))
    g = np.array(Image.fromarray(gt.astype(np.uint8) * 255).resize((nw, nh), Image.NEAREST)) > 127
    py, px_ = Hd - nh, Wd - nw
    im = np.pad(im, ((0, py), (0, px_), (0, 0)), mode="reflect")
    g = np.pad(g, ((0, py), (0, px_)), mode="reflect")
    return Image.fromarray(im), g


def augment_base(img, gt):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT); gt = gt[:, ::-1].copy()
    if random.random() < 0.5:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3))
    return img, gt


# ---------------------------------------------------------------- E1 训练
def train_variant(vname, use_jitter, use_area, train_cells, model, processor, base_sd, quick,
                  sut_ref, sut_linear, erows, wrows):
    seed_all(SEED)
    model.load_state_dict(base_sd)
    for p in model.parameters(): p.requires_grad_(False)
    for n, p in model.named_parameters():
        if any(m in n for m in MODS): p.requires_grad_(True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4)
    train = train_pairs(); train = train[:12] if quick else train
    eps = 1 if quick else EPOCHS; evals = [1] if quick else EVAL_EPS
    print(f"\n==== 变体 {vname}: scale_aware={use_jitter}(目标 {TARGET_CELLS} 格, 已量 {len(train_cells)} 张) area_loss={use_area}  训练 {len(train)} 张 × {eps} ep", flush=True)
    fs = []
    for ep in range(1, eps + 1):
        model.train(); random.shuffle(train); t0 = time.time(); losses = []; aL = []; nskip = 0
        for ip, mp, thr in train:
            try:
                img = Image.open(ip).convert("RGB")
                gt = np.array(Image.open(mp).convert("L")) > thr
                img, gt = augment_base(img, gt)
                if use_jitter:
                    w0, h0 = img.size
                    img, gt = scale_jitter(img, gt, train_cells.get(ip))
                    fs.append(img.size[0] / w0 if img.size[0] != w0 else -1)
                gt_t = torch.from_numpy(np.ascontiguousarray(gt)).float().to(DEV)
                out = model(**make_inputs(processor, img))
                pm = out["pred_masks"][0]; lg = out["pred_logits"][0]; sem = out["semantic_seg"][0, 0]
                loss = W_SEM * (dice_loss(sem, gt_t) + bce_mask(sem, gt_t))
                if use_area:
                    la = area_loss(sem, gt_t); loss = loss + W_AREA * la; aL.append(la.item())
                if pm.shape[0] > 0:
                    with torch.no_grad():
                        gs = F.interpolate(gt_t[None, None], size=pm.shape[-2:], mode="nearest")[0, 0]
                        pmb = (pm > 0).float(); inter = (pmb * gs).sum((1, 2))
                        iou = inter / (pmb.sum((1, 2)) + gs.sum() - inter + 1e-6); match = int(iou.argmax())
                    loss = loss + W_MASK * (dice_loss(pm[match], gt_t) + bce_mask(pm[match], gt_t))
                    st = torch.zeros_like(lg); st[match] = 1.0
                    loss = loss + W_SCORE * F.binary_cross_entropy_with_logits(lg.float(), st)
                opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
            except Exception:
                torch.cuda.empty_cache(); nskip += 1; continue
        msg = f"  {vname} ep{ep:02d} loss={np.mean(losses):.3f}" + (f" area={np.mean(aL):.3f}" if aL else "") + f" skip={nskip} {time.time()-t0:.0f}s"
        if fs:
            fs_ = np.array(fs); msg += f" | 本epoch缩放: 裁剪(放大) {np.mean(fs_ > 0 ):.0%}, 降采样(缩小) {np.mean(fs_ < 0):.0%}"; fs = []
        print(msg, flush=True)
        if ep in evals:
            snap = {n: model.state_dict()[n] for n in model.state_dict() if any(m in n for m in MODS)}
            torch.save(snap, os.path.join(CKD, f"{vname}_ep{ep}.pt"))
            full_eval(model, processor, f"{vname}_ep{ep}", quick, sut_ref, sut_linear, erows, wrows)
    for p in model.parameters(): p.requires_grad_(False)
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- 汇总
def summarize(erows):
    print("\n" + "=" * 100 + "\n汇总：两种 checkpoint 选择准则下的宽度保真\n" + "=" * 100)
    print("准则1 = held-out Dice 均值最高（论文现行）；准则2 = ds400-test 上宽度斜率最接近 1（宽度感知，不触碰 held-out 与 SUT）")
    zs = [r for r in erows if r["tag"] == "zeroshot"]
    if zs: print(f"  零样本参照: SUT slope={zs[0]['slope_SUT']:.2f} ρ={zs[0]['rho_SUT']:.2f} held={zs[0]['heldout_mean']:.4f}")
    groups = []
    for r in erows:
        gname = r["tag"].rsplit("_ep", 1)[0] + "_ep" if "_ep" in r["tag"] else None
        if gname and gname not in groups: groups.append(gname)
    for group in groups:
        rr = [r for r in erows if r["tag"].startswith(group)]
        if not rr: continue
        s1 = max(rr, key=lambda r: r["heldout_mean"])
        s2 = min(rr, key=lambda r: abs(r["slope_dataset_400"] - 1) if not np.isnan(r["slope_dataset_400"]) else 9)
        print(f"  {group[:-3]:4s} 准则1 -> {s1['tag']:10s} held={s1['heldout_mean']:.4f} SUT slope={s1['slope_SUT']:.2f} ρ={s1['rho_SUT']:.2f} | "
              f"准则2 -> {s2['tag']:10s} held={s2['heldout_mean']:.4f} SUT slope={s2['slope_SUT']:.2f} ρ={s2['rho_SUT']:.2f}")
    print("\n逐点表（tag | heldout Dice | r_NHA | slope ds400 / NHA / SUT | ρ_SUT | MAE_SUT）")
    for r in erows:
        print(f"  {r['tag']:12s} {r['heldout_mean']:.4f}  {r['r_NHA12D']:5.2f}  {r['slope_dataset_400']:5.2f} {r['slope_NHA12D']:5.2f} "
              f"{r['slope_SUT']:5.2f}  {r['rho_SUT']:5.2f}  {r['mae_SUT']:5.2f}")
    print("\n判读: 若某变体在准则1选中的 epoch 上 SUT slope ≥ 0.6 且 held-out 不低于对照 A，缓解成立；"
          "若仅准则2能保住斜率，则宽度感知选择是可写入方法的程序性缓解；两者皆否则塌缩源于标注惯例。")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip_yolo", action="store_true")
    ap.add_argument("--skip_e0b", action="store_true")
    ap.add_argument("--variants", type=str, default="A,B,C,D")
    ap.add_argument("--lr", type=float, default=None, help="覆盖学习率（默认 1e-4）")
    ap.add_argument("--skip_e0a", action="store_true")
    ap.add_argument("--tag", type=str, default="", help="变体名后缀，如 _lr1e-5")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数；>10 时评测点自动加 15,20,25")
    args = ap.parse_args()
    if args.lr is not None: globals()["LR"] = args.lr
    if args.seed is not None: globals()["SEED"] = args.seed
    if args.epochs is not None:
        globals()["EPOCHS"] = args.epochs
        globals()["EVAL_EPS"] = [e for e in [1, 2, 3, 5, 10, 15, 20, 25, 30, 40] if e <= args.epochs]
    sys.stdout = Tee(LOGP)
    T0 = time.time()
    print(f"e1_collapse_mitigation.py  {ts}  quick={args.quick}  lr={LR:g}  seed={SEED}  epochs={EPOCHS}  eval={EVAL_EPS}  variants={args.variants}{args.tag}  {torch.cuda.get_device_name(0)}  pavement_codes={'yes' if HAVE_PC else 'no'}")
    print(f"log -> {LOGP}\n")
    erows = []; wrows = []

    # S0
    print("=" * 100 + "\n[S0] YOLO 掩膜落盘\n" + "=" * 100)
    if not args.skip_yolo:
        try: stage_yolo(args.quick)
        except Exception: traceback.print_exc()
    else: print("  跳过")

    # E0a
    print("\n" + "=" * 100 + "\n[E0a] GT 宽度（输出格单位）：训练集 vs SUT\n" + "=" * 100)
    train_cells = {}
    if not args.skip_e0a:
        try: train_cells = stage_e0a(args.quick)
        except Exception: traceback.print_exc()
    else: print("  跳过")

    # 模型
    seed_all(SEED)
    processor = Sam3Processor.from_pretrained(W)
    model = Sam3Model.from_pretrained(W).to(DEV).eval()
    for p in model.parameters(): p.requires_grad_(False)
    base_sd = {k: v.clone() for k, v in model.state_dict().items()}
    sut_ref, sut_linear = load_sut_ref()
    print(f"\nSUT 参考: {len(sut_ref)} 张, 线状 {len(sut_linear)} 张")

    # E0b
    print("\n" + "=" * 100 + "\n[E0b] 现有 v3 快照的宽度斜率曲线（论文 Table collapse SUT: zs 0.79, ep1 0.38, ep5 0.27, ep15 0.16, ep25 0.21）\n" + "=" * 100)
    try:
        full_eval(model, processor, "zeroshot", args.quick, sut_ref, sut_linear, erows, wrows)
        if not args.skip_e0b:
            snaps = sorted(glob.glob(os.path.join(V3DIR, "sam3_v3_ep*.pt")), key=epkey)
            if args.quick: snaps = snaps[:2]
            for sp in snaps:
                ep = epkey(sp); load_snapshot(model, base_sd, sp)
                full_eval(model, processor, f"v3_ep{ep}", args.quick, sut_ref, sut_linear, erows, wrows)
            load_snapshot(model, base_sd, None)
    except Exception: traceback.print_exc()

    # E1
    print("\n" + "=" * 100 + "\n[E1] 训练变体\n" + "=" * 100)
    spec = {"A": (False, False), "B": (True, False), "C": (False, True), "D": (True, True)}
    for v in [s.strip() for s in args.variants.split(",") if s.strip()]:
        try: train_variant(v + args.tag, *spec[v], train_cells, model, processor, base_sd, args.quick, sut_ref, sut_linear, erows, wrows)
        except Exception: traceback.print_exc()

    # 输出
    if erows:
        keys = sorted(set(k for r in erows for k in r), key=lambda k: (k != "tag", k))
        with open(CSV_EVAL, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(erows)
    if wrows:
        with open(CSV_W, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(wrows[0].keys())); w.writeheader(); w.writerows(wrows)
    try: summarize(erows)
    except Exception: traceback.print_exc()
    print(f"\n评测CSV {CSV_EVAL}\n逐图宽度CSV {CSV_W}\n快照 {CKD}/\n总耗时 {(time.time()-T0)/60:.1f} min")


if __name__ == "__main__":
    main()