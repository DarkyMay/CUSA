#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
h_hs_prep.py —— R3-9：为 Hybrid-Segmentor 准备"与 SAM3 微调完全相同的 620 张标注"训练集，并生成改好配置的训练目录

  数据: /root/autodl-tmp/data/Ours620-HS-format/{train,val,test}/{IMG,GT}
        train = dataset_400 train(320) + DeepCrack train(300) 切 256×256 原分辨率块（含裂缝块全保留 + 25% 空白块）
        val = test = dataset_400 test(20) 全部切块   —— 与 SAM3 v3 的 checkpoint 选择同一依据，不碰 held-out
        同名 .png（HS 的 dataset.py 用同一文件名在 IMG/GT 下取图与掩膜）
  代码: /root/autodl-tmp/Hybrid-Segmentor/Hybrid_Segmentor_ours620/  —— 复制原目录（不含 checkpoints/tb_logs），
        改 config.py 的 dataset / DATASET_SIZE / NUM_EPOCHS，callback.py 的 dirpath / patience，trainer.py 的 logger 名
用法 python3 h_hs_prep.py
"""
import os, re, glob, shutil, random
import numpy as np
from PIL import Image

SRC = "/root/autodl-tmp/Hybrid-Segmentor/Hybrid_Segmentor"
DST = "/root/autodl-tmp/Hybrid-Segmentor/Hybrid_Segmentor_ours620"
DATA = "/root/autodl-tmp/data/Ours620-HS-format"
DS400 = "/root/autodl-tmp/dataset_400_clean"; DEEPCRACK = "/root/autodl-tmp/DeepCrack-master/dataset"
T = 256; EPOCHS = 150; PATIENCE = 30; SEED = 20260728
random.seed(SEED)


def pairs_train():
    items = []
    for f in sorted(os.listdir(os.path.join(DS400, "images/train"))):
        if f.endswith(".jpg"): items.append((os.path.join(DS400, "images/train", f), os.path.join(DS400, "masks/train", os.path.splitext(f)[0] + ".png"), 127))
    for ip in sorted(glob.glob(os.path.join(DEEPCRACK, "train_img", "*.jpg"))):
        mp = os.path.join(DEEPCRACK, "train_lab", os.path.splitext(os.path.basename(ip))[0] + ".png")
        if os.path.exists(mp): items.append((ip, mp, 127))
    return items


def pairs_val():
    return [(os.path.join(DS400, "images/test", f), os.path.join(DS400, "masks/test", os.path.splitext(f)[0] + ".png"), 127)
            for f in sorted(os.listdir(os.path.join(DS400, "images/test"))) if f.endswith(".jpg")]


def tile_and_save(items, outdir, keep_empty_frac):
    os.makedirs(os.path.join(outdir, "IMG"), exist_ok=True); os.makedirs(os.path.join(outdir, "GT"), exist_ok=True)
    n_pos = n_neg = 0
    for ip, mp, thr in items:
        img = np.array(Image.open(ip).convert("RGB")); gt = (np.array(Image.open(mp).convert("L")) > thr)
        H, W = gt.shape; stem = os.path.splitext(os.path.basename(ip))[0]
        ph, pw = (-H) % T, (-W) % T
        if ph or pw:
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect"); gt = np.pad(gt, ((0, ph), (0, pw)), mode="reflect")
        for y in range(0, gt.shape[0], T):
            for x in range(0, gt.shape[1], T):
                g = gt[y:y + T, x:x + T]; frac = g.mean()
                if frac < 0.001:
                    if random.random() > keep_empty_frac: continue
                    n_neg += 1
                else: n_pos += 1
                name = f"{stem}_{y}_{x}.png"
                Image.fromarray(img[y:y + T, x:x + T]).save(os.path.join(outdir, "IMG", name))
                Image.fromarray((g.astype(np.uint8) * 255)).save(os.path.join(outdir, "GT", name))
    return n_pos, n_neg


def main():
    if os.path.exists(DATA): shutil.rmtree(DATA)
    tp, tn = tile_and_save(pairs_train(), os.path.join(DATA, "train"), 0.25)
    vp, vn = tile_and_save(pairs_val(), os.path.join(DATA, "val"), 1.0)
    # test 目录 = val（trainer.test 需要；只用于日志）
    shutil.copytree(os.path.join(DATA, "val"), os.path.join(DATA, "test"))
    n_tr = len(os.listdir(os.path.join(DATA, "train/IMG"))); n_va = len(os.listdir(os.path.join(DATA, "val/IMG")))
    print(f"train 块 {n_tr}（含裂缝 {tp} + 空白 {tn}），val 块 {n_va}（含裂缝 {vp} + 空白 {vn}）  -> {DATA}")

    if os.path.exists(DST): shutil.rmtree(DST)
    shutil.copytree(SRC, DST, ignore=shutil.ignore_patterns("checkpoints", "tb_logs", "__pycache__", "*.ckpt"))
    cfg = open(os.path.join(DST, "config.py")).read()
    cfg = re.sub(r"^NUM_EPOCHS\s*=.*$", f"NUM_EPOCHS = {EPOCHS}", cfg, flags=re.M)
    cfg = re.sub(r"^DATASET_SIZE\s*=.*$", f"DATASET_SIZE = {{'train' : {n_tr}, 'val' : {n_va}, 'test' : {n_va}}}", cfg, flags=re.M)
    cfg = re.sub(r"^dataset\s*=.*$", f"dataset = '{DATA}/'", cfg, flags=re.M)
    open(os.path.join(DST, "config.py"), "w").write(cfg)
    cb = open(os.path.join(DST, "callback.py")).read()
    cb = re.sub(r"dirpath=os\.path\.join\(os\.getcwd\(\), 'checkpoints', '[^']*'\)", "dirpath=os.path.join(os.getcwd(), 'checkpoints', 'v7_ours620')", cb)
    cb = re.sub(r"patience=\d+", f"patience={PATIENCE}", cb)
    open(os.path.join(DST, "callback.py"), "w").write(cb)
    tr = open(os.path.join(DST, "trainer.py")).read()
    tr = re.sub(r'TensorBoardLogger\("tb_logs", name="[^"]*"\)', 'TensorBoardLogger("tb_logs", name="v7_ours620")', tr)
    open(os.path.join(DST, "trainer.py"), "w").write(tr)
    print(f"训练目录 {DST}")
    for f, pat in [("config.py", r"^(NUM_EPOCHS|DATASET_SIZE|dataset)\s*="), ("callback.py", r"dirpath=|patience="), ("trainer.py", r"TensorBoardLogger")]:
        for line in open(os.path.join(DST, f)):
            if re.search(pat, line): print(f"  {f}: {line.rstrip()}")


if __name__ == "__main__":
    main()