#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
e8_hybrid_segmentor_baseline.py —— 裂缝专用监督基线 Hybrid-Segmentor（HS）在四个评估域上的 Dice（回应审稿 M10）

权重
  hs_cv12k : weights/hybrid_segmentor_BCE_2.ckpt              官方，CrackVision12K 训练（λ=0.2）
  hs_rob   : checkpoints/v7_BCEDICE0_2_final_roboflow/*ep83*  本组复训，Roboflow 训练
公平性
  CV12K 包含 CRACK500 与 DeepCrack → hs_cv12k 在这两个域上属域内，只作参考并加标记；在 NHA12D、dataset_400 上公平。
  hs_rob 在四个域上均为外部数据基线，全部公平。
推理协议（HS 只接受 256×256）
  whole : 整图缩到 256×256（HS 训练/测试方式），logit 双线性放回原尺寸后 >0
  tile  : 原分辨率 256×256 滑窗（步长 256，边缘补零），并集
图像集与配对
  与 results/paired_v2.csv 完全一致（NHA12D 80 / Crack500 100 / DeepCrack 60 / dataset_400 20），
  zeroshot 列 = SAM3 零样本逐图 Dice，用于 Wilcoxon 配对；unet_resnet34_whole 列作参照。
用法：python3 e8_hybrid_segmentor_baseline.py [--limit N]
输出：results/e8_hs_<ts>.txt, results/e8_hs_perimage_<ts>.csv
"""
import os, sys, csv, time, argparse, datetime
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from scipy.stats import wilcoxon

HS_DIR = "/root/autodl-tmp/Hybrid-Segmentor/Hybrid_Segmentor"
sys.path.insert(0, HS_DIR)
OUT = "/root/autodl-tmp/proj_bridge"
PAIRED = os.path.join(OUT, "results", "paired_v2.csv")
WEIGHTS = [("hs_cv12k", "/root/autodl-tmp/Hybrid-Segmentor/weights/hybrid_segmentor_BCE_2.ckpt"),
           ("hs_rob", "/root/autodl-tmp/Hybrid-Segmentor/Hybrid_Segmentor/checkpoints/v7_BCEDICE0_2_final_roboflow/"
                      "v7-epochepoch=83-val_lossval_loss=0.6817.ckpt")]
LEAK = {"hs_cv12k": {"Crack500", "DeepCrack"}}   # 域内泄漏标记
DS400 = "/root/autodl-tmp/dataset_400_clean"; DEEPCRACK = "/root/autodl-tmp/DeepCrack-master/dataset"
NHA = "/root/autodl-tmp/NHA12D/NHA12D_dataset"; CRACK500 = "/root/autodl-tmp/CRACK500"
MU = torch.tensor([0.51789941, 0.51360926, 0.547762]).view(1, 3, 1, 1)
SD = torch.tensor([0.1812099, 0.17746663, 0.20386334]).view(1, 3, 1, 1)
DEV = "cuda"; TILE = 256; BATCH = 32
TESTSETS = ["dataset_400", "DeepCrack", "NHA12D", "Crack500"]
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOGP = os.path.join(OUT, "results", f"e8_hs_{ts}.txt")
CSVP = os.path.join(OUT, "results", f"e8_hs_perimage_{ts}.csv")


class Tee:
    def __init__(s, p): s.f = open(p, "w", encoding="utf-8"); s.o = sys.stdout
    def write(s, x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()


def dice_np(a, b): s = a.sum() + b.sum(); return 1.0 if s == 0 else 2.0 * np.logical_and(a, b).sum() / s


def resolve(ds, fname, sub):
    stem = os.path.splitext(fname)[0]
    if ds == "dataset_400": return os.path.join(DS400, "images/test", fname), os.path.join(DS400, "masks/test", stem + ".png"), 127
    if ds == "DeepCrack": return os.path.join(DEEPCRACK, "test_img", fname), os.path.join(DEEPCRACK, "test_lab", stem + ".png"), 127
    if ds == "NHA12D":
        subs = [sub] if sub and sub != "-" else ["Asphalt_pavement_FD", "Asphalt_pavement_VD", "Concrete_pavement_FD", "Concrete_pavement_VD"]
        for s_ in subs:
            ip = os.path.join(NHA, s_, "Raw_images", fname); mp = os.path.join(NHA, s_, "Masks", stem + ".png")
            if os.path.exists(ip) and os.path.exists(mp): return ip, mp, 0
        return None
    if ds == "Crack500":
        for d in ("testcrop", "traincrop", "valcrop"):
            ip = os.path.join(CRACK500, d, fname); mp = os.path.join(CRACK500, d, stem + ".png")
            if os.path.exists(ip) and os.path.exists(mp): return ip, mp, 127
        return None
    return None


def load_items():
    rows = list(csv.DictReader(open(PAIRED, encoding="utf-8")))
    items = {ds: [] for ds in TESTSETS}; missing = 0
    for r in rows:
        res = resolve(r["ds"], r["file"], r.get("sub", "-"))
        if res is None: missing += 1; continue
        items[r["ds"]].append(dict(file=r["file"], img=res[0], mask=res[1], thr=res[2],
                                   zs=float(r["zeroshot"]), unet=float(r.get("unet_resnet34_whole", "nan") or "nan")))
    return items, missing


def to_tensor(img):
    x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None]
    return (x - MU) / SD


@torch.no_grad()
def hs_logits(model, x):
    out = model(x.to(DEV))
    if isinstance(out, (list, tuple)): out = out[0]
    return out.float()          # (B,1,256,256) logits


@torch.no_grad()
def predict_whole(model, img):
    Wd, Hd = img.size
    x = to_tensor(img.resize((TILE, TILE), Image.BILINEAR))
    lg = hs_logits(model, x)
    return (F.interpolate(lg, size=(Hd, Wd), mode="bilinear")[0, 0] > 0).cpu().numpy()


@torch.no_grad()
def predict_tile(model, img):
    Wd, Hd = img.size
    arr = np.asarray(img, dtype=np.float32) / 255.0
    Hp, Wp = int(np.ceil(Hd / TILE)) * TILE, int(np.ceil(Wd / TILE)) * TILE
    pad = np.zeros((Hp, Wp, 3), np.float32); pad[:Hd, :Wd] = arr
    x = ((torch.from_numpy(pad).permute(2, 0, 1)[None]) - MU) / SD
    tiles = x.unfold(2, TILE, TILE).unfold(3, TILE, TILE)           # 1,3,ny,nx,T,T
    ny, nx = tiles.shape[2], tiles.shape[3]
    tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, TILE, TILE)
    outs = []
    for i in range(0, tiles.shape[0], BATCH):
        outs.append(hs_logits(model, tiles[i:i + BATCH]).cpu())
    lg = torch.cat(outs, 0).reshape(ny, nx, TILE, TILE).permute(0, 2, 1, 3).reshape(Hp, Wp)
    return (lg[:Hd, :Wd] > 0).numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); args = ap.parse_args()
    sys.stdout = Tee(LOGP); T0 = time.time()
    print(f"e8_hybrid_segmentor_baseline.py  {ts}  torch {torch.__version__}")
    items, missing = load_items()
    print("  图像集（与 paired_v2.csv 一致）: " + "  ".join(f"{ds} {len(v)}" for ds, v in items.items()) + f"   未解析 {missing}")
    if args.limit:
        items = {ds: v[:args.limit] for ds, v in items.items()}

    # --- torchmetrics 兼容垫片：新版本移除了 classification.Dice，HS 的 model.py 顶层会 import 它 ---
    import torchmetrics, torchmetrics.classification as tmc
    if not hasattr(tmc, "Dice"):
        class _DiceStub(torch.nn.Module):            # 推理用不到该指标；只需能以任意参数实例化
            def __init__(self, *a, **k): super().__init__()
            def update(self, *a, **k): pass
            def compute(self): return torch.tensor(0.0)
            def forward(self, *a, **k): return torch.tensor(0.0)
            def reset(self): pass
        tmc.Dice = _DiceStub
        print(f"  [垫片] torchmetrics {torchmetrics.__version__} 无 classification.Dice，已注入空壳类")
    from model import HybridSegmentor
    rows = []; summary = {}
    for wname, wpath in WEIGHTS:
        if not os.path.exists(wpath): print(f"  !! 缺少权重 {wpath}"); continue
        model = HybridSegmentor().to(DEV)
        ck = torch.load(wpath, map_location="cpu", weights_only=False)
        sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
        missing_k, unexpected_k = model.load_state_dict(sd, strict=False)
        print(f"\n[{wname}] 载入 {len(sd)} 张量；missing {len(missing_k)}  unexpected {len(unexpected_k)}"
              + (f"  示例 missing {list(missing_k)[:3]}" if missing_k else ""))
        model.eval(); del ck, sd
        nparam = sum(p.numel() for p in model.parameters()) / 1e6; print(f"  参数量 {nparam:.1f} M")
        for ds in TESTSETS:
            t0 = time.time(); d_w, d_t, zs, un = [], [], [], []
            for it in items[ds]:
                img = Image.open(it["img"]).convert("RGB")
                gt = np.array(Image.open(it["mask"]).convert("L")) > it["thr"]
                if (img.size[1], img.size[0]) != gt.shape:
                    img = img.resize((gt.shape[1], gt.shape[0]), Image.BILINEAR)
                try:
                    mw = predict_whole(model, img); mt = predict_tile(model, img)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache(); print(f"  !! OOM {ds}/{it['file']}"); continue
                dw, dt = dice_np(mw, gt), dice_np(mt, gt)
                d_w.append(dw); d_t.append(dt); zs.append(it["zs"]); un.append(it["unet"])
                rows.append(dict(weight=wname, ds=ds, file=it["file"], dice_whole=dw, dice_tile=dt,
                                 fg_whole=mw.mean() / max(gt.mean(), 1e-9), fg_tile=mt.mean() / max(gt.mean(), 1e-9),
                                 zeroshot=it["zs"], unet_whole=it["unet"]))
            if not d_w: continue
            d_w, d_t, zs, un = map(np.array, (d_w, d_t, zs, un))
            leak = ds in LEAK.get(wname, set())
            summary[(wname, ds)] = dict(whole=d_w.mean(), tile=d_t.mean(), zs=zs.mean(), unet=np.nanmean(un), n=len(d_w), leak=leak,
                                        p_w=wilcoxon(d_w, zs).pvalue if len(d_w) > 5 and np.any(d_w != zs) else np.nan,
                                        p_t=wilcoxon(d_t, zs).pvalue if len(d_t) > 5 and np.any(d_t != zs) else np.nan,
                                        better_w=np.mean(d_w > zs), better_t=np.mean(d_t > zs),
                                        med_w=np.median(d_w - zs), med_t=np.median(d_t - zs))
            print(f"  {ds:<12} n={len(d_w):>3}  whole {d_w.mean():.4f}  tile {d_t.mean():.4f}  | SAM3 zs {zs.mean():.4f}  U-Net {np.nanmean(un):.4f}"
                  f"  {'[域内泄漏]' if leak else ''}  ({time.time()-t0:.0f}s)", flush=True)
        del model; torch.cuda.empty_cache()

    if rows:
        with open(CSVP, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)

    print("\n" + "=" * 110); print("[汇总] Dice（数据集均值）；配对列为 HS 相对 SAM3 零样本：更优图占比 / 中位差 / Wilcoxon p"); print("=" * 110)
    print(f"  {'权重':<9}{'数据集':<12}{'n':>4}{'HS whole':>10}{'HS tile':>9}{'SAM3 zs':>9}{'U-Net':>8}{'':>3}"
          f"{'whole: 更优/中位差/p':>26}{'tile: 更优/中位差/p':>26}")
    for (wname, ds), s in summary.items():
        flag = "泄漏" if s["leak"] else ""
        cw = "%.0f%% / %+.3f / %.1e" % (100 * s["better_w"], s["med_w"], s["p_w"])
        ct = "%.0f%% / %+.3f / %.1e" % (100 * s["better_t"], s["med_t"], s["p_t"])
        print(f"  {wname:<9}{ds:<12}{s['n']:>4}{s['whole']:>10.4f}{s['tile']:>9.4f}{s['zs']:>9.4f}{s['unet']:>8.4f}{flag:>3}{cw:>26}{ct:>26}")
    # held-out 均值（与论文 Table segmain 的 Held-out mean 对齐）
    for wname, _ in WEIGHTS:
        ks = [(wname, "NHA12D"), (wname, "Crack500")]
        if all(k in summary for k in ks):
            hw = np.mean([summary[k]["whole"] for k in ks]); ht = np.mean([summary[k]["tile"] for k in ks])
            print(f"  {wname} held-out mean (NHA12D, Crack500): whole {hw:.4f}  tile {ht:.4f}   [SAM3 zs 0.578; U-Net 0.476]"
                  + ("   注意 Crack500 对 hs_cv12k 是域内" if wname == "hs_cv12k" else ""))
    print("\n  判读：论文 Table segmain 的公平新增行 = hs_rob 全部四域 + hs_cv12k 的 NHA12D/dataset_400；")
    print("        hs_cv12k 在 Crack500/DeepCrack 是训练域，数值只说明监督上限，不进入'held-out'比较。")
    print(f"\n>> 逐图CSV: {CSVP}\n>> 日志: {LOGP}\n总耗时 {(time.time()-T0)/60:.1f} min")


if __name__ == "__main__":
    main()