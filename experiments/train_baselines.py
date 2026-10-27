"""监督分割基线(U-Net/DeepLabV3+/SegFormer) vs SAM3微调 公平对比
同数据(dataset_400 train320)同评测(三数据集test), Dice+BCE loss, 512输入
基线用ImageNet预训练backbone(对基线公平). 跨域测泛化.
产物: ckpts/base_{arch}_best.pt, results/train_baselines_summary.txt"""
import sys, os, csv, datetime, random, time, glob
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import segmentation_models_pytorch as smp

OUT="/root/autodl-tmp/proj_bridge"
for d in ["ckpts","logs","results"]: os.makedirs(os.path.join(OUT,d),exist_ok=True)
ts=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
class Tee:
    def __init__(s,p): s.f=open(p,"w",encoding="utf-8"); s.o=sys.stdout
    def write(s,x): s.o.write(x); s.f.write(x); s.f.flush()
    def flush(s): s.o.flush(); s.f.flush()
    def close(s): s.f.close()
sumlog=os.path.join(OUT,"results",f"train_baselines_summary_{ts}.txt")
tee=Tee(os.path.join(OUT,"logs",f"train_baselines_{ts}.log")); sys.stdout=tee

DEV="cuda"; SIZE=512; EPOCHS=40; LR=1e-4; BS=8
ROOT="/root/autodl-tmp/dataset_400_clean"

# 三个基线架构(smp)
def build_segformer():
    m = smp.Segformer("mit_b2", encoder_weights=None, in_channels=3, classes=1)  # 不联网
    sd = torch.load("/root/autodl-tmp/mit_b2.pth", map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
    # smp的mit encoder在 m.encoder 下, 尝试直接加载; 若前缀不符自动适配
    enc_keys = set(m.encoder.state_dict().keys())
    # 情况1: 键名直接匹配
    if any(k in enc_keys for k in sd.keys()):
        load_sd = sd
    else:
        # 情况2: 试加/去 encoder. 前缀
        load_sd = {}
        for k,v in sd.items():
            for cand in (k, "encoder."+k, k.replace("encoder.","")):
                if cand in enc_keys: load_sd[cand]=v; break
    missing, unexpected = m.encoder.load_state_dict(load_sd, strict=False)
    n_loaded = len(m.encoder.state_dict()) - len(missing)
    print(f"  SegFormer encoder加载: 成功{n_loaded}/{len(m.encoder.state_dict())} missing={len(missing)} unexpected={len(unexpected)}")
    if n_loaded < len(m.encoder.state_dict())*0.8:
        print("  !! 警告: 加载率<80%, 键名可能没对上, SegFormer结果可能无效")
    return m

BASELINES=[
    ("unet_resnet34",    lambda: smp.Unet("resnet34", encoder_weights="imagenet", in_channels=3, classes=1)),
    ("deeplabv3p_res50", lambda: smp.DeepLabV3Plus("resnet50", encoder_weights="imagenet", in_channels=3, classes=1)),
    ("segformer_mitb2",  build_segformer),
]

def dice_np(a,b):
    a=a.astype(bool); b=b.astype(bool); s=a.sum()+b.sum()
    return 1.0 if s==0 else 2.0*np.logical_and(a,b).sum()/s

# ---- 数据 ----
class CrackDS(Dataset):
    def __init__(self, split):
        self.items=[]
        idir=os.path.join(ROOT,"images",split); mdir=os.path.join(ROOT,"masks",split)
        for f in sorted(os.listdir(idir)):
            if f.endswith(".jpg"):
                self.items.append((os.path.join(idir,f),
                                   os.path.join(mdir,os.path.splitext(f)[0]+".png")))
    def __len__(s): return len(s.items)
    def __getitem__(s,i):
        ip,mp=s.items[i]
        img=Image.open(ip).convert("RGB").resize((SIZE,SIZE))
        m=Image.open(mp).convert("L").resize((SIZE,SIZE),Image.NEAREST)
        x=torch.from_numpy(np.array(img)).float().permute(2,0,1)/255.
        # ImageNet归一化
        mean=torch.tensor([0.485,0.456,0.406]).view(3,1,1); std=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
        x=(x-mean)/std
        y=torch.from_numpy((np.array(m)>127).astype(np.float32))[None]
        return x,y

def dice_bce_loss(logit,target):
    bce=F.binary_cross_entropy_with_logits(logit,target)
    p=torch.sigmoid(logit); inter=(p*target).sum((2,3)); s=p.sum((2,3))+target.sum((2,3))
    dice=1-((2*inter+1)/(s+1)).mean()
    return bce+dice

# ---- 跨域评测集(原图+掩码, 不resize GT, pred resize到GT) ----
def eval_pairs(name):
    if name=="dataset_400":
        d=ROOT; return [(os.path.join(d,"images/test",f),os.path.join(d,"masks/test",os.path.splitext(f)[0]+".png"),127)
                        for f in sorted(os.listdir(os.path.join(d,"images/test"))) if f.endswith(".jpg")]
    if name=="DeepCrack":
        d="/root/autodl-tmp/DeepCrack-master/dataset"; out=[]
        for img in sorted(glob.glob(os.path.join(d,"test_img","*.jpg"))):
            m=os.path.join(d,"test_lab",os.path.splitext(os.path.basename(img))[0]+".png")
            if os.path.exists(m): out.append((img,m,127))
        return out[:60]
    if name=="NHA12D":
        d="/root/autodl-tmp/NHA12D/NHA12D_dataset"; out=[]
        for sub in ["Asphalt_pavement_FD","Asphalt_pavement_VD","Concrete_pavement_FD","Concrete_pavement_VD"]:
            rd=os.path.join(d,sub,"Raw_images"); md=os.path.join(d,sub,"Masks")
            if not os.path.isdir(rd): continue
            for img in sorted(glob.glob(os.path.join(rd,"*.jpg"))):
                m=os.path.join(md,os.path.splitext(os.path.basename(img))[0]+".png")
                if os.path.exists(m): out.append((img,m,0))
        return out[:60]

mean_t=torch.tensor([0.485,0.456,0.406]).view(3,1,1); std_t=torch.tensor([0.229,0.224,0.225]).view(3,1,1)
@torch.no_grad()
def evaluate(model, dsname):
    model.eval(); ds=[]
    for ip,mp,thr in eval_pairs(dsname):
        img=Image.open(ip).convert("RGB"); W0,H0=img.size
        gt=np.array(Image.open(mp).convert("L"))>thr
        x=torch.from_numpy(np.array(img.resize((SIZE,SIZE)))).float().permute(2,0,1)/255.
        x=((x-mean_t)/std_t)[None].to(DEV)
        with torch.autocast(DEV,dtype=torch.bfloat16):
            logit=model(x)
        p=torch.sigmoid(logit.float())[0,0].cpu().numpy()
        pred=np.array(Image.fromarray((p*255).astype(np.uint8)).resize((W0,H0)))>127
        ds.append(dice_np(pred,gt))
    return float(np.mean(ds))

train_loader=DataLoader(CrackDS("train"),batch_size=BS,shuffle=True,num_workers=4,drop_last=True)
print(f"训练集{len(CrackDS('train'))} | 评测: dataset_400/DeepCrack/NHA12D\n")

all_results={}
for arch, build in BASELINES:
    print(f"\n{'='*60}\n训练 {arch}\n{'='*60}")
    model=build().to(DEV)
    opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
    best=-1; best_ep=0
    for ep in range(1,EPOCHS+1):
        model.train(); t0=time.time(); losses=[]
        for x,y in train_loader:
            x,y=x.to(DEV),y.to(DEV)
            with torch.autocast(DEV,dtype=torch.bfloat16):
                logit=model(x); loss=dice_bce_loss(logit,y)
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        if ep%5==0 or ep==EPOCHS:
            din=evaluate(model,"dataset_400")
            print(f"  ep{ep:02d} loss={np.mean(losses):.3f} val_dataset_400={din:.4f} {time.time()-t0:.0f}s")
            if din>best:
                best=din; best_ep=ep
                torch.save(model.state_dict(), os.path.join(OUT,"ckpts",f"base_{arch}_best.pt"))
    # 用best权重跨域评测
    model.load_state_dict(torch.load(os.path.join(OUT,"ckpts",f"base_{arch}_best.pt")))
    r={ds:evaluate(model,ds) for ds in ["dataset_400","DeepCrack","NHA12D"]}
    all_results[arch]=r
    print(f"  [{arch}] 域内={r['dataset_400']:.4f} DeepCrack={r['DeepCrack']:.4f} NHA12D={r['NHA12D']:.4f}")
    del model; torch.cuda.empty_cache()

# 汇总 + 对比SAM3微调
sam3_ft={"dataset_400":0.60,"DeepCrack":0.74,"NHA12D":0.50}
lines=["="*78,"监督分割基线 vs SAM3微调 (同训练数据dataset_400, 跨域泛化对比)","="*78,
       f"{'方法':<22}{'域内(ds400)':>13}{'DeepCrack':>12}{'NHA12D':>10}{'跨域均值':>10}"]
for arch,r in all_results.items():
    cd=(r['DeepCrack']+r['NHA12D'])/2
    lines.append(f"{arch:<22}{r['dataset_400']:>13.4f}{r['DeepCrack']:>12.4f}{r['NHA12D']:>10.4f}{cd:>10.4f}")
cd_sam3=(sam3_ft['DeepCrack']+sam3_ft['NHA12D'])/2
lines.append(f"{'SAM3-ft(文本+框,15M)':<22}{sam3_ft['dataset_400']:>13.4f}{sam3_ft['DeepCrack']:>12.4f}{sam3_ft['NHA12D']:>10.4f}{cd_sam3:>10.4f}")
lines+=["-"*78,
  "[关键] 看'跨域均值': 基线域内可能可比, 但跨域是否显著低于SAM3微调?",
  "  SAM3跨域>基线 -> 文本先验泛化优势成立(核心论点)",
  "  基线跨域≈SAM3 -> 泛化优势论点弱, 退守效率论点(15M vs 全网训练)",
  "="*78]
rep="\n".join(lines); print("\n"+rep)
with open(sumlog,"w",encoding="utf-8") as f: f.write(rep+"\n")
print(f"\n>> 汇总: {sumlog}")
sys.stdout=tee.o; tee.close()