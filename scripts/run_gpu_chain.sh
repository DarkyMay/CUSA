#!/bin/bash
# run_gpu_chain.sh —— GPU 链：RDD 门控 → HS(620) 数据准备 → HS 训练 → HS 评测(e8 协议) → 关机
# 用法: cd /root/autodl-tmp/proj_bridge && bash run_gpu_chain.sh            （跑完关机）
#       cd /root/autodl-tmp/proj_bridge && bash run_gpu_chain.sh --no-shutdown
cd /root/autodl-tmp/proj_bridge
TS=$(date +%Y%m%d_%H%M%S); LOG=results/gpuchain_${TS}.txt
HSDIR=/root/autodl-tmp/Hybrid-Segmentor/Hybrid_Segmentor_ours620
run () { local name=$1; shift
  echo "===== [$(date '+%F %T')] START $name" | tee -a $LOG
  "$@" >> results/gpuchain_${name}_${TS}.out 2>&1
  echo "===== [$(date '+%F %T')] END   $name  exit=$?" | tee -a $LOG; }

# 0. 依赖
run pip_deps    pip install albumentations torchsummary

# 1. RDD2022 门控验证（R2-4 / R3-6）：全量，factor 扫描 1.5–20，约 3 min
run rdd_full    python3 g_rdd_gate.py

# 2. HS 用我们的 620 张训练（R3-9）；先 import 自检，缺包时立刻可见
run hs_prep     python3 h_hs_prep.py
cat > $HSDIR/train_wrapper.py <<'PY'
# 与 e8 相同的 torchmetrics 垫片：新版本移除了 classification.Dice，HS 的 model.py 顶层会 import 它
import sys, runpy, torch, torchmetrics, torchmetrics.classification as tmc
if not hasattr(tmc, "Dice"):
    class _DiceStub(torch.nn.Module):
        def __init__(self, *a, **k): super().__init__()
        def update(self, *a, **k): pass
        def compute(self): return torch.tensor(0.0)
        def forward(self, *a, **k): return torch.tensor(0.0)
        def reset(self): pass
    tmc.Dice = _DiceStub; print(f"[垫片] torchmetrics {torchmetrics.__version__} 无 classification.Dice，已注入空壳类")
sys.argv = ["trainer.py"]
runpy.run_path("trainer.py", run_name="__main__")
PY
run hs_import   bash -c "cd $HSDIR && python3 -c 'import albumentations, pytorch_lightning, torchsummary; from dataloader import get_loaders; import config; print(\"imports ok\")'"
run hs_train    bash -c "cd $HSDIR && OMP_NUM_THREADS=4 python3 train_wrapper.py"

# 3. 选 val_loss 最低的 ckpt，用 e8 的协议评测四域（whole / tile）
BEST=$(ls $HSDIR/checkpoints/v7_ours620/v7-epoch*val_loss*.ckpt 2>/dev/null | sed -E 's/.*val_loss=?([0-9.]+)\.ckpt/\1 &/' | sort -n | head -1 | cut -d' ' -f2)
echo "best ckpt: $BEST" | tee -a $LOG
if [ -n "$BEST" ]; then
  run hs_eval   bash -c "cd /root/autodl-tmp && python3 -c \"
import sys; sys.argv=['e8']
import e8_hybrid_segmentor_baseline as e8
e8.WEIGHTS=[('hs_ours620','$BEST')]
e8.LEAK['hs_ours620']={'DeepCrack','dataset_400'}
e8.main()
\""
else
  echo "!! 没找到 ckpt，跳过评测" | tee -a $LOG
fi

echo "===== [$(date '+%F %T')] ALL DONE" | tee -a $LOG
ls -lt results/ | head -20 >> $LOG
sync
if [ "$1" != "--no-shutdown" ]; then
  echo "60 秒后关机（Ctrl-C 可取消）" | tee -a $LOG; sleep 60; shutdown
fi