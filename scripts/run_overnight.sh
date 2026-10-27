#!/bin/bash
# run_overnight.sh —— 夜间串行链，任一环节失败继续下一环节；总时长约 8–10 h
# 用法: cd /root/autodl-tmp/proj_bridge && bash run_overnight.sh
cd /root/autodl-tmp/proj_bridge
TS=$(date +%Y%m%d_%H%M%S)
LOG=results/overnight_${TS}.txt
run () {  # run <名称> <命令...>
  local name=$1; shift
  echo "===== [$(date '+%F %T')] START $name" | tee -a $LOG
  "$@" >> results/overnight_${name}_${TS}.out 2>&1
  echo "===== [$(date '+%F %T')] END   $name  exit=$?" | tee -a $LOG
}

# 1. 冒烟（约 10 min）：三个新脚本先各跑一小份，失败能在日志里一眼看到
run smoke_tilt     python3 c_tilt_simulation.py --limit 3 --thetas 5,10
run smoke_prompt   python3 d_prompt_sweep.py --limit 4

# 2. 倾角仿真全量：C1 几何 + C2 SAM3 + C3 畸变（约 40–60 min；CPU 为主）
run tilt           python3 c_tilt_simulation.py

# 4. lr 1e-5 长程：25 epoch 单 seed，看 Dice 是否继续涨 / 斜率是否继续掉（约 2 h）
run e1_lr1e5_s1234_25ep  python3 e1_collapse_mitigation.py --variants A --lr 1e-5 --tag _lr1e-5 --epochs 25 --seed 1234 --skip_yolo --skip_e0a --skip_e0b

# 5. lr 1e-5 另两个 seed × 10 epoch（约 2 h）
run e1_lr1e5_s2025  python3 e1_collapse_mitigation.py --variants A --lr 1e-5 --tag _lr1e-5 --epochs 10 --seed 2025 --skip_yolo --skip_e0a --skip_e0b
run e1_lr1e5_s7     python3 e1_collapse_mitigation.py --variants A --lr 1e-5 --tag _lr1e-5 --epochs 10 --seed 7    --skip_yolo --skip_e0a --skip_e0b

# 6. 中间学习率 3e-5 × 3 seed × 10 epoch，补全前沿（约 3 h）
run e1_lr3e5_s1234  python3 e1_collapse_mitigation.py --variants A --lr 3e-5 --tag _lr3e-5 --epochs 10 --seed 1234 --skip_yolo --skip_e0a --skip_e0b
run e1_lr3e5_s2025  python3 e1_collapse_mitigation.py --variants A --lr 3e-5 --tag _lr3e-5 --epochs 10 --seed 2025 --skip_yolo --skip_e0a --skip_e0b
run e1_lr3e5_s7     python3 e1_collapse_mitigation.py --variants A --lr 3e-5 --tag _lr3e-5 --epochs 10 --seed 7    --skip_yolo --skip_e0a --skip_e0b

# 7. 提示词扫描（约 3–3.5 h，放最后）
run prompt         python3 d_prompt_sweep.py

echo "===== [$(date '+%F %T')] ALL DONE" | tee -a $LOG
ls -la results/ | tail -40 >> $LOG