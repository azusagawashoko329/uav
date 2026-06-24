#!/bin/bash
# OCBE code_dim sweet-spot 扫描:solar 单场景(小、~10min/次、信号中等最敏感),验证"太窄/太宽都降"。
# baseline code_dim=64 已有 86.12(20260611_rd_baseline),这里补其余点。双卡分配。
cd /data1/data8tb/liangyuxuan/code/uav
PY=/home/lyx/miniconda3/envs/DMAD/bin/python
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

run() {  # $1=code_dim  $2=gpu
  $PY train.py --scene solar --code_dim "$1" --exp "codedim/cd$1" --gpu "$2" --epochs 15
  $PY eval.py  --exp "codedim/cd$1" --scenes solar --gpu "$2"
}

( for cd in 8 32 256; do run $cd 0; done ) &
( for cd in 16 128;   do run $cd 1; done ) &
wait
echo "=== code_dim sweep DONE ==="
for cd in 8 16 32 128 256; do
  echo -n "code_dim=$cd  "; cat experiments/codedim/cd$cd/results.csv 2>/dev/null | grep solar || echo "(missing)"
done
