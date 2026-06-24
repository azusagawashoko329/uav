#!/bin/bash
# 全 7 场景 code_dim=16 训练+eval,与 64(20260611_rd_baseline)对比,判断是否该把 baseline 从 64 下调。
# 其余配方与 baseline 完全一致、ep15、同 seed。双卡静态均衡(按 baseline 实测耗时:crossroads41/bike20 一卡,farmland32/highway30/vehicle3/railway1 另一卡)。
cd /data1/data8tb/liangyuxuan/code/uav
PY=/home/lyx/miniconda3/envs/DMAD/bin/python
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p experiments/codedim16/ckpt
cp -n experiments/codedim/cd16/ckpt/solar.pth experiments/codedim16/ckpt/solar.pth   # solar 已扫过,复用

T(){ $PY train.py --scene "$1" --code_dim 16 --exp codedim16 --gpu "$2" --epochs 15; }
( T crossroads 0; T bike 0 ) &
( T farmland 1; T highway 1; T vehicle 1; T railway 1 ) &
wait

$PY eval.py --exp codedim16 --scenes bike,crossroads,farmland,highway,railway,solar,vehicle --gpu 0
echo "=== code_dim=16 全场景 DONE ==="
cat experiments/codedim16/results.csv
