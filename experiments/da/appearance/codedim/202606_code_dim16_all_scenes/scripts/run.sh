#!/bin/bash
# 全 7 场景 code_dim=16 训练+eval,与 64(20260611_rd_baseline)对比,判断是否该把 baseline 从 64 下调。
# 其余配方与 baseline 完全一致、ep15、同 seed。双卡静态均衡(按 baseline 实测耗时:crossroads41/bike20 一卡,farmland32/highway30/vehicle3/railway1 另一卡)。
cd "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)" || exit 1
PY="${PYTHON:-python}"
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p experiments/da/appearance/codedim/202606_code_dim16_all_scenes/ckpt
cp -n experiments/da/appearance/codedim/cd16/ckpt/solar.pth experiments/da/appearance/codedim/202606_code_dim16_all_scenes/ckpt/solar.pth   # solar 已扫过,复用

T(){ "$PY" train.py --scene "$1" --code_dim 16 --exp da/appearance/codedim/202606_code_dim16_all_scenes --gpu "$2" --epochs 15; }
( T crossroads 0; T bike 0 ) &
( T farmland 1; T highway 1; T vehicle 1; T railway 1 ) &
wait

"$PY" eval.py --exp da/appearance/codedim/202606_code_dim16_all_scenes --scenes bike,crossroads,farmland,highway,railway,solar,vehicle --gpu 0
echo "=== code_dim=16 全场景 DONE ==="
cat experiments/da/appearance/codedim/202606_code_dim16_all_scenes/results.csv
