"""教师消融 wrapper(评测):eval.py 仅把教师换成 MADViT 的 DeiT,其余 —— 打分 readout
(topk/max/mean/flat)、全局 concat / per-video macro 协议、argparse、results.csv 落盘 —— 全部复用
eval.py,**仅换教师**(只 monkey-patch 教师工厂)。

输入分辨率不需在此指定:eval.py 从 ckpt 的 cfg['img'] 自动读取(训练时由 train_deit.py 存为 224)。
须在仓库根目录运行(import 同目录的 eval.py)。Env: base.
例:  python eval_deit.py --exp da/appearance/20260613_deit_teacher --gpu 0
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))
import eval
from model.teacher_deit import build_deit_teacher

# 唯一改动:替换 eval 模块里被 run_scene() 引用的教师工厂(eval.py 第 62 行 build_appearance_teacher(...))
eval.build_appearance_teacher = build_deit_teacher

if __name__ == '__main__':
    eval.main()
