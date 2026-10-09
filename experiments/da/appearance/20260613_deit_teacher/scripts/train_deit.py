"""教师消融 wrapper(训练):把 RD 管线的教师从 DINOv3 换成 MADViT 的 DeiT
(deit_base_distilled_patch16_224),其余一切 —— OCBE / student / top-k loss / taps / 打分 / 配方 /
per-scene 协议 / argparse / ckpt —— 全部复用 train.py,**仅换教师**(故只 monkey-patch 教师工厂,
不复制 train.py 逻辑,杜绝"复制时手抖改了别处")。

DeiT 用 learned 绝对位置编码,必须 224 输入(196 patches),本 wrapper 强制 --img 224。
ckpt 落到 train.py 的 --exp 目录(建议 --exp da/appearance/20260613_deit_teacher),eval 时用 eval_deit.py。
Env: base.  例:  python train_deit.py --scene railway --gpu 0 --exp da/appearance/20260613_deit_teacher
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))
import train
from model.teacher_deit import build_deit_teacher

# 唯一改动:替换 train 模块里被 main() 引用的教师工厂(train.py 第 73 行 build_appearance_teacher(...))
train.build_appearance_teacher = build_deit_teacher

if __name__ == '__main__':
    argv = sys.argv[1:]
    if '--img' in argv:
        i = argv.index('--img')
        if i + 1 < len(argv) and argv[i + 1] != '224':
            print(f'[deit] DeiT learned pos-embed 固定 224,强制 --img 224(忽略传入的 {argv[i + 1]})', flush=True)
            argv[i + 1] = '224'
    else:
        argv += ['--img', '224']
    sys.argv = [sys.argv[0]] + argv
    train.main()
