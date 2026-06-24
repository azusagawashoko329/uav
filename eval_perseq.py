"""外观分支 per-seq(逐子视频)AUC —— 复用 eval.py 的 score_video/build,不改它。
对每个场景加载 experiments/<exp>/ckpt/<scene>.pth,逐 test 子视频单独算 AUC(topk/max/mean 读出),
写 experiments/<exp>/results_perseq.csv。用于 note/ 的逐 seq 对比表(与 results.csv 的 concat 同权重、可对账)。
Env: base/DMAD(torch+timm)。例: HF_HUB_OFFLINE=1 python eval_perseq.py --exp 20260611_rd_baseline --gpu 0
"""
import argparse, os, glob
import numpy as np
import torch
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score

import eval as E
from model import build_appearance_teacher, build_bottleneck, build_student


def run_scene(scene_key, exp, dev, bs):
    scene = E.SCENES.get(scene_key, scene_key)
    ckpt = os.path.join('experiments', exp, 'ckpt', f'{scene_key}.pth')
    if not os.path.exists(ckpt):
        return []
    ck = torch.load(ckpt, map_location=dev); cfg = ck['cfg']
    taps, code_dim, depth, img, k = cfg['taps'], cfg['code_dim'], cfg['depth'], cfg['img'], cfg['k']
    teacher = build_appearance_teacher(layers=taps).to(dev).eval()
    ocbe = build_bottleneck(n_taps=len(taps), code_dim=code_dim,
                            use_tap_norm=not cfg.get('no_tap_norm', False)).to(dev).eval()
    student = build_student(n_taps=len(taps), code_dim=code_dim, depth=depth).to(dev).eval()
    ocbe.load_state_dict(ck['ocbe']); student.load_state_dict(ck['student'])
    tf = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(teacher.img_mean, teacher.img_std)])

    vids = sorted(d for d in glob.glob(os.path.join(E.DATA_ROOT, scene, 'sequence*', 'test', '*')) if os.path.isdir(d))
    rows = []
    for vp in vids:
        sc, n = E.score_video(teacher, ocbe, student, vp, tf, dev, k, bs)
        lab = np.asarray(np.load(vp + '.npy', allow_pickle=True)).astype(int).ravel()[-n:]
        seq = os.path.basename(os.path.dirname(os.path.dirname(vp)))     # sequenceX
        sub = os.path.basename(vp)
        if 0 < lab.sum() < len(lab):
            a = {r: roc_auc_score(lab, sc[r][:n]) * 100 for r in ('topk', 'max', 'mean')}
        else:
            a = {r: float('nan') for r in ('topk', 'max', 'mean')}      # 单类子视频 AUC 无定义
        rows.append((scene_key, seq, sub, n, int(lab.sum()), a['topk'], a['max'], a['mean']))
        print(f'  {scene_key} {seq}-{sub} frames={n} anom={int(lab.sum())} '
              f'topk={a["topk"]:.2f} max={a["max"]:.2f} mean={a["mean"]:.2f}', flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='20260611_rd_baseline')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--scenes', default=E.ALL_SCENES)
    a = ap.parse_args()
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    all_rows = []
    for sc in a.scenes.split(','):
        all_rows += run_scene(sc, a.exp, dev, a.bs)
    outp = os.path.join('experiments', a.exp, 'results_perseq.csv')
    with open(outp, 'w') as f:
        f.write('scene,sequence,subvideo,frames,anom,topk_AUC,max_AUC,mean_AUC\n')
        for r in all_rows:
            f.write(f'{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.2f},{r[6]:.2f},{r[7]:.2f}\n')
    print(f'\nwrote {outp} ({len(all_rows)} sub-videos)', flush=True)


if __name__ == '__main__':
    main()
