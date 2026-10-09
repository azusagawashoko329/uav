"""诊断/融合工具:导出外观分支(RD-topk)逐帧分数,按 normal vs anom 看量级,并落盘 npz(供融合)。
复用 eval.py 的 score_video(同 headline=topk 读数),只是收集帧分数而非直接算 AUC。零改动已有文件。
--train 时额外导出 train 正常帧分数(融合/百分位标准化的 CDF 锚)。
Env: base(需 HF_HUB_OFFLINE=1 用本地缓存教师)。可从仓库任意位置运行(自带 repo-root 到 sys.path)。
例: HF_HUB_OFFLINE=1 python fusion/dump_appearance_scores.py --scenes crossroads,vehicle --train --gpu 0
"""
import argparse, os, sys, glob
import numpy as np
import torch
import torchvision.transforms as T

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)        # fusion/ 子目录下仍能 import 根目录的 eval / model
import eval as E


def stats(s):
    p = np.percentile(s, [0, 5, 50, 95, 100])
    return (f'n={len(s):5d} mean={s.mean():.4f} std={s.std():.4f} '
            f'min={p[0]:.4f} p5={p[1]:.4f} med={p[2]:.4f} p95={p[3]:.4f} max={p[4]:.4f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', default='20260611_rd_baseline')
    ap.add_argument('--scenes', default='crossroads,vehicle')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--out', default=os.path.join(_ROOT, 'experiments/da/fusion/legacy_score_dump/scores'))
    ap.add_argument('--train', action='store_true', help='额外 dump train 正常帧分数作 CDF 锚(百分位标准化/融合用)')
    a = ap.parse_args()
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    os.makedirs(a.out, exist_ok=True)
    for sk in a.scenes.split(','):
        scene = E.SCENES.get(sk, sk)
        ck = torch.load(os.path.join(_ROOT, 'experiments', a.exp, 'ckpt', f'{sk}.pth'), map_location=dev)
        cfg = ck['cfg']
        use_tap_norm = not cfg.get('no_tap_norm', False)
        teacher = E.build_appearance_teacher(layers=cfg['taps']).to(dev).eval()
        ocbe = E.build_bottleneck(n_taps=len(cfg['taps']), code_dim=cfg['code_dim'], use_tap_norm=use_tap_norm).to(dev).eval()
        student = E.build_student(n_taps=len(cfg['taps']), code_dim=cfg['code_dim'], depth=cfg['depth']).to(dev).eval()
        ocbe.load_state_dict(ck['ocbe']); student.load_state_dict(ck['student'])
        tf = T.Compose([T.Resize((cfg['img'], cfg['img'])), T.ToTensor(),
                        T.Normalize(teacher.img_mean, teacher.img_std)])
        vids = sorted(d for d in glob.glob(os.path.join(E.DATA_ROOT, scene, 'sequence*', 'test', '*')) if os.path.isdir(d))
        S, Y = [], []
        for vp in vids:
            sc, n = E.score_video(teacher, ocbe, student, vp, tf, dev, cfg['k'], a.bs)
            S.append(sc['topk'][:n])
            Y.append(np.asarray(np.load(vp + '.npy', allow_pickle=True)).astype(int).ravel()[-n:])
        S = np.concatenate(S); Y = np.concatenate(Y)
        np.savez(os.path.join(a.out, f'app_{sk}.npz'), scores=S, labels=Y)
        nm, an = S[Y == 0], S[Y == 1]
        print(f'[{sk}] 外观 RD-topk  (落盘 app_{sk}.npz)')
        print(f'   normal {stats(nm)}')
        print(f'   anom   {stats(an)}')
        print(f'   anom/normal mean 比 = {an.mean() / nm.mean():.3f}', flush=True)
        if a.train:
            tvids = sorted(d for d in glob.glob(os.path.join(E.DATA_ROOT, scene, 'sequence*', 'train', '*')) if os.path.isdir(d))
            ST = np.concatenate([E.score_video(teacher, ocbe, student, vp, tf, dev, cfg['k'], a.bs)[0]['topk'] for vp in tvids])
            np.savez(os.path.join(a.out, f'app_{sk}_train.npz'), scores=ST)
            print(f'   train  {stats(ST)}  (CDF锚 → app_{sk}_train.npz)', flush=True)
        print(flush=True)


if __name__ == '__main__':
    main()
