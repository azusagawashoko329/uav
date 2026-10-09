"""Feasibility probe 2: does packing 3 teacher layers into 1 target destroy the anomaly signal?

probe_group_agg.py showed the plain group mean is almost lossless for the group's *content*
(held-out R^2 0.92-0.98 decoding any member back out of it).  So "fidelity" is not where a fancier
aggregator could win.  The remaining question is the one that actually matters for detection:
**after packing, can a normal frame still be told apart from an anomalous one?**

Training-free proxy (no student, no labels): take normal frames, build a pseudo-anomalous version by
CutPaste (paste a patch from the same image elsewhere — the closest cheap proxy for this dataset's
"a new object appears" anomalies: highway animals, farmland cars, solar sheep, bike bus), then
measure how far the teacher features MOVE, per raw layer vs. per merged group.

  gap   = 1 - cos(f_normal[p], f_pseudo[p]), per patch p
  ratio = gap(merged target) / mean(gap of the 3 members)      <- headline
          ~1.0 => packing keeps the signal;  <<1 => packing washes the anomaly out.

Also reports top-k patch agreement: does the merged target still point at the same patches the
individual layers point at?  (The top-k readout is what carries detection.)

Usage:  python probe_group_gap.py --gpu 1 --scenes railway bike --frames 64
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '../../../../..')))
import argparse
import glob
import os

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import build_appearance_teacher

DATA_ROOT = os.environ.get('UAV_DA_ROOT', '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly')
SCENES = {
    'bike': 'Bike Roundabout', 'crossroads': 'Crossroads', 'farmland': 'Farmland Inspection',
    'highway': 'Highway', 'railway': 'Railway Inspection', 'solar': 'Solar Panel Inspection',
    'vehicle': 'Vehicle Roundabout',
}
GROUPS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]
PATCH = 16                      # ViT-B/16


class Frames(Dataset):
    def __init__(self, scene_dir, img, mean, std, limit):
        paths = sorted(glob.glob(os.path.join(scene_dir, 'sequence*', 'train', '*', '*.jpg')))
        if not paths:
            raise FileNotFoundError(f'no train .jpg under {scene_dir}')
        step = max(1, len(paths) // limit)
        self.paths = paths[::step][:limit]
        self.tf = T.Compose([T.Resize((img, img)), T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('RGB'))


def cutpaste(x, size, gen):
    """x:(B,3,H,W) -> (pseudo, patch-level mask (B,P) marking the pasted region)."""
    b, _, h, w = x.shape
    n = h // PATCH
    out = x.clone()
    mask = torch.zeros(b, n * n, dtype=torch.bool, device=x.device)
    for i in range(b):
        src_y = int(torch.randint(0, h - size + 1, (1,), generator=gen))
        src_x = int(torch.randint(0, w - size + 1, (1,), generator=gen))
        dst_y = int(torch.randint(0, h - size + 1, (1,), generator=gen))
        dst_x = int(torch.randint(0, w - size + 1, (1,), generator=gen))
        out[i, :, dst_y:dst_y + size, dst_x:dst_x + size] = x[i, :, src_y:src_y + size, src_x:src_x + size]
        for r in range(dst_y // PATCH, (dst_y + size - 1) // PATCH + 1):
            for c in range(dst_x // PATCH, (dst_x + size - 1) // PATCH + 1):
                mask[i, r * n + c] = True
    return out, mask


def dist(a, b):
    return 1.0 - torch.cosine_similarity(a, b, dim=-1)          # (B,P)


def topk_overlap(d_ref, d_other, k=5):
    """Jaccard of the k hardest patches per frame."""
    a = d_ref.topk(k, dim=-1).indices
    b = d_other.topk(k, dim=-1).indices
    inter = torch.stack([torch.isin(a[i], b[i]).sum() for i in range(a.shape[0])]).float().mean()
    return (inter / (2 * k - inter)).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--scenes', nargs='+', default=['railway', 'bike'])
    ap.add_argument('--frames', type=int, default=64)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--img', type=int, default=256)
    ap.add_argument('--paste', type=int, default=64, help='CutPaste patch size in pixels')
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    gen = torch.Generator().manual_seed(a.seed)
    teacher = build_appearance_teacher(layers=list(range(12))).to(dev).eval()
    ln = nn.LayerNorm(teacher.embed_dim).to(dev)
    print(f'teacher: 12 blocks | CutPaste {a.paste}px | frames={a.frames}/scene', flush=True)

    for scene_key in a.scenes:
        ds = Frames(os.path.join(DATA_ROOT, SCENES.get(scene_key, scene_key)),
                    a.img, teacher.img_mean, teacher.img_std, a.frames)
        dl = DataLoader(ds, batch_size=a.bs, shuffle=False, num_workers=a.workers, generator=gen)
        agg = {l: {'all': [], 'hit': []} for l in range(12)}
        agg.update({f'g{g}': {'all': [], 'hit': []} for g in range(4)})
        ov = {g: [] for g in range(4)}
        n_frames = 0

        with torch.no_grad():
            for x in dl:
                x = x.to(dev)
                xp, mask = cutpaste(x, a.paste, gen)
                fn = teacher(x)['patches']
                fp = teacher(xp)['patches']
                n_frames += x.shape[0]
                ln_n = [ln(t) for t in fn]
                ln_p = [ln(t) for t in fp]
                d = {l: dist(ln_n[l], ln_p[l]) for l in range(12)}
                for l in range(12):
                    agg[l]['all'].append(d[l].mean().item())
                    agg[l]['hit'].append(d[l][mask].mean().item() if mask.any() else float('nan'))
                for g, idx in enumerate(GROUPS):
                    m_n = torch.stack([ln_n[i] for i in idx]).mean(0)
                    m_p = torch.stack([ln_p[i] for i in idx]).mean(0)
                    dg = dist(m_n, m_p)
                    agg[f'g{g}']['all'].append(dg.mean().item())
                    agg[f'g{g}']['hit'].append(dg[mask].mean().item() if mask.any() else float('nan'))
                    ov[g].append(sum(topk_overlap(dg, d[i]) for i in idx) / 3)
                del fn, fp, ln_n, ln_p

        avg = lambda v: sum(v) / len(v)
        print(f'\n===== scene={scene_key}  frames={n_frames}  (gap = 1-cos(正常,伪异常)) =====')
        print('  层/组            全图gap   粘贴区gap   |  vs组员均值  vs最好组员   top5 patch 重合')
        for g, idx in enumerate(GROUPS):
            mem = [avg(agg[i]['hit']) for i in idx]
            mem_hit = sum(mem) / 3
            gh, ga = avg(agg[f'g{g}']['hit']), avg(agg[f'g{g}']['all'])
            ratio = gh / mem_hit if mem_hit else float('nan')
            best = gh / max(mem) if max(mem) else float('nan')
            print(f'  合并G{g} {idx}   {ga:.5f}   {gh:.5f}   |  {ratio:.3f}      {best:.3f}      '
                  f'jaccard={avg(ov[g]):.3f}')
            for i in idx:
                print(f'    └ L{i:<2}          {avg(agg[i]["all"]):.5f}   {avg(agg[i]["hit"]):.5f}')


if __name__ == '__main__':
    main()
