"""Probe 1 — is there LEARNABLE temporal signal in FROZEN single-frame DINOv3 features?

Trains a small masked-target temporal transformer on NORMAL windows: given W past frames'
patch features (one frozen DINOv3 layer), predict the (masked) next frame's features. Anomaly
score = per-frame top-k cosine PREDICTION error. This is fundamentally different from the
training-free pointwise temporal ops (frame/feature diff, square) that died before: it is
LEARNED and attention-based, so it can model patch CORRESPONDENCE / displacement — "pointwise
uncorrelated" does NOT imply "no learnable temporal structure".

Go/no-go: if crossroads/vehicle beat the appearance ceiling (~60) the frozen single-frame
substrate carries motion -> motion branch can SHARE the DINOv3 backbone (design A). If they sit
near chance, the substrate lacks motion -> justifies a VIDEO teacher (design B). railway is a
control: a STATIC appearance anomaly should NOT light up a motion predictor.

Teacher features are precomputed ONCE per scene (fp16, cached in RAM) so predictor training is
cheap and co-runs with other GPU jobs. Same per-scene / global-concat protocol as the appearance
branch, so the AUCs are directly comparable.

Example:  python probe_motion_v1.py --scenes crossroads,vehicle,railway --gpu 0
"""
import argparse, os, glob, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, roc_curve

from model import build_appearance_teacher

DATA_ROOT = '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly'
SCENES = {
    'bike': 'Bike Roundabout', 'crossroads': 'Crossroads', 'farmland': 'Farmland Inspection',
    'highway': 'Highway', 'railway': 'Railway Inspection', 'solar': 'Solar Panel Inspection',
    'vehicle': 'Vehicle Roundabout',
}
APP_REF = {'crossroads': 59.69, 'vehicle': 60.71, 'railway': 95.91, 'bike': 98.08,
           'highway': 92.69, 'farmland': 99.47, 'solar': 86.12}   # appearance-branch concat topk


def frame_idx(p):
    return int(os.path.splitext(os.path.basename(p))[0])


def eer_of(y, s):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    return float(fpr[np.nanargmin(np.abs(fnr - fpr))])


@torch.no_grad()
def encode_dir(teacher, tf, vid_dir, dev, bs=32):
    """One test/train sub-video -> (n, P, 768) fp16 CPU tensor of the tapped layer, frame order."""
    frames = sorted(glob.glob(os.path.join(vid_dir, '*.jpg')), key=frame_idx)
    out = []
    for i in range(0, len(frames), bs):
        xb = torch.stack([tf(Image.open(f).convert('RGB')) for f in frames[i:i + bs]]).to(dev)
        with torch.autocast('cuda', dtype=torch.float16):
            p = teacher(xb)['patches'][0]                      # (b, P, 768)
        out.append(p.half().cpu())
    return torch.cat(out), len(frames)


class Predictor(nn.Module):
    """Masked-target spatiotemporal transformer: W past frames + 1 mask slot -> predict target."""
    def __init__(self, D=768, d=256, P=256, W=4, depth=4, heads=8):
        super().__init__()
        self.W, self.P, self.d = W, P, d
        self.in_proj = nn.Linear(D, d)
        self.out_proj = nn.Linear(d, D)
        self.spatial = nn.Parameter(torch.zeros(1, 1, P, d))
        self.temporal = nn.Parameter(torch.zeros(1, W + 1, 1, d))
        self.mask = nn.Parameter(torch.zeros(1, 1, P, d))
        for p in (self.spatial, self.temporal, self.mask):
            nn.init.trunc_normal_(p, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, batch_first=True,
                                           norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, depth)

    def forward(self, ctx):                                    # ctx: (B, W, P, D)
        B = ctx.size(0)
        x = self.in_proj(ctx)                                 # (B, W, P, d)
        tgt = self.mask.expand(B, 1, self.P, self.d)
        x = torch.cat([x, tgt], dim=1)                        # (B, W+1, P, d)
        x = x + self.spatial + self.temporal                  # broadcast pos
        x = self.enc(x.reshape(B, (self.W + 1) * self.P, self.d))
        x = x.reshape(B, self.W + 1, self.P, self.d)[:, -1]   # target slot (B, P, d)
        return self.out_proj(x)                               # (B, P, D)


def topk_cos(pred, true, k=5):                                # (B,P,D),(B,P,D) -> (B,)
    d = 1.0 - F.cosine_similarity(pred.float(), true.float(), dim=-1)   # (B, P)
    return d.topk(k, dim=-1).values.mean(-1)


def run_scene(sk, teacher, tf, dev, a):
    scene_dir = os.path.join(DATA_ROOT, SCENES[sk])
    # --- precompute teacher features (cached fp16 CPU) ---
    tr_dirs = sorted(d for d in glob.glob(os.path.join(scene_dir, 'sequence*', 'train', '*')) if os.path.isdir(d))
    te_dirs = sorted(d for d in glob.glob(os.path.join(scene_dir, 'sequence*', 'test', '*')) if os.path.isdir(d))
    train_feats = [encode_dir(teacher, tf, d, dev)[0] for d in tr_dirs]
    test_feats, test_labels = [], []
    for d in te_dirs:
        f, n = encode_dir(teacher, tf, d, dev)
        test_feats.append(f)
        test_labels.append(np.load(d + '.npy', allow_pickle=True).astype(int).ravel()[-n:])
    P = train_feats[0].shape[1]
    W = a.W

    # --- training window index (within-video, no boundary crossing) ---
    idx = [(v, s) for v, f in enumerate(train_feats) for s in range(0, len(f) - W, a.stride)]
    model = Predictor(D=768, d=a.d, P=P, W=W, depth=a.depth, heads=a.heads).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    nstep = (len(idx) + a.bs - 1) // a.bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, steps_per_epoch=nstep, epochs=a.epochs)
    nparam = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'[{sk}] {len(tr_dirs)} train vids ({sum(len(f) for f in train_feats)} frames) '
          f'{len(te_dirs)} test vids | windows={len(idx)} predictor={nparam:.1f}M', flush=True)

    for ep in range(a.epochs):
        model.train(); np.random.shuffle(idx); tot = 0.0; t0 = time.time()
        for bi in range(nstep):
            b = idx[bi * a.bs:(bi + 1) * a.bs]
            ctx = torch.stack([train_feats[v][s:s + W] for v, s in b]).to(dev).float()      # (B,W,P,D)
            true = torch.stack([train_feats[v][s + W] for v, s in b]).to(dev).float()        # (B,P,D)
            loss = topk_cos(model(ctx), true, a.k).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * len(b)
        print(f'  ep{ep+1:02d}/{a.epochs} loss={tot/len(idx):.4f} ({time.time()-t0:.0f}s)', flush=True)

    # --- eval: per-frame prediction error, concat protocol ---
    model.eval(); agg, labels = [], []
    with torch.no_grad():
        for f, lab in zip(test_feats, test_labels):
            n = len(f); sc = np.zeros(n, dtype=np.float32)
            tgts = list(range(W, n))
            for i in range(0, len(tgts), a.bs):
                tb = tgts[i:i + a.bs]
                ctx = torch.stack([f[t - W:t] for t in tb]).to(dev).float()
                true = torch.stack([f[t] for t in tb]).to(dev).float()
                e = topk_cos(model(ctx), true, a.k).cpu().numpy()
                for j, t in enumerate(tb):
                    sc[t] = e[j]
            sc[:W] = sc[W] if n > W else 0.0                  # backfill the first W frames
            agg.append(sc); labels.append(lab[:n])
    y = np.concatenate(labels); s = np.concatenate(agg)
    auc = roc_auc_score(y, s) * 100
    pv = [roc_auc_score(l, a_) * 100 for a_, l in zip(agg, labels) if 0 < l.sum() < len(l)]
    return auc, float(np.mean(pv)), eer_of(y, s), int(y.sum()), len(y), len(te_dirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenes', default='crossroads,vehicle,railway')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--layer', type=int, default=7, help='which frozen DINOv3 block to tap')
    ap.add_argument('--W', type=int, default=4, help='past frames -> predict the next')
    ap.add_argument('--d', type=int, default=256)
    ap.add_argument('--depth', type=int, default=4)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--stride', type=int, default=1)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--wd', type=float, default=5e-2)
    ap.add_argument('--exp', default='20260612_motion_probe1')
    a = ap.parse_args()
    dev = torch.device(f'cuda:{a.gpu}')
    teacher = build_appearance_teacher(layers=[a.layer]).to(dev).eval()
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(teacher.img_mean, teacher.img_std)])

    out = os.path.join('experiments', a.exp); os.makedirs(out, exist_ok=True)
    rows = []
    print(f'=== motion probe1 | layer={a.layer} W={a.W} d={a.d} depth={a.depth} k={a.k} '
          f'ep={a.epochs} bs={a.bs} ===', flush=True)
    for sk in a.scenes.split(','):
        auc, auc_mean, eer, npos, ntot, nvid = run_scene(sk, teacher, tf, dev, a)
        ref = APP_REF.get(sk, float('nan'))
        rows.append((sk, auc, auc_mean, eer, ref, npos, ntot, nvid))
        print(f'>>> [{sk}] motion AUC concat={auc:.2f} mean={auc_mean:.2f} eer={eer:.3f} | '
              f'appearance ref={ref:.2f} | Δ(concat-app)={auc-ref:+.2f}', flush=True)

    with open(os.path.join(out, 'results.csv'), 'w') as fh:
        fh.write('scene,motion_concat_AUC,motion_mean_AUC,motion_EER,appearance_ref_AUC,delta,npos,ntot,nvid\n')
        for sk, auc, aucm, eer, ref, npos, ntot, nvid in rows:
            fh.write(f'{sk},{auc:.2f},{aucm:.2f},{eer:.3f},{ref:.2f},{auc-ref:+.2f},{npos},{ntot},{nvid}\n')
    print(f'\nwrote {out}/results.csv', flush=True)


if __name__ == '__main__':
    main()
