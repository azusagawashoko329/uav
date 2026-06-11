"""Train the reverse-distillation appearance pipeline on ONE Drone-Anomaly SCENE (normal frames).

Protocol (one-class, PER-SCENE — same as MADViT): a scene's train set is ALL of its sequences'
train sub-videos MERGED into one model (e.g. crossroads = sequence1+2+3). One model per scene (7
total). The same scene's merged test split is scored later by eval. Teacher is FROZEN; only OCBE +
student decoder are trained, by minimizing the Topk reconstruction loss on normal frames.

Self-contained (does NOT import the MAD-ViT repro) — only references its frame-loading idea. DINOv3
native resolution is 256 (-> 256 patches); use --img 224 for an apples-to-apples comparison with the
old DeiT pipeline (-> 196 patches). The pipeline is patch-count-agnostic either way.

Example:
  python train.py --scene railway --gpu 0
  python train.py --scene crossroads --epochs 15 --img 256 --gpu 1   # merges sequence1+2+3
"""
import argparse, os, glob, time
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

from model import build_appearance_teacher, build_bottleneck, build_student, topk_loss

DATA_ROOT = '/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly'
SCENES = {
    'bike': 'Bike Roundabout', 'crossroads': 'Crossroads', 'farmland': 'Farmland Inspection',
    'highway': 'Highway', 'railway': 'Railway Inspection', 'solar': 'Solar Panel Inspection',
    'vehicle': 'Vehicle Roundabout',
}


class FrameDataset(Dataset):
    """All normal train frames across ALL sequences + sub-videos of one scene (MADViT per-scene merge)."""
    def __init__(self, scene_dir, img_size, mean, std):
        self.paths = sorted(glob.glob(os.path.join(scene_dir, 'sequence*', 'train', '*', '*.jpg')))
        if not self.paths:
            raise FileNotFoundError(f'no train .jpg under {scene_dir}/sequence*/train/*/')
        self.tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean, std)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('RGB'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True, help='short key (railway/highway/...) or full folder name')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--img', type=int, default=256, help='256 = DINOv3 native (256 patches); 224 -> 196')
    ap.add_argument('--taps', type=int, nargs='+', default=[5, 7, 9, 11])
    ap.add_argument('--code_dim', type=int, default=64)
    ap.add_argument('--depth', type=int, default=12)
    ap.add_argument('--k', type=int, default=5, help='top-k hardest patches in the loss')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=5e-4, help='OneCycleLR max_lr')
    ap.add_argument('--wd', type=float, default=5e-2, help='AdamW weight decay')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max_steps', type=int, default=0, help='>0 limits batches/epoch (smoke test)')
    ap.add_argument('--exp', default='default', help='experiment name -> experiments/<exp>/ckpt/<scene>.pth')
    ap.add_argument('--out', default=None, help='override ckpt dir (default: experiments/<exp>/ckpt)')
    a = ap.parse_args()

    scene = SCENES.get(a.scene, a.scene)
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')

    teacher = build_appearance_teacher(layers=a.taps).to(dev).eval()
    ocbe = build_bottleneck(n_taps=len(a.taps), code_dim=a.code_dim).to(dev)
    student = build_student(n_taps=len(a.taps), code_dim=a.code_dim, depth=a.depth).to(dev)

    scene_dir = os.path.join(DATA_ROOT, scene)
    n_seq = len(glob.glob(os.path.join(scene_dir, 'sequence*')))
    ds = FrameDataset(scene_dir, a.img, teacher.img_mean, teacher.img_std)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                    pin_memory=True, drop_last=True)
    params = list(ocbe.parameters()) + list(student.parameters())
    n_trainable = sum(p.numel() for p in params) / 1e6
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, steps_per_epoch=len(dl), epochs=a.epochs)
    print(f'[{scene}] {n_seq} seq merged, {len(ds)} frames, {len(dl)} batches/ep | img={a.img} '
          f'taps={a.taps} code_dim={a.code_dim} depth={a.depth} k={a.k} | trainable={n_trainable:.1f}M | '
          f'AdamW lr(max)={a.lr} wd={a.wd} OneCycleLR bs={a.bs} ep={a.epochs}', flush=True)

    for ep in range(a.epochs):
        ocbe.train(); student.train()
        tot, n, t0 = 0.0, 0, time.time()
        for bi, x in enumerate(dl):
            if a.max_steps and bi >= a.max_steps:
                break
            x = x.to(dev, non_blocking=True)
            taps = teacher(x)['patches']                 # frozen, no grad
            recon = student(ocbe(taps))
            if ep == 0 and bi == 0:
                print(f'  shape check: patches={tuple(taps[0].shape)} recon={tuple(recon[0].shape)}', flush=True)
            loss = topk_loss(taps, recon, k=a.k)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        print(f'  ep{ep+1:02d}/{a.epochs} loss={tot/max(n,1):.4f} lr={sched.get_last_lr()[0]:.2e}  ({time.time()-t0:.0f}s)', flush=True)

    ckpt_dir = a.out or os.path.join('experiments', a.exp, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, f'{a.scene}.pth')
    torch.save({'ocbe': ocbe.state_dict(), 'student': student.state_dict(), 'cfg': vars(a),
                'scene': scene}, ckpt)
    print(f'saved {ckpt}', flush=True)


if __name__ == '__main__':
    main()
