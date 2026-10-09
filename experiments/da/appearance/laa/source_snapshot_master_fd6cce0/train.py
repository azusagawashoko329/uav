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
import argparse, os, glob, time, random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from model import (build_appearance_teacher, build_bottleneck, build_student, build_adapter,
                   topk_loss, full_loss, per_layer_topk_dist, adapter_objective)

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


def perturb(x, frac=0.5, patch=16, max_blocks=2):
    """Gentle input-image pseudo-anomaly for the LAA layer-selection signal (TRAIN ONLY; no test contact).

    On the already-normalized batch (B,3,H,W): for a random `frac` of samples, mildly corrupt 1..max_blocks
    patch-sized (patch x patch) blocks by one of: light Gaussian BLUR, paint WHITE, paint BLACK, or SWAP two
    blocks' contents within the same frame. Deliberately small and local (no foreign paste, no heavy noise)
    so the per-layer gap reflects genuine local sensitivity rather than gross corruption. White/black use
    moderate normalized-space constants (+/-2.0 ~ near-white/black, not fully saturated). Returns a copy.

    The mode mix spans levels so the per-layer gap isn't biased to one depth: blur->texture(shallow),
    white/black->local contrast, swap->structure(deeper)."""
    x = x.clone()
    B, C, H, W = x.shape
    gh, gw = H // patch, W // patch
    if gh < 2 or gw < 2:
        return x
    for i in torch.randperm(B, device=x.device)[:max(1, int(round(B * frac)))].tolist():
        mode = random.choice(['blur', 'white', 'black', 'swap'])
        nb = random.randint(1, max_blocks)
        if mode == 'swap':
            for _ in range(nb):
                ay = random.randint(0, gh - 1) * patch; ax = random.randint(0, gw - 1) * patch
                by = random.randint(0, gh - 1) * patch; bx = random.randint(0, gw - 1) * patch
                a = x[i, :, ay:ay + patch, ax:ax + patch].clone()
                x[i, :, ay:ay + patch, ax:ax + patch] = x[i, :, by:by + patch, bx:bx + patch]
                x[i, :, by:by + patch, bx:bx + patch] = a
        else:
            for _ in range(nb):
                py = random.randint(0, gh - 1) * patch; px = random.randint(0, gw - 1) * patch
                sl = (slice(None), slice(py, py + patch), slice(px, px + patch))
                if mode == 'blur':
                    x[i][sl] = TF.gaussian_blur(x[i][sl].unsqueeze(0), kernel_size=5, sigma=2.0)[0]
                elif mode == 'white':
                    x[i][sl] = 2.0
                else:                                                 # black
                    x[i][sl] = -2.0
    return x


def _feather_mask(s, dev):
    """Center=1, edge->0 circular feather weight (1,s,s) for soft-boundary blend (~seamless, no hard edge)."""
    r = torch.linspace(-1, 1, s, device=dev)
    yy, xx = torch.meshgrid(r, r, indexing='ij')
    d = torch.sqrt(yy ** 2 + xx ** 2).clamp(max=1.0)
    return (1.0 - d).unsqueeze(0)


def perturb_v2(x, frac=0.5, max_blocks=2):
    """Upgraded pseudo-anomaly generator (v2). On normalized batch (B,3,H,W), for a random `frac` of samples,
    1..max_blocks MULTI-SCALE blocks (8..48px) corrupted mainly by SELF-IMAGE CutPaste (cut a block, paste
    elsewhere = local inconsistency = robust OOD, not reliant on rare objects), plus a little noise/color OOD,
    blended with a SOFT FEATHER boundary (~seamless, avoids the 'find-the-box-edge' shortcut). Aim: out-of-train
    local representations + activate shallow->semantic layer spectrum. Quality measured by g/t_pseudo (note/LAA.md)."""
    x = x.clone(); B, C, H, W = x.shape; dev = x.device
    smax = max(8, min(H, W) // 4)
    for i in torch.randperm(B, device=dev)[:max(1, int(round(B * frac)))].tolist():
        for _ in range(random.randint(1, max_blocks)):
            s = min(random.choice([8, 16, 24, 32, 48]), smax)
            ty = random.randint(0, H - s); tx = random.randint(0, W - s)
            mode = random.choices(['cutpaste', 'noise', 'black', 'white'], weights=[0.6, 0.2, 0.1, 0.1])[0]
            if mode == 'cutpaste':
                sy = random.randint(0, H - s); sx = random.randint(0, W - s)
                src = x[i, :, sy:sy + s, sx:sx + s].clone()
                if random.random() < 0.5:
                    src = torch.flip(src, dims=[random.choice([1, 2])])      # flip for extra novelty
            elif mode == 'noise':
                src = x[i, :, ty:ty + s, tx:tx + s] + torch.randn(C, s, s, device=dev) * 0.8
            elif mode == 'black':
                src = torch.full((C, s, s), -2.0, device=dev)
            else:
                src = torch.full((C, s, s), 2.0, device=dev)
            m = _feather_mask(s, dev); tgt = x[i, :, ty:ty + s, tx:tx + s]
            x[i, :, ty:ty + s, tx:tx + s] = m * src + (1.0 - m) * tgt
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', required=True, help='short key (railway/highway/...) or full folder name')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--img', type=int, default=256, help='256 = DINOv3 native (256 patches); 224 -> 196')
    ap.add_argument('--taps', type=int, nargs='+', default=[5, 7, 9, 11])
    ap.add_argument('--code_dim', type=int, default=64)
    ap.add_argument('--no_tap_norm', action='store_true',
                    help='ablation: concat RAW taps (no per-tap LayerNorm) before the OCBE squeeze')
    ap.add_argument('--depth', type=int, default=12)
    ap.add_argument('--k', type=int, default=5, help='top-k hardest patches in the loss')
    ap.add_argument('--loss', default='topk', choices=['topk', 'full'],
                    help="topk = top-k hardest patches (default); full = ablation, mean over ALL patches (MADViT-style)")
    # --- Layer-Attention Adapter (LAA) path (default OFF = current fixed-tap method, untouched) ---
    ap.add_argument('--tap_adapter', action='store_true',
                    help='learn a per-layer weighting over ALL 12 teacher blocks (adapter) instead of fixed taps; '
                         'student reconstructs all 12; top-k unchanged, layers combined by learned w')
    ap.add_argument('--pseudo', action='store_true',
                    help='train the adapter weights by the input-image pseudo-anomaly gap (requires --tap_adapter)')
    ap.add_argument('--adapter_input', action='store_true',
                    help='variant A: also fold w into the OCBE input (sum_l w_l*LN_l(A_l), OCBE n_taps=1); '
                         'default (off) = variant B, OCBE input = all-12 concat, w only weights loss/score')
    ap.add_argument('--perturb_frac', type=float, default=0.5, help='fraction of batch perturbed for --pseudo')
    ap.add_argument('--tau', type=float, default=1.0, help='softmax temperature mapping per-layer gap -> target w')
    ap.add_argument('--lam_sel', type=float, default=1.0, help='weight of the selection (KL) term')
    ap.add_argument('--lam_ent', type=float, default=0.01, help='weight of the entropy term (anti-collapse on w)')
    ap.add_argument('--gen', default='v1', choices=['v1', 'v2'],
                    help="pseudo-anomaly generator: v1=pixel corrupt (default, zero-regression); "
                         "v2=multi-scale + self-image CutPaste + soft boundary")
    ap.add_argument('--gap_zscore', action='store_true',
                    help='TODO3: z-score the per-layer gap g before softmax -> push w non-uniform (default off)')
    ap.add_argument('--adapter_warmup', type=int, default=0,
                    help='staged: first N epochs pure alltap (student learns first), enable adapter at ep>=N (default 0=all-along)')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=5e-4, help='OneCycleLR max_lr')
    ap.add_argument('--wd', type=float, default=5e-2, help='AdamW weight decay')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max_steps', type=int, default=0, help='>0 limits batches/epoch (smoke test)')
    ap.add_argument('--base', default='experiments', help='experiment root dir (e.g. experiment_LAA) -> <base>/<exp>/ckpt')
    ap.add_argument('--exp', default='default', help='experiment name -> <base>/<exp>/ckpt/<scene>.pth')
    ap.add_argument('--out', default=None, help='override ckpt dir (default: <base>/<exp>/ckpt)')
    a = ap.parse_args()

    scene = SCENES.get(a.scene, a.scene)
    dev = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')

    # LAA path: tap ALL 12 blocks, student reconstructs all 12, adapter's w re-weights per-layer loss/score.
    #   variant B (default): OCBE input = all-12 concat (n_taps=12).
    #   variant A (--adapter_input): OCBE input = w-aggregated single view (n_taps=1).
    # Default (no --tap_adapter) is untouched: fixed taps, OCBE/student n_taps=len(taps).
    agg = a.tap_adapter and a.adapter_input
    teacher_layers = list(range(12)) if a.tap_adapter else a.taps
    n_recon = 12 if a.tap_adapter else len(a.taps)
    ocbe_taps = 1 if agg else n_recon
    teacher = build_appearance_teacher(layers=teacher_layers).to(dev).eval()
    adapter = build_adapter(n_layers=12, embed_dim=teacher.embed_dim, input_agg=agg).to(dev) if a.tap_adapter else None
    ocbe = build_bottleneck(n_taps=ocbe_taps, code_dim=a.code_dim, use_tap_norm=not a.no_tap_norm).to(dev)
    student = build_student(n_taps=n_recon, code_dim=a.code_dim, depth=a.depth).to(dev)

    scene_dir = os.path.join(DATA_ROOT, scene)
    n_seq = len(glob.glob(os.path.join(scene_dir, 'sequence*')))
    ds = FrameDataset(scene_dir, a.img, teacher.img_mean, teacher.img_std)
    dl = DataLoader(ds, batch_size=a.bs, shuffle=True, num_workers=a.workers,
                    pin_memory=True, drop_last=True)
    params = list(ocbe.parameters()) + list(student.parameters())
    if adapter is not None:
        params += list(adapter.parameters())
    n_trainable = sum(p.numel() for p in params) / 1e6
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, steps_per_epoch=len(dl), epochs=a.epochs)
    mode = (f'LAA-{"A(w-agg input, OCBE n_taps=1)" if agg else "B(all-12 concat)"} -> {n_recon} recon, '
            f'pseudo={a.pseudo}') if a.tap_adapter else f'taps={a.taps}'
    print(f'[{scene}] {n_seq} seq merged, {len(ds)} frames, {len(dl)} batches/ep | img={a.img} '
          f'{mode} code_dim={a.code_dim} depth={a.depth} k={a.k} loss={a.loss} tap_norm={not a.no_tap_norm} | '
          f'trainable={n_trainable:.1f}M | AdamW lr(max)={a.lr} wd={a.wd} OneCycleLR bs={a.bs} ep={a.epochs}', flush=True)

    for ep in range(a.epochs):
        ocbe.train(); student.train()
        if adapter is not None:
            adapter.train()
        tot, n, t0 = 0.0, 0, time.time()
        for bi, x in enumerate(dl):
            if a.max_steps and bi >= a.max_steps:
                break
            x = x.to(dev, non_blocking=True)
            if a.tap_adapter and ep >= a.adapter_warmup:             # staged: warmup epochs (ep<N) fall to alltap below
                taps = teacher(x)['patches']                          # 12 frozen taps, no grad
                code = ocbe([adapter.aggregate(taps)]) if agg else ocbe(taps)   # A: w-agg input | B: concat
                recon = student(code)                                 # -> 12 reconstructions
                t_norm = per_layer_topk_dist(taps, recon, k=a.k)      # (B, 12)
                t_pseudo = None
                if a.pseudo:
                    gen_fn = perturb_v2 if a.gen == 'v2' else perturb
                    tp = teacher(gen_fn(x, a.perturb_frac))['patches']
                    code_p = ocbe([adapter.aggregate(tp)]) if agg else ocbe(tp)
                    t_pseudo = per_layer_topk_dist(tp, student(code_p), k=a.k)
                loss, logs = adapter_objective(adapter.w, t_norm, t_pseudo, tau=a.tau, lam_sel=a.lam_sel,
                                               lam_ent=a.lam_ent, gap_zscore=a.gap_zscore)
            else:
                taps = teacher(x)['patches']                          # frozen, no grad
                recon = student(ocbe(taps))
                loss = topk_loss(taps, recon, k=a.k) if a.loss == 'topk' else full_loss(taps, recon)
            if ep == 0 and bi == 0:
                print(f'  shape check: patches={tuple(taps[0].shape)} recon={tuple(recon[0].shape)} '
                      f'n_recon={len(recon)} loss={a.loss}{" (LAA)" if a.tap_adapter else ""}', flush=True)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item() * x.size(0); n += x.size(0)
        msg = f'  ep{ep+1:02d}/{a.epochs} loss={tot/max(n,1):.4f} lr={sched.get_last_lr()[0]:.2e}  ({time.time()-t0:.0f}s)'
        if adapter is not None:
            top = torch.topk(adapter.w.detach(), 4)
            msg += '  w_top4=' + ','.join(f'L{int(i)}:{float(v):.2f}' for v, i in zip(top.values, top.indices))
        print(msg, flush=True)

    ckpt_dir = a.out or os.path.join(a.base, a.exp, 'ckpt')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, f'{a.scene}.pth')
    blob = {'ocbe': ocbe.state_dict(), 'student': student.state_dict(), 'cfg': vars(a), 'scene': scene}
    if adapter is not None:
        blob['adapter'] = adapter.state_dict()
    torch.save(blob, ckpt)
    print(f'saved {ckpt}', flush=True)


if __name__ == '__main__':
    main()
