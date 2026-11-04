"""Train/evaluate one UIT-ADrone appearance variant.

The experiment directory's config.json selects the student/OCBE variant.  Data follows
UIT-ADrone's official train/frames and test/frames layout and test_frame_mask labels.
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, math, os, random, time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from model import anomaly_readouts, build_appearance_teacher, build_bottleneck
from model.bottleneck_depthrank_learnable import build_learnable_depthrank2_bottleneck
from model.student import Block, StudentDecoder
from model.student_structural import FourBlockGlobalLocalDecoder

UIT_ROOT_DEFAULT = Path('/data1/data8tb/liangyuxuan/dataset/UIT-ADrone')
DEFAULT_CHECKPOINT_EPOCHS = (10, 20, 30, 40, 45)

# Canonical variant ID -> (label, builder key, expected teacher taps).
# This guards against colloquial names being mapped to the wrong architecture.
VARIANT_CANONICAL = {
    'rd_s12_t4_h512': ('V0', 'uit_12block_dense512', [5, 7, 9, 11]),
    'rd_s4_t12_h512': ('V1', 'uit_4block_plain', list(range(12))),
    'rd_s4_t12_gl768_h512': ('V2', 'uit_4block_gl768', list(range(12))),
    'rd_s4_t12_gl192_h512': ('V3', 'uit_4block_gl192', list(range(12))),
    'rd_s4_t12_gl192_h128': ('V4', 'uit_gl192_dense128', list(range(12))),
    'rd_s4_t12_gl192_h128_qr2': ('V5', 'uit_gl192_dense128_rank2', list(range(12))),
}


class PlainFourBlockDecoder(torch.nn.Module):
    """Historical plain four-block decoder: three heads emitted per stage."""
    def __init__(self, code_dim=64, embed_dim=768, heads=8):
        super().__init__()
        self.proj_in = torch.nn.Linear(code_dim, embed_dim)
        self.blocks = torch.nn.ModuleList([Block(embed_dim, heads=heads, mlp_ratio=4.0) for _ in range(4)])
        self.heads = torch.nn.ModuleList([torch.nn.Linear(embed_dim, embed_dim) for _ in range(12)])
    def forward(self, code):
        x = self.proj_in(code); out = [None] * 12
        for stage, block in enumerate(self.blocks):
            x = block(x)
            group = 3 - stage
            for layer in range(group * 3, group * 3 + 3):
                out[layer] = self.heads[layer](x)
        return out


class Frames(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert('RGB'))


def frame_key(path):
    return int(Path(path).stem)


def manifest(root):
    train_dirs = sorted(p for p in (root / 'train' / 'frames').iterdir() if p.is_dir())
    test_dirs = sorted(p for p in (root / 'test' / 'frames').iterdir() if p.is_dir())
    if not train_dirs or not test_dirs:
        raise FileNotFoundError(f'UIT-ADrone train/test directories missing under {root}')
    train = [p for d in train_dirs for p in sorted(d.glob('*.jpg'), key=frame_key)]
    lines = [f'{p.relative_to(root)}:{p.stat().st_size}' for p in train]
    for d in test_dirs:
        paths = sorted(d.glob('*.jpg'), key=frame_key)
        label = root / 'test' / 'test_frame_mask' / f'{d.name}.npy'
        if not paths or not label.is_file():
            raise FileNotFoundError(f'missing test frames or label: {d}')
        y = np.asarray(np.load(label, allow_pickle=True)).astype(int).ravel()
        if len(paths) != len(y):
            raise ValueError(f'frame/label count mismatch {d.name}: {len(paths)} vs {len(y)}')
        lines.extend(f'{p.relative_to(root)}:{p.stat().st_size}' for p in paths)
        lines.append(f'{label.relative_to(root)}:{hashlib.sha256(label.read_bytes()).hexdigest()}')
    return train, test_dirs, hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def loss_for(taps, recons, k=5, alpha=0.5):
    per = []
    for a, b in zip(taps, recons):
        d = 1 - F.cosine_similarity(a, b, dim=-1)
        per.append(d.topk(k, dim=-1).values.mean(-1))
    # Historical 12-block baseline uses four taps [5,7,9,11] and the original
    # tap-mean top-k loss. The lightweight 4-block variants use all 12 taps,
    # grouped as four adjacent teacher triplets with the group-balanced loss.
    if len(per) == 4:
        return torch.stack(per, dim=1).mean()
    if len(per) != 12:
        raise ValueError(f'expected either 4 historical taps or 12 grouped taps, got {len(per)}')
    groups = torch.stack(per, dim=1).reshape(-1, 4, 3)
    avg, worst = groups.mean(-1), groups.max(-1).values
    return (avg + alpha * (worst - avg)).mean()


def build_variant(cfg, seed):
    v = cfg['variant']
    if v == 'uit_12block_dense512':
        ocbe = build_bottleneck(n_taps=4, code_dim=64, fuse_hidden=512, use_tap_norm=True)
        student = StudentDecoder(n_taps=4, code_dim=64, embed_dim=768, depth=12, heads=8)
    elif v == 'uit_4block_plain':
        ocbe = build_bottleneck(n_taps=12, code_dim=64, fuse_hidden=512, use_tap_norm=True)
        # Absolute reverse alignment gives exactly three heads at each of four stages.
        student = PlainFourBlockDecoder(code_dim=64, embed_dim=768, heads=8)
    elif v == 'uit_4block_gl768':
        ocbe = build_bottleneck(n_taps=12, code_dim=64, fuse_hidden=512, use_tap_norm=True)
        student = FourBlockGlobalLocalDecoder(dim=768, teacher_dim=768)
    elif v == 'uit_4block_gl192':
        ocbe = build_bottleneck(n_taps=12, code_dim=64, fuse_hidden=512, use_tap_norm=True)
        student = FourBlockGlobalLocalDecoder(dim=192, teacher_dim=768)
    elif v == 'uit_gl192_dense128':
        ocbe = build_bottleneck(n_taps=12, code_dim=64, fuse_hidden=128, use_tap_norm=True)
        student = FourBlockGlobalLocalDecoder(dim=192, teacher_dim=768)
    elif v == 'uit_gl192_dense128_rank2':
        ocbe = build_learnable_depthrank2_bottleneck(n_taps=12, code_dim=64, fuse_hidden=128,
                                                     basis_init='random_orthonormal', basis_seed=seed)
        student = FourBlockGlobalLocalDecoder(dim=192, teacher_dim=768)
    else:
        raise ValueError(f'unknown UIT variant: {v}')
    return ocbe, student


def save_checkpoint(output, label, epoch, variant, seed, fingerprint, ocbe, student,
                    optimizer, scheduler, train_loss, trainable_parameters):
    """Save a model-weight snapshot; checkpoints are intentionally local/ignored.

    Optimizer/scheduler objects are not serialized: these snapshots are for stage
    comparison and safe weight loading, not implicit training continuation.
    """
    path = output / 'ckpt' / f'uit_epoch{epoch}.pth' if label == 'epoch' else output / 'ckpt' / 'uit_final.pth'
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format_version': 2, 'dataset': 'UIT-ADrone', 'variant': variant,
        'seed': seed, 'epoch': epoch, 'split_fingerprint': fingerprint,
        'trainable_parameters': trainable_parameters, 'train_loss': train_loss,
        'ocbe_state_dict': {k: v.detach().cpu() for k, v in ocbe.state_dict().items()},
        'student_state_dict': {k: v.detach().cpu() for k, v in student.state_dict().items()},
    }
    torch.save(payload, path)
    return path


def eer_of(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    return float(fpr[np.nanargmin(np.abs(fnr - fpr))])


@torch.no_grad()
def evaluate(teacher, ocbe, student, test_dirs, tf, device, out_csv, batch):
    rows, scores, labels = [], [], []
    for video in test_dirs:
        paths = sorted(video.glob('*.jpg'), key=frame_key)
        y = np.asarray(np.load(str(video.parent.parent / 'test_frame_mask' / f'{video.name}.npy'), allow_pickle=True)).astype(int).ravel()
        loader = DataLoader(Frames(paths, tf), batch_size=batch, shuffle=False, num_workers=4)
        vid_scores = []
        for xb in loader:
            xb = xb.to(device, non_blocking=True)
            taps = teacher(xb)['patches']
            rec = student(ocbe(taps))
            vid_scores.extend(anomaly_readouts(taps, rec, k=5)['topk'].cpu().tolist())
        for path, label, score in zip(paths, y, vid_scores):
            rows.append((video.name, path.name, int(label), float(score)))
        scores.extend(vid_scores); labels.extend(y.tolist())
    auc = roc_auc_score(labels, scores) * 100
    eer = eer_of(labels, scores)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['video','frame','label','anomaly_score']); w.writerows(rows)
    return auc, eer, len(labels), int(sum(labels))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--data-root', type=Path, default=Path(os.environ.get('UAV_UIT_ROOT', str(UIT_ROOT_DEFAULT))))
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--validate-only', action='store_true',
                   help='check variant mapping/taps/params and exit before data/GPU work')
    a = p.parse_args()
    cfg = json.loads((a.output / 'config.json').read_text())
    if cfg['seed'] != a.seed: raise ValueError(f"seed mismatch config={cfg['seed']} cli={a.seed}")
    if a.epochs != int(cfg.get('training', {}).get('epochs', a.epochs)):
        raise ValueError(f"epochs mismatch config={cfg.get('training', {}).get('epochs')} cli={a.epochs}")
    variant_id = cfg.get('variant_id')
    if variant_id is not None:
        if variant_id not in VARIANT_CANONICAL:
            raise ValueError(f'unknown canonical variant_id: {variant_id}')
        label, builder_key, expected_taps = VARIANT_CANONICAL[variant_id]
        if cfg.get('variant') != builder_key:
            raise ValueError(f"variant/builder mismatch: config variant={cfg.get('variant')} "
                             f"canonical {variant_id} -> {builder_key}")
        if cfg.get('variant_label') not in (None, label):
            raise ValueError(f"variant label mismatch: config={cfg.get('variant_label')} canonical={label}")
        cfg_teacher_layers = cfg.get('teacher_layers', list(range(12)))
        if list(cfg_teacher_layers) != expected_taps:
            raise ValueError(f"teacher_taps mismatch for {variant_id}: "
                             f"config={cfg_teacher_layers} canonical={expected_taps}")
    if a.validate_only:
        ocbe, student = build_variant(cfg, a.seed)
        actual = sum(p.numel() for p in ocbe.parameters()) + sum(p.numel() for p in student.parameters())
        expected = cfg.get('resources', {}).get('expected_trainable_parameters')
        if expected is not None and int(expected) != actual:
            raise RuntimeError(f'parameter mismatch: config={expected} actual={actual}')
        print(json.dumps({'validate_only': True, 'id': cfg.get('id'), 'variant_id': variant_id,
                          'variant': cfg.get('variant'), 'variant_label': cfg.get('variant_label'),
                          'teacher_layers': cfg.get('teacher_layers', list(range(12))),
                          'trainable_parameters': actual}, ensure_ascii=False), flush=True)
        return
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    device = torch.device(f'cuda:{a.gpu}')
    train, test_dirs, fingerprint = manifest(a.data_root)
    teacher_layers = cfg.get('teacher_layers', list(range(12)))
    teacher = build_appearance_teacher(layers=teacher_layers).to(device).eval()
    tf = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(teacher.img_mean, teacher.img_std)])
    loader = DataLoader(Frames(train, tf), batch_size=a.batch, shuffle=True,
                        generator=torch.Generator().manual_seed(a.seed), num_workers=a.workers,
                        pin_memory=True, drop_last=True)
    ocbe, student = build_variant(cfg, a.seed); ocbe=ocbe.to(device); student=student.to(device)
    params = list(ocbe.parameters()) + list(student.parameters())
    opt = torch.optim.AdamW(params, lr=5e-4, weight_decay=5e-2)
    steps_per_epoch = min(len(loader), a.max_steps) if a.max_steps else len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-4,
                                                steps_per_epoch=steps_per_epoch, epochs=a.epochs)
    checkpoint_epochs = tuple(int(x) for x in cfg.get('training', {}).get(
        'checkpoint_epochs', DEFAULT_CHECKPOINT_EPOCHS))
    if any(ep < 1 or ep > a.epochs for ep in checkpoint_epochs):
        raise ValueError(f'checkpoint epochs must be within 1..{a.epochs}: {checkpoint_epochs}')
    trainable_parameters = sum(p.numel() for p in params)
    expected = cfg.get('resources', {}).get('expected_trainable_parameters')
    if expected is not None and int(expected) != trainable_parameters:
        raise RuntimeError(f'parameter mismatch: config={expected} actual={trainable_parameters}')
    print(json.dumps({'dataset':'UIT-ADrone','variant':cfg['variant'],'seed':a.seed,
                      'train_videos':len(set(str(Path(x).parent) for x in train)),
                      'train_frames':len(train),'test_videos':len(test_dirs),
                      'test_frames':sum(len(list(d.glob('*.jpg'))) for d in test_dirs),
                      'split_fingerprint':fingerprint,'trainable_parameters':trainable_parameters,
                      'teacher_layers':teacher_layers,'checkpoint_epochs':checkpoint_epochs,
                      'cuda':torch.cuda.get_device_name(device)}, ensure_ascii=False), flush=True)
    t0=time.monotonic()
    for ep in range(a.epochs):
        ocbe.train(); student.train(); total=count=0
        for step, xb in enumerate(loader):
            if a.max_steps and step >= a.max_steps: break
            xb=xb.to(device, non_blocking=True)
            with torch.no_grad(): taps=teacher(xb)['patches']
            rec=student(ocbe(taps)); loss=loss_for(taps, rec)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            total += loss.item()*len(xb); count += len(xb)
        train_loss = total / count
        print(f'epoch={ep+1} loss={train_loss:.6f} elapsed_s={time.monotonic()-t0:.1f}', flush=True)
        if ep + 1 in checkpoint_epochs:
            save_checkpoint(a.output, 'epoch', ep + 1, cfg['variant'], a.seed, fingerprint,
                            ocbe, student, opt, sched, train_loss, trainable_parameters)
    ocbe.eval(); student.eval(); out=a.output/'results'
    auc, eer, n, pos=evaluate(teacher, ocbe, student, test_dirs, tf, device, out/'uit_scores.csv', 32)
    ckpt_path = save_checkpoint(a.output, 'final', a.epochs, cfg['variant'], a.seed,
                                fingerprint, ocbe, student, opt, sched, train_loss,
                                trainable_parameters)
    sha=hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    metric={'dataset':'UIT-ADrone','variant':cfg['variant'],'seed':a.seed,'auc_percent':auc,'eer':eer,
            'frames':n,'positive_frames':pos,'train_frames':len(train),'test_videos':len(test_dirs),
            'split_fingerprint':fingerprint,'trainable_parameters':trainable_parameters,
            'checkpoint':'ckpt/uit_final.pth','checkpoint_sha256':sha,'elapsed_s':time.monotonic()-t0,'max_steps':a.max_steps}
    metric['checkpoint_epochs'] = list(checkpoint_epochs)
    (out/'uit.json').write_text(json.dumps(metric,ensure_ascii=False,indent=2)+'\n');print(json.dumps(metric),flush=True)

if __name__=='__main__': main()
