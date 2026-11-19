#!/usr/bin/env python3
"""Dedicated DA LGGD-without-LAGA runner; no historical training entrypoint."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from model.teacher import build_appearance_teacher
from model.lggd_no_laga import VARIANT, ablation_spec, build_ablation_model, ablation_guard, ablation_match

SCENES = {'bike':'Bike Roundabout', 'crossroads':'Crossroads', 'farmland':'Farmland Inspection',
          'highway':'Highway', 'railway':'Railway Inspection', 'solar':'Solar Panel Inspection',
          'vehicle':'Vehicle Roundabout'}
TRAINING = {'epochs':30, 'batch':64, 'workers':8, 'image_size':256, 'max_lr':5e-4,
            'weight_decay':.05, 'optimizer':'AdamW', 'scheduler':'OneCycleLR',
            'data_root':'/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly',
            'deterministic':True, 'drop_last':True}
EVALUATION = {'match':'ablation_match', 'patches':256, 'cosine_channels':768, 'topk':5,
              'alpha':0.0, 'batch':32, 'workers':4, 'score_direction':'larger=more anomalous',
              'label':'sequence*/test/<video>.npy; 1=anomaly',
              'aggregation':'concat within scene; macro mean of 7 scenes'}


def now():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def read_json(path):
    return json.loads(path.read_text())


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save_state(path, value):
    tmp = path.with_name(path.name+'.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def checked_config(exp):
    cfg = read_json(exp/'config.json')
    if (cfg['id'] != exp.name or cfg['variant'] != VARIANT or cfg['dataset'] != 'da'
            or cfg['seed'] not in (42,43,44) or cfg['model'] != ablation_spec()
            or cfg['training'] != TRAINING or cfg['evaluation'] != EVALUATION):
        raise ValueError(f'noncanonical LGGD without LAGA configuration: {exp}')
    baseline = read_json(ROOT/cfg['baseline']/'config.json')
    expected = {**baseline['model'], 'label':'HAP-RD w/o LAGA', 'alpha':0.0}
    if cfg['model'] != expected or cfg['training'] != baseline['training']:
        raise ValueError('baseline controlled variables changed')
    return cfg


def data_manifest(scene_dir):
    training = sorted(scene_dir.glob('sequence*/train/*/*.jpg'))
    videos = sorted(p for p in scene_dir.glob('sequence*/test/*') if p.is_dir())
    if not training or not videos:
        raise FileNotFoundError(scene_dir)
    fingerprint = [f'{p.relative_to(scene_dir)}:{p.stat().st_size}' for p in training]
    for video in videos:
        frames = sorted(video.glob('*.jpg'), key=lambda p:int(p.stem))
        label = Path(str(video)+'.npy')
        labels = np.asarray(np.load(label, allow_pickle=True)).astype(int).ravel()
        if not frames or len(frames) != len(labels) or not set(labels).issubset({0,1}):
            raise ValueError(f'bad video labels: {video}')
        fingerprint.extend(f'{p.relative_to(scene_dir)}:{p.stat().st_size}' for p in frames)
        fingerprint.append(f'{label.relative_to(scene_dir)}:{sha(label)}')
    return training, videos, hashlib.sha256('\n'.join(fingerprint).encode()).hexdigest()


class Frames(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert('RGB'))


@torch.no_grad()
def evaluate(teacher, bottleneck, decoder, videos, transform, device, path, scene_dir):
    ys, scores = [], []
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['video','frame','label','anomaly_score'])
        for video in videos:
            frames = sorted(video.glob('*.jpg'), key=lambda p:int(p.stem))
            labels = np.asarray(np.load(str(video)+'.npy', allow_pickle=True)).astype(int).ravel()
            video_scores = []
            for xb in DataLoader(Frames(frames,transform), batch_size=32, shuffle=False, num_workers=4):
                features = teacher(xb.to(device, non_blocking=True))['patches']
                video_scores.extend(ablation_match(features, decoder(bottleneck(features))).cpu().tolist())
            if len(video_scores) != len(labels) or not np.isfinite(video_scores).all():
                raise ValueError('invalid matched evaluation output')
            writer.writerows((str(video.relative_to(scene_dir)), p.name, int(y), score)
                             for p,y,score in zip(frames,labels,video_scores))
            ys.extend(labels.tolist())
            scores.extend(video_scores)
    tmp.replace(path)
    fpr,tpr,_ = roc_curve(ys,scores)
    return {'auc_percent':float(roc_auc_score(ys,scores)*100),
            'eer':float(fpr[np.nanargmin(np.abs(1-tpr-fpr))]),
            'frames':len(ys), 'positive_frames':sum(ys)}


def main(default_experiment=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment',type=Path,default=default_experiment,required=default_experiment is None)
    parser.add_argument('--scene',choices=SCENES,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    exp = args.experiment.resolve()
    cfg = checked_config(exp)
    audit = read_json(args.preflight)
    if audit['status'] != 'passed':
        raise ValueError('preflight not passed')
    for rel,digest in audit['code_hashes'].items():
        if sha(ROOT/rel) != digest:
            raise ValueError(f'changed pinned code/config: {rel}')
    result = exp/'results'/f'{args.scene}.json'
    checkpoint = exp/'ckpt'/f'{args.scene}_final.pth'
    resume = exp/'ckpt'/f'{args.scene}_resume.pth'
    if result.exists():
        raise FileExistsError('completed result must be skipped by scheduler')
    seed = cfg['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    device = torch.device('cuda:0')
    teacher = build_appearance_teacher(layers=list(range(12))).to(device).eval()
    if any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('teacher must be frozen')
    scene_dir = Path(TRAINING['data_root'])/SCENES[args.scene]
    paths,videos,fingerprint = data_manifest(scene_dir)
    if fingerprint != audit['data'][args.scene]['split_fingerprint']:
        raise ValueError('data split changed')
    transform = T.Compose([T.Resize((256,256)),T.ToTensor(),T.Normalize(teacher.img_mean,teacher.img_std)])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(Frames(paths,transform),batch_size=64,shuffle=True,generator=generator,
                        num_workers=8,pin_memory=True,drop_last=True)
    bottleneck,decoder = build_ablation_model()
    bottleneck,decoder = bottleneck.to(device),decoder.to(device)
    guard = ablation_guard(bottleneck,decoder)
    optimizer = torch.optim.AdamW(list(bottleneck.parameters())+list(decoder.parameters()),lr=5e-4,weight_decay=.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=5e-4,epochs=30,steps_per_epoch=len(loader))
    metadata = {'experiment_id':exp.name,'dataset':'Drone-Anomaly','variant':VARIANT,'seed':seed,'scene':args.scene,
                'epochs':30,'guard':guard,'run_id':args.run_id,'source_commit':args.source_commit,
                'code_hashes':audit['code_hashes'],'baseline':cfg['baseline'],'split_fingerprint':fingerprint,
                'train_frames':len(paths),'test_videos':len(videos),'cuda':torch.cuda.get_device_name(device),
                'match':'ablation_match: top5 patch cosine; alpha0 equal group means; train=batch mean, test=frame score',
                'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda}}
    epoch_start,last_loss = 0,None
    if args.resume and resume.exists():
        state = torch.load(resume,map_location='cpu',weights_only=False)
        for field in ('variant','seed','scene','guard','split_fingerprint','code_hashes','source_commit'):
            if state[field] != metadata[field]:
                raise ValueError(f'resume mismatch: {field}')
        bottleneck.load_state_dict(state['ocbe_state_dict']); decoder.load_state_dict(state['student_state_dict'])
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        for values in optimizer.state.values():
            for key,value in values.items():
                if isinstance(value,torch.Tensor) and key != 'step':
                    values[key]=value.to(device)
        epoch_start,last_loss = state['epoch'],state['train_loss']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        generator.set_state(state['loader_rng'])
        del state
    print(json.dumps({**metadata,'kind':'model_guard','resumed_epoch':epoch_start}),flush=True)
    started = time.monotonic()
    for epoch in range(epoch_start,30):
        bottleneck.train(); decoder.train()
        total,count = 0.,0
        for xb in loader:
            with torch.no_grad():
                features = teacher(xb.to(device,non_blocking=True))['patches']
            loss = ablation_match(features,decoder(bottleneck(features))).mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            optimizer.zero_grad(); loss.backward(); optimizer.step(); scheduler.step()
            total += loss.item()*len(xb); count += len(xb)
        last_loss = total/count
        save_state(resume,{**metadata,'epoch':epoch+1,'train_loss':last_loss,
                          'ocbe_state_dict':bottleneck.state_dict(),'student_state_dict':decoder.state_dict(),
                          'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                          'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                          'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                          'loader_rng':generator.get_state()})
        print(f'epoch={epoch+1} loss={last_loss:.6f} elapsed_s={time.monotonic()-started:.1f}',flush=True)
        save_json(exp/'results'/f'{args.scene}_status.json',
                  {'status':'running','epoch':epoch+1,'run_id':args.run_id,'time':now(),'last_loss':last_loss})
    bottleneck.eval(); decoder.eval()
    save_state(checkpoint,{**metadata,'train_loss':last_loss,
                           'ocbe_state_dict':{k:v.detach().cpu() for k,v in bottleneck.state_dict().items()},
                           'student_state_dict':{k:v.detach().cpu() for k,v in decoder.state_dict().items()}})
    metrics = evaluate(teacher,bottleneck,decoder,videos,transform,device,
                       exp/'results'/f'{args.scene}_scores.csv',scene_dir)
    output = {**metadata,**metrics,'train_loss':last_loss,'checkpoint':str(checkpoint.relative_to(exp)),
              'checkpoint_sha256':sha(checkpoint),'elapsed_s':time.monotonic()-started,'finished_at':now()}
    save_json(result,output)
    save_json(exp/'results'/f'{args.scene}_status.json',
              {'status':'completed','epoch':30,'run_id':args.run_id,'time':now()})
    print(json.dumps(output),flush=True)


if __name__ == '__main__':
    main()
