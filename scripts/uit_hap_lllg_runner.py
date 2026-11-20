#!/usr/bin/env python3
"""Fresh whole-UIT HAP-RD/Lite training with independent LLLG and PLDD code."""
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score,roc_curve
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
from model.teacher import build_appearance_teacher
from model.hap_lllg import model_spec,build_hap_lllg,actual_guard,pldd_match

DATA_ROOT=Path('/data1/data8tb/liangyuxuan/dataset/UIT-ADrone')
TRAINING={'epochs':45,'batch':64,'workers':8,'image_size':256,'max_lr':5e-4,'weight_decay':.05,
          'optimizer':'AdamW','scheduler':'OneCycleLR','deterministic':True,'drop_last':True,
          'data_root':str(DATA_ROOT),'checkpoint_epochs':[10,20,30,40,45]}
EVALUATION={'match':'pldd_match','topk':5,'alpha':.5,'batch':32,'workers':4,
            'score_direction':'larger=more anomalous','aggregation':'concat all 51 test videos; one global AUC/EER',
            'labels':'test/test_frame_mask/<video>.npy; 1=anomaly','official_overlap_retained':True}


def now():
    return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def read(path):
    return json.loads(path.read_text())


def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def state_hash(module):
    h=hashlib.sha256()
    for name,tensor in module.state_dict().items():
        h.update(name.encode());h.update(str(tuple(tensor.shape)).encode());h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_state(path,value):
    tmp=path.with_name(path.name+'.tmp')
    torch.save(value,tmp)
    tmp.replace(path)


def checked_config(exp):
    cfg=read(exp/'config.json')
    if (cfg['id']!=exp.name or cfg['dataset']!='uit' or cfg['seed'] not in [42,43,44]
            or cfg['model']!=model_spec(cfg['variant']) or cfg['training']!=TRAINING or cfg['evaluation']!=EVALUATION):
        raise ValueError(f'wrong formal UIT LLLG configuration: {exp}')
    return cfg


def data_manifest():
    train_dirs=sorted(p for p in (DATA_ROOT/'train/frames').iterdir() if p.is_dir())
    videos=sorted(p for p in (DATA_ROOT/'test/frames').iterdir() if p.is_dir())
    training=[p for d in train_dirs for p in sorted(d.glob('*.jpg'),key=lambda p:int(p.stem))]
    lines=[f'{p.relative_to(DATA_ROOT)}:{p.stat().st_size}' for p in training]
    test_frames=positives=0
    for video in videos:
        frames=sorted(video.glob('*.jpg'),key=lambda p:int(p.stem))
        label=DATA_ROOT/'test/test_frame_mask'/f'{video.name}.npy'
        labels=np.asarray(np.load(label,allow_pickle=True)).astype(int).ravel()
        if not frames or len(frames)!=len(labels) or not set(labels).issubset({0,1}):
            raise ValueError(f'bad test data or labels: {video}')
        lines.extend(f'{p.relative_to(DATA_ROOT)}:{p.stat().st_size}' for p in frames)
        lines.append(f'{label.relative_to(DATA_ROOT)}:{sha(label)}')
        test_frames+=len(frames);positives+=int(labels.sum())
    if (len(train_dirs),len(training),len(videos),test_frames,positives)!=(41,59186,51,147005,63485):
        raise ValueError('UIT official split counts changed')
    fingerprint=hashlib.sha256('\n'.join(lines).encode()).hexdigest()
    return training,videos,{'train_videos':41,'train_frames':59186,'test_videos':51,'test_frames':147005,
                            'positive_frames':63485,'split_fingerprint':fingerprint,
                            'official_overlap_retained':True}


class Frames(Dataset):
    def __init__(self,paths,transform):
        self.paths,self.transform=paths,transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,index):
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert('RGB'))


@torch.no_grad()
def evaluate(teacher,bn,decoder,videos,transform,device,exp):
    ys,scores=[],[]
    out=exp/'results/matched_scores.csv'
    tmp=out.with_name(out.name+'.tmp')
    with tmp.open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['video','frame','label','anomaly_score'])
        for index,video in enumerate(videos):
            frames=sorted(video.glob('*.jpg'),key=lambda p:int(p.stem))
            labels=np.asarray(np.load(DATA_ROOT/'test/test_frame_mask'/f'{video.name}.npy',allow_pickle=True)).astype(int).ravel()
            video_scores=[]
            for xb in DataLoader(Frames(frames,transform),batch_size=32,shuffle=False,num_workers=4):
                taps=teacher(xb.to(device,non_blocking=True))['patches']
                video_scores.extend(pldd_match(taps,decoder(bn(taps))).cpu().tolist())
            if len(video_scores)!=len(labels) or not np.isfinite(video_scores).all():
                raise ValueError('invalid per-frame match output')
            writer.writerows((video.name,p.name,int(y),s) for p,y,s in zip(frames,labels,video_scores))
            ys.extend(labels.tolist());scores.extend(video_scores)
            save_json(exp/'results/eval_status.json',{'status':'running','completed_videos':index+1,
                      'total_videos':len(videos),'frames':len(ys),'time':now()})
            print(f'eval_video={index+1}/{len(videos)} frames={len(ys)}',flush=True)
    tmp.replace(out)
    fpr,tpr,_=roc_curve(ys,scores)
    return {'auc_percent':float(roc_auc_score(ys,scores)*100),
            'eer':float(fpr[np.nanargmin(np.abs(1-tpr-fpr))]),'frames':len(ys),'positive_frames':sum(ys)}


def main(default_experiment=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--experiment',type=Path,default=default_experiment,required=default_experiment is None)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    exp=args.experiment.resolve();cfg=checked_config(exp);audit=read(args.preflight)
    if audit['status']!='passed':
        raise ValueError('preflight has not passed')
    for rel,digest in audit['code_hashes'].items():
        if sha(ROOT/rel)!=digest:
            raise ValueError(f'code/config changed after preflight: {rel}')
    if (exp/'results/uit.json').exists():
        raise FileExistsError('completed experiment must be skipped by scheduler')
    seed=cfg['seed'];variant=cfg['variant']
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0')
    paths,videos,data=data_manifest()
    if data!=audit['data']:
        raise ValueError('UIT split/fingerprint changed')
    teacher=build_appearance_teacher(layers=list(range(12))).eval()
    if state_hash(teacher)!=audit['teacher']['state_sha256'] or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError('frozen teacher weights changed')
    teacher=teacher.to(device)
    transform=T.Compose([T.Resize((256,256)),T.ToTensor(),T.Normalize(teacher.img_mean,teacher.img_std)])
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(Frames(paths,transform),batch_size=64,shuffle=True,generator=generator,
                      num_workers=8,pin_memory=True,drop_last=True)
    bn,decoder=build_hap_lllg(variant,seed);bn=bn.to(device);decoder=decoder.to(device)
    guard=actual_guard(bn,decoder,variant)
    parameters=list(bn.parameters())+list(decoder.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=5e-4,weight_decay=.05)
    scheduler=torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=5e-4,epochs=45,steps_per_epoch=len(loader))
    metadata={'experiment_id':exp.name,'dataset':'UIT-ADrone','variant':variant,'seed':seed,'epochs':45,
              'run_id':args.run_id,'source_commit':args.source_commit,'guard':guard,**data,
              'code_hashes':audit['code_hashes'],'teacher_state_sha256':audit['teacher']['state_sha256'],
              'match':'pldd_match top5 patch cosine, four groups alpha0.5; training=batch mean, test=frame score',
              'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
                             'gpu':torch.cuda.get_device_name(device)}}
    resume=exp/'ckpt/uit_resume.pth';start_epoch=0;last_loss=None
    if args.resume and resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        for key in ['variant','seed','source_commit','guard','split_fingerprint','code_hashes','teacher_state_sha256']:
            if state[key]!=metadata[key]:
                raise ValueError(f'resume mismatch: {key}')
        bn.load_state_dict(state['bottleneck_state_dict']);decoder.load_state_dict(state['decoder_state_dict'])
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        for values in optimizer.state.values():
            for key,value in values.items():
                if isinstance(value,torch.Tensor) and key!='step':values[key]=value.to(device)
        start_epoch=state['epoch'];last_loss=state['train_loss']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        generator.set_state(state['loader_rng']);del state
    print(json.dumps({**metadata,'kind':'actual_model_guard','resumed_epoch':start_epoch}),flush=True)
    started=time.monotonic()
    for epoch in range(start_epoch,45):
        bn.train();decoder.train();total=count=0
        for xb in loader:
            with torch.no_grad():taps=teacher(xb.to(device,non_blocking=True))['patches']
            loss=pldd_match(taps,decoder(bn(taps))).mean()
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            optimizer.zero_grad();loss.backward();optimizer.step();scheduler.step()
            total+=loss.item()*len(xb);count+=len(xb)
        last_loss=total/count
        state={**metadata,'epoch':epoch+1,'train_loss':last_loss,'bottleneck_state_dict':bn.state_dict(),
               'decoder_state_dict':decoder.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
               'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':generator.get_state()}
        save_state(resume,state)
        if epoch+1 in [10,20,30,40,45]:
            save_state(exp/'ckpt'/f'uit_epoch{epoch+1}.pth',{**metadata,'epoch':epoch+1,'train_loss':last_loss,
                'bottleneck_state_dict':{k:v.detach().cpu() for k,v in bn.state_dict().items()},
                'decoder_state_dict':{k:v.detach().cpu() for k,v in decoder.state_dict().items()}})
        print(f'epoch={epoch+1} loss={last_loss:.6f} elapsed_s={time.monotonic()-started:.1f}',flush=True)
        save_json(exp/'results/status.json',{'status':'running','stage':'train','epoch':epoch+1,'last_loss':last_loss,'time':now()})
    bn.eval();decoder.eval()
    final=exp/'ckpt/uit_final.pth'
    save_state(final,{**metadata,'epoch':45,'train_loss':last_loss,
               'bottleneck_state_dict':{k:v.detach().cpu() for k,v in bn.state_dict().items()},
               'decoder_state_dict':{k:v.detach().cpu() for k,v in decoder.state_dict().items()}})
    save_json(exp/'results/status.json',{'status':'running','stage':'evaluate','epoch':45,'time':now()})
    output={**metadata,**evaluate(teacher,bn,decoder,videos,transform,device,exp),'train_loss':last_loss,
            'checkpoint':'ckpt/uit_final.pth','checkpoint_sha256':sha(final),
            'elapsed_s':time.monotonic()-started,'finished_at':now()}
    save_json(exp/'results/uit.json',output)
    save_json(exp/'results/status.json',{'status':'completed','stage':'evaluated','epoch':45,'time':now()})
    save_json(exp/'results/eval_status.json',{'status':'completed','completed_videos':51,'total_videos':51,'frames':147005,'time':now()})
    print(json.dumps(output),flush=True)


if __name__=='__main__':
    main()
