#!/usr/bin/env python3
"""New DA publication-chain training and matched evaluation entrypoint.

Does not import any historical variant/GL/Loss runner. All experiment settings
are checked against the explicit publication registry before model execution.
"""
from __future__ import annotations
import argparse, csv, datetime as dt, hashlib, json, os, random, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from model.teacher import build_appearance_teacher
from model.publication_rd import model_spec, build_publication_model, architecture_guard, publication_match
SCENES={'bike':'Bike Roundabout','crossroads':'Crossroads','farmland':'Farmland Inspection',
        'highway':'Highway','railway':'Railway Inspection','solar':'Solar Panel Inspection','vehicle':'Vehicle Roundabout'}

def now(): return dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def save_json(p,d):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save_torch(p,d):
    p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_name(p.name+'.tmp');torch.save(d,tmp);tmp.replace(p)
def frame_key(p):return int(p.stem)
class Frames(Dataset):
    def __init__(self,paths,tf):self.paths,self.tf=paths,tf
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as im:return self.tf(im.convert('RGB'))
def data_manifest(scene_dir):
    train=sorted(scene_dir.glob('sequence*/train/*/*.jpg'))
    videos=sorted(p for p in scene_dir.glob('sequence*/test/*') if p.is_dir())
    if not train or not videos:raise FileNotFoundError(scene_dir)
    lines=[f'{p.relative_to(scene_dir)}:{p.stat().st_size}' for p in train]
    for v in videos:
        frames=sorted(v.glob('*.jpg'),key=frame_key); lab=Path(str(v)+'.npy')
        if not frames or not lab.exists():raise FileNotFoundError(v)
        y=np.asarray(np.load(lab,allow_pickle=True)).astype(int).ravel()
        if len(frames)!=len(y) or not set(y).issubset({0,1}):raise ValueError(f'bad labels {v}')
        lines += [f'{p.relative_to(scene_dir)}:{p.stat().st_size}' for p in frames]
        lines.append(f'{lab.relative_to(scene_dir)}:{sha(lab)}')
    return train,videos,hashlib.sha256('\n'.join(lines).encode()).hexdigest()
def config(exp):
    c=json.loads((exp/'config.json').read_text());s=model_spec(c['variant'])
    if c['dataset']!='da' or c['seed'] not in [42,43,44] or c['model']!=s:raise ValueError('wrong configuration')
    t=c['training']
    if t!={'epochs':30,'batch':64,'workers':8,'image_size':256,'max_lr':.0005,'weight_decay':.05,
           'optimizer':'AdamW','scheduler':'OneCycleLR','data_root':'/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly',
           'deterministic':True,'drop_last':True}:raise ValueError('noncanonical training configuration')
    return c
@torch.no_grad()
def evaluate(teacher,bn,dec,videos,tf,device,path,variant,scene_dir):
    ys,ss=[],[]
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['video','frame','label','anomaly_score'])
        for v in videos:
            paths=sorted(v.glob('*.jpg'),key=frame_key)
            y=np.asarray(np.load(str(v)+'.npy',allow_pickle=True)).astype(int).ravel()
            scores=[]
            for xb in DataLoader(Frames(paths,tf),batch_size=32,shuffle=False,num_workers=4):
                taps=teacher(xb.to(device,non_blocking=True))['patches']
                scores += publication_match(taps,dec(bn(taps)),variant).cpu().tolist()
            if len(scores)!=len(y):raise RuntimeError('test output/frame mismatch')
            for p,label,score in zip(paths,y,scores):w.writerow([str(v.relative_to(scene_dir)),p.name,int(label),score])
            ys+=y.tolist();ss+=scores
    tmp.replace(path)
    fpr,tpr,_=roc_curve(ys,ss)
    return {'auc_percent':roc_auc_score(ys,ss)*100,'eer':float(fpr[np.nanargmin(np.abs(1-tpr-fpr))]),
            'frames':len(ys),'positive_frames':sum(ys)}
def main(default_experiment=None):
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',type=Path,default=default_experiment,required=default_experiment is None)
    ap.add_argument('--scene',required=True,choices=SCENES);ap.add_argument('--run-id',required=True)
    ap.add_argument('--source-commit',required=True);ap.add_argument('--source-hashes',type=Path,required=True)
    ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    exp=a.experiment.resolve();c=config(exp);name=c['variant'];seed=c['seed'];spec=c['model']
    # Check code bytes against the prelaunch audit even while scheduler updates metadata.
    code_hashes=json.loads(a.source_hashes.read_text())['code_hashes']
    for rel,h in code_hashes.items():
        if sha(ROOT/rel)!=h:raise RuntimeError(f'source changed since preflight: {rel}')
    metricpath=exp/'results'/f'{a.scene}.json';ckpt=exp/'ckpt'/f'{a.scene}_final.pth';progress=exp/'ckpt'/f'{a.scene}_resume.pth'
    if metricpath.exists():raise FileExistsError(f'complete result already exists {metricpath}')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    device=torch.device('cuda:0')
    teacher=build_appearance_teacher(layers=spec['taps']).to(device).eval()
    if any(p.requires_grad for p in teacher.parameters()):raise RuntimeError('teacher must be frozen')
    scene_dir=Path(c['training']['data_root'])/SCENES[a.scene]
    train,videos,fingerprint=data_manifest(scene_dir)
    tf=T.Compose([T.Resize((256,256)),T.ToTensor(),T.Normalize(teacher.img_mean,teacher.img_std)])
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(Frames(train,tf),batch_size=64,shuffle=True,generator=generator,num_workers=8,pin_memory=True,drop_last=True)
    bn,dec=build_publication_model(name,seed);bn=bn.to(device);dec=dec.to(device)
    guard=architecture_guard(bn,dec,name)
    params=list(bn.parameters())+list(dec.parameters());opt=torch.optim.AdamW(params,lr=.0005,weight_decay=.05)
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.0005,epochs=30,steps_per_epoch=len(loader))
    metadata={'dataset':'Drone-Anomaly','experiment_id':exp.name,'variant':name,'seed':seed,'scene':a.scene,
              'epochs':30,'source_commit':a.source_commit,'run_id':a.run_id,'guard':guard,
              'split_fingerprint':fingerprint,'train_frames':len(train),'test_videos':len(videos),
              'data_root':c['training']['data_root'],'cuda':torch.cuda.get_device_name(device),
              'environment':{'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda},
              'match':'publication_match train=batch mean; test=per-frame score','code_hashes':code_hashes}
    start_epoch=0;last_loss=None
    if a.resume and progress.exists():
        state=torch.load(progress,map_location='cpu',weights_only=False)
        for key in ['variant','seed','scene','split_fingerprint','guard']:
            if state[key]!=metadata[key]:raise RuntimeError(f'resume mismatch {key}')
        if state['code_hashes']!=code_hashes:raise RuntimeError('resume source changed')
        bn.load_state_dict(state['ocbe_state_dict']);dec.load_state_dict(state['student_state_dict'])
        opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
        for value in opt.state.values():
            for k,v in value.items():
                if isinstance(v,torch.Tensor) and k!='step':value[k]=v.to(device)
        start_epoch=state['epoch'];last_loss=state['train_loss']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
        generator.set_state(state['loader_rng'])
    print(json.dumps({**metadata,'kind':'model_guard','resumed_epoch':start_epoch},ensure_ascii=False),flush=True)
    t0=time.monotonic()
    for ep in range(start_epoch,30):
        bn.train();dec.train();total=0.;count=0
        for xb in loader:
            xb=xb.to(device,non_blocking=True)
            with torch.no_grad():taps=teacher(xb)['patches']
            loss=publication_match(taps,dec(bn(taps)),name).mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
            opt.zero_grad();loss.backward();opt.step();sched.step()
            total+=loss.item()*len(xb);count+=len(xb)
        last_loss=total/count
        state={**metadata,'epoch':ep+1,'train_loss':last_loss,'ocbe_state_dict':bn.state_dict(),
               'student_state_dict':dec.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
               'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state_all(),'loader_rng':generator.get_state()}
        save_torch(progress,state)
        print(f'epoch={ep+1} loss={last_loss:.6f} elapsed_s={time.monotonic()-t0:.1f}',flush=True)
        save_json(exp/'results'/f'{a.scene}_status.json',{'status':'running','epoch':ep+1,'run_id':a.run_id,'last_loss':last_loss,'time':now()})
    bn.eval();dec.eval()
    save_torch(ckpt,{**metadata,'train_loss':last_loss,
                    'ocbe_state_dict':{k:v.detach().cpu() for k,v in bn.state_dict().items()},
                    'student_state_dict':{k:v.detach().cpu() for k,v in dec.state_dict().items()}})
    metric={**metadata,**evaluate(teacher,bn,dec,videos,tf,device,exp/'results'/f'{a.scene}_scores.csv',name,scene_dir),
            'train_loss':last_loss,'checkpoint':str(ckpt.relative_to(exp)),'checkpoint_sha256':sha(ckpt),
            'elapsed_s':time.monotonic()-t0,'finished_at':now()}
    save_json(metricpath,metric)
    save_json(exp/'results'/f'{a.scene}_status.json',{'status':'completed','epoch':30,'run_id':a.run_id,'time':now()})
    print(json.dumps(metric,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
