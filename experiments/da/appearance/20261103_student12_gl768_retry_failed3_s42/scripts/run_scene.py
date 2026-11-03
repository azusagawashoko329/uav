import argparse,csv,glob,hashlib,json,os,random,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score,roc_curve
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
from model import build_appearance_teacher,build_bottleneck,anomaly_readouts
from model.student import StudentDecoder,Block
from model.student_structural import GlobalMixerBlock,LocalMixerBlock
SCENES={'bike':'Bike Roundabout','crossroads':'Crossroads','farmland':'Farmland Inspection','highway':'Highway','railway':'Railway Inspection','solar':'Solar Panel Inspection','vehicle':'Vehicle Roundabout'}
class Frames(Dataset):
 def __init__(self,p,t):self.p,self.t=p,t
 def __len__(self):return len(self.p)
 def __getitem__(self,i):return self.t(Image.open(self.p[i]).convert('RGB'))
def fk(p):return int(Path(p).stem)
def manifest(d):
 tr=sorted(glob.glob(str(d/'sequence*'/'train'/'*'/'*.jpg'))); vids=sorted(Path(p) for p in glob.glob(str(d/'sequence*'/'test'/'*')) if Path(p).is_dir()); lines=[f'{p}:{Path(p).stat().st_size}' for p in tr]
 for v in vids:
  ps=sorted(v.glob('*.jpg'),key=fk); y=Path(str(v)+'.npy'); assert len(ps)==len(np.load(y)); lines += [f'{p}:{p.stat().st_size}' for p in ps]+[f'{y}:{hashlib.sha256(y.read_bytes()).hexdigest()}']
 return tr,vids,hashlib.sha256('\n'.join(lines).encode()).hexdigest()
class GL12(nn.Module):
 def __init__(self):
  super().__init__();self.proj=nn.Linear(64,768);self.blocks=nn.ModuleList(sum(([GlobalMixerBlock(768,8,2.0)]+[LocalMixerBlock(768,2.0) for _ in range(3)] for _ in range(3)), []));self.recon_at=[2,5,8,11];self.heads=nn.ModuleList([nn.Linear(768,768) for _ in range(4)])
 def forward(self,x):
  x=self.proj(x);o=[]
  hi=0
  for i,b in enumerate(self.blocks):
   x=b(x)
   if i in self.recon_at:o.append(self.heads[hi](x));hi+=1
  return o[::-1]
def loss(t,r):return torch.stack([(1-F.cosine_similarity(a,b,dim=-1)).topk(5,dim=-1).values.mean(-1) for a,b in zip(t,r)],1).mean()
def eer(y,s):
 f,t,_=roc_curve(y,s);return float(f[np.argmin(np.abs((1-t)-f))])
@torch.no_grad()
def evaluate(teacher,ocbe,student,vids,tf,dev,out):
 rows=[];ys=[];ss=[]
 for v in vids:
  ps=sorted(v.glob('*.jpg'),key=fk);y=np.load(str(v)+'.npy').astype(int).ravel();ldr=DataLoader(Frames(ps,tf),32,num_workers=4);z=[]
  for x in ldr:z += anomaly_readouts(teacher(x.to(dev))['patches'],student(ocbe(teacher(x.to(dev))['patches'])) ,5)['topk'].cpu().tolist()
  rows += [(v.name,p.name,int(a),float(b)) for p,a,b in zip(ps,y,z)];ys += y.tolist();ss += z
 with open(out,'w',newline='') as f: w=csv.writer(f);w.writerow(['video','frame','label','anomaly_score']);w.writerows(rows)
 return roc_auc_score(ys,ss)*100,eer(ys,ss),len(ys),int(sum(ys))
def main():
 p=argparse.ArgumentParser();p.add_argument('--scene',choices=SCENES,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch',type=int,default=64);p.add_argument('--workers',type=int,default=8);p.add_argument('--variant',choices=['plain12','gl12'],required=True);a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed);dev=torch.device(f'cuda:{a.gpu}');root=Path(os.environ.get('UAV_DA_ROOT','/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly'))/SCENES[a.scene];tr,vids,fp=manifest(root);teacher=build_appearance_teacher(layers=[5,7,9,11]).to(dev).eval();tf=T.Compose([T.Resize((256,256)),T.ToTensor(),T.Normalize(teacher.img_mean,teacher.img_std)]);ldr=DataLoader(Frames(tr,tf),a.batch,shuffle=True,num_workers=a.workers,pin_memory=True,drop_last=True,generator=torch.Generator().manual_seed(a.seed));ocbe=build_bottleneck(n_taps=4,code_dim=64,fuse_hidden=512).to(dev);student=StudentDecoder(4,64,768,12,8).to(dev) if a.variant=='plain12' else GL12().to(dev);params=list(ocbe.parameters())+list(student.parameters());opt=torch.optim.AdamW(params,lr=5e-4,weight_decay=.05);sch=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=5e-4,steps_per_epoch=len(ldr),epochs=a.epochs);print(json.dumps({'scene':a.scene,'variant':a.variant,'trainable_parameters':sum(x.numel() for x in params),'split_fingerprint':fp,'cuda':torch.cuda.get_device_name(dev)}),flush=True);t0=time.monotonic()
 for ep in range(a.epochs):
  ocbe.train();student.train();tot=n=0
  for x in ldr:
   x=x.to(dev)
   with torch.no_grad():taps=teacher(x)['patches']
   r=student(ocbe(taps));q=loss(taps,r);opt.zero_grad();q.backward();opt.step();sch.step();tot+=q.item()*len(x);n+=len(x)
  print(f'epoch={ep+1} loss={tot/n:.6f} elapsed_s={time.monotonic()-t0:.1f}',flush=True)
 ocbe.eval();student.eval();out=a.output/'results';out.mkdir(exist_ok=True);auc,e,n,pos=evaluate(teacher,ocbe,student,vids,tf,dev,out/f'{a.scene}_scores.csv');ck=a.output/'ckpt'/f'{a.scene}_final.pth';ck.parent.mkdir(exist_ok=True);torch.save({'scene':a.scene,'variant':a.variant,'seed':a.seed,'split_fingerprint':fp,'trainable_parameters':sum(x.numel() for x in params),'ocbe_state_dict':ocbe.state_dict(),'student_state_dict':student.state_dict()},ck);m={'scene':a.scene,'variant':a.variant,'seed':a.seed,'auc_percent':auc,'eer':e,'frames':n,'positive_frames':pos,'trainable_parameters':sum(x.numel() for x in params),'elapsed_s':time.monotonic()-t0,'checkpoint':str(ck.relative_to(a.output)),'checkpoint_sha256':hashlib.sha256(ck.read_bytes()).hexdigest(),'split_fingerprint':fp};(out/f'{a.scene}.json').write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(m),flush=True)
if __name__=='__main__':main()
