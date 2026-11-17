#!/usr/bin/env python3
"""Run architecture/gradient/routing tests and audit copied DA results before launch."""
from __future__ import annotations
import csv,gc,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,roc_curve
from model.publication_rd import SPECS,build_publication_model,architecture_guard,publication_match,load_historical_weights
from model.teacher import build_appearance_teacher
from model.student import StudentDecoder
from model.student_structural import FourBlockGlobalLocalDecoder
from model.loss import matched_score
from scripts.da_publication_scene import SCENES,data_manifest,sha,save_json,config
BASE=ROOT/'ablation/da/publication'

def legacy_reference(payload,cfg):
    name=cfg['variant']
    if name=='pt_rd':
        ref=StudentDecoder(n_taps=4,depth=12)
        ref.load_state_dict(payload['student_state_dict'])
    elif name=='hap_rd':
        ref=FourBlockGlobalLocalDecoder(block_order='LLLG')
        ref.load_state_dict(payload['student_state_dict'])
    else:
        # Equivalent historical standard-block wrapper: map V1's .mlp into
        # the GGGG+GELU FFN wrapper, without importing any old runner.
        ref=FourBlockGlobalLocalDecoder(block_order='GGGG',ffn_type='gelumlp')
        sd=payload['student_state_dict']
        if not any('.ffn.mlp.' in k for k in sd):
            sd={k.replace('.mlp.','.ffn.mlp.'):v for k,v in sd.items()}
        ref.load_state_dict(sd)
    return ref

def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    plan=json.loads((BASE/'publication_chain_plan.json').read_text())
    report={'status':'running','torch':torch.__version__,'model_checks':[], 'reused_experiments':[], 'data':{},'code_hashes':{}}
    for scene,dirname in SCENES.items():
        train,videos,fp=data_manifest(Path('/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly')/dirname)
        report['data'][scene]={'train_frames':len(train),'test_videos':len(videos),'split_fingerprint':fp}
    teacher=build_appearance_teacher(layers=list(range(12))).cuda().eval()
    with torch.no_grad():features=teacher(torch.zeros(1,3,256,256,device='cuda'))['patches']
    assert len(features)==12 and all(t.shape==(1,256,768) for t in features)
    assert not any(p.requires_grad for p in teacher.parameters())
    report['teacher']={'patch_shape':[1,256,768],'taps':12,'frozen':True}
    del teacher,features;torch.cuda.empty_cache()
    # Exercise every fresh experiment config (including all seeds), actual
    # forward reconstruction shape, backward path, Q and head routes.
    for entry in plan['entries']:
        exp=ROOT/entry['path'];c=config(exp);name=c['variant'];seed=c['seed']
        torch.manual_seed(seed)
        bn,dec=build_publication_model(name,seed);bn=bn.cuda();dec=dec.cuda()
        guard=architecture_guard(bn,dec,name)
        taps=[torch.randn(1,256,768,device='cuda') for _ in c['model']['taps']]
        rec=dec(bn(taps));scores=publication_match(taps,rec,name)
        assert scores.shape==(1,) and torch.isfinite(scores).all()
        scores.mean().backward()
        assert all(h.weight.grad is not None and torch.isfinite(h.weight.grad).all() for h in dec.heads)
        assert bn.fuse[0].weight.grad is not None
        if c['model']['rank2']:
            assert bn.raw_depth_basis.grad is not None and torch.isfinite(bn.raw_depth_basis.grad).all()
            other,_=build_publication_model(name,seed)
            assert torch.allclose(bn.effective_depth_basis().detach().cpu(),other.effective_depth_basis().detach(),atol=1e-6)
            del other
        with torch.no_grad():
            ref=matched_score(taps,rec,k=5,alpha=c['model']['alpha'])
            assert torch.allclose(scores,ref,atol=1e-7)
            if name=='c_ablation':
                per=torch.stack([(1-torch.nn.functional.cosine_similarity(t,r,dim=-1)).topk(5,dim=-1).values.mean(-1) for t,r in zip(taps,rec)],1)
                assert torch.allclose(scores,per.mean(1),atol=1e-7)
        # Observe which stage each independent head consumes.
        stage_tensors={};head_sources={};handles=[]
        for idx,block in enumerate(dec.blocks):
            handles.append(block.register_forward_hook(lambda m,a,o,i=idx:stage_tensors.__setitem__(i,o)))
        for idx,head in enumerate(dec.heads):
            handles.append(head.register_forward_pre_hook(lambda m,a,i=idx:head_sources.__setitem__(i,a[0])))
        with torch.no_grad():dec(torch.randn(1,256,64,device='cuda'))
        for stage,routes in dec.routes.items():
            for head,tap in routes:assert head_sources[head] is stage_tensors[stage]
        for h in handles:h.remove()
        report['model_checks'].append({'experiment':exp.name,'guard':guard,'forward_backward':'passed','head_route_hooks':'passed','matched_score':'passed'})
        del bn,dec,taps,rec,scores,ref,stage_tensors,head_sources,handles;gc.collect();torch.cuda.empty_cache()
        print('model checked',exp.name,flush=True)
    reference_keys={}
    for entry in plan['entries']:
        if not entry['reused_from']:continue
        exp=ROOT/entry['path'];c=config(exp);name=c['variant'];seed=c['seed'];scene_checks=[]
        for scene in SCENES:
            metric=json.loads((exp/'results'/f'{scene}.json').read_text())
            assert metric['seed']==seed and metric['epochs']==30 and metric['trainable_parameters']==c['model']['params']
            assert metric['split_fingerprint']==report['data'][scene]['split_fingerprint']
            ckpt=exp/'ckpt'/f'{scene}_final.pth';assert sha(ckpt)==metric['checkpoint_sha256']
            with (exp/'results'/f'{scene}_scores.csv').open() as f:rows=list(csv.DictReader(f))
            keys=[(x['video'],x['frame'],int(x['label'])) for x in rows];reference_keys.setdefault(scene,keys)
            assert keys==reference_keys[scene] and len({(x[0],x[1]) for x in keys})==len(keys)
            y=[x[2] for x in keys];s=[float(x['anomaly_score']) for x in rows]
            assert np.isfinite(s).all();auc=roc_auc_score(y,s)*100;fpr,tpr,_=roc_curve(y,s)
            eer=float(fpr[np.nanargmin(np.abs(1-tpr-fpr))])
            assert abs(auc-metric['auc_percent'])<1e-9 and abs(eer-metric['eer'])<1e-12
            payload=torch.load(ckpt,map_location='cpu',weights_only=False)
            bn,dec=build_publication_model(name,seed);load_historical_weights(bn,dec,payload)
            guard=architecture_guard(bn,dec,name)
            old=legacy_reference(payload,c)
            code=torch.randn(1,256,64)
            with torch.no_grad():
                actual=dec(code);legacy=old(code)
                assert len(actual)==len(legacy) and all(torch.allclose(a,b,atol=1e-6,rtol=1e-5) for a,b in zip(actual,legacy))
            if metric.get('model_guard'):
                old_guard=metric['model_guard'];assert old_guard['block_order']==c['model']['order']
            if name=='pt_rd':assert payload['teacher_taps']==[5,7,9,11]
            elif name=='d_ablation' and seed==42:assert payload['teacher_taps']==list(range(12))
            elif name=='d_ablation':assert payload['ablation']['block_order']=='GGGG' and payload['ablation']['ffn_type']=='gelumlp'
            scene_checks.append({'scene':scene,'checkpoint_sha256':metric['checkpoint_sha256'],'hash_verified':True,'auc_recomputed':auc,'eer_recomputed':eer,'new_entry_strict_load':'passed','reference_decoder_forward':'passed'})
            del bn,dec,old,payload,code,actual,legacy;gc.collect()
        r=json.loads((exp/'record.json').read_text());r['review_status']='passed_preflight';save_json(exp/'record.json',r)
        report['reused_experiments'].append({'experiment':exp.name,'source':entry['reused_from'],'scenes':scene_checks})
        print('copied experiment verified',exp.name,flush=True)
    paths=['model/publication_rd.py','model/student_structural.py','model/bottleneck.py','model/bottleneck_depthrank.py',
           'model/bottleneck_depthrank_learnable.py','model/teacher.py','scripts/da_publication_scene.py',
           'scripts/da_publication_scheduler.py','scripts/preflight_da_publication.py']
    paths += [str((ROOT/e['path']/'scripts/run_scene.py').relative_to(ROOT)) for e in plan['entries']]
    paths += [str((ROOT/e['path']/'config.json').relative_to(ROOT)) for e in plan['entries']]
    report['code_hashes']={p:sha(ROOT/p) for p in paths}
    report['status']='passed';save_json(BASE/'publication_preflight.json',report)
    print('PREFLIGHT PASSED',len(report['model_checks']),'configs,',len(report['reused_experiments'])*7,'copied scene checkpoints',flush=True)
if __name__=='__main__':main()
