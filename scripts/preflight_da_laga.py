#!/usr/bin/env python3
"""Preflight actual teacher, LGGD architecture, routes, gradients and alpha0 match."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.lggd_no_laga import ablation_guard, ablation_match, build_ablation_model
from model.publication_rd import build_publication_model
from model.loss import matched_score
from model.teacher import build_appearance_teacher
from scripts.da_laga_scene import SCENES, checked_config, data_manifest, save_json, sha


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    plan=json.loads(args.manifest.read_text())
    owner=args.manifest.resolve().parent
    report={'status':'running','torch':torch.__version__,'data':{},'model_checks':[],'code_hashes':{}}
    torch.set_num_threads(2)
    for scene,dirname in SCENES.items():
        training,videos,fp=data_manifest(Path('/data1/data8tb/liangyuxuan/dataset/Drone-Anomaly')/dirname)
        reference=json.loads((ROOT/'ablation/da/publication/20261117_hap_rd_s42/results'/f'{scene}.json').read_text())
        assert fp==reference['split_fingerprint']
        report['data'][scene]={'train_frames':len(training),'test_videos':len(videos),'split_fingerprint':fp}
    teacher=build_appearance_teacher(layers=list(range(12))).cuda().eval()
    with torch.no_grad():
        features=teacher(torch.zeros(1,3,256,256,device='cuda'))['patches']
    assert len(features)==12 and all(t.shape==(1,256,768) for t in features)
    assert not any(p.requires_grad for p in teacher.parameters())
    report['teacher']={'frozen':True,'taps':12,'patch_shape':[1,256,768]}
    del teacher,features
    torch.cuda.empty_cache()
    for rel in plan['experiments']:
        exp=ROOT/rel
        cfg=checked_config(exp)
        seed=cfg['seed']
        torch.manual_seed(seed)
        bn,decoder=build_ablation_model()
        # The baseline and ablation have exactly the same initial trainable tensors.
        torch.manual_seed(seed)
        reference_bn,reference_decoder=build_publication_model('hap_rd',seed)
        for actual,reference in [(bn,reference_bn),(decoder,reference_decoder)]:
            assert actual.state_dict().keys()==reference.state_dict().keys()
            assert all(torch.equal(value,reference.state_dict()[key]) for key,value in actual.state_dict().items())
        del reference_bn,reference_decoder
        bn,decoder=bn.cuda(),decoder.cuda()
        guard=ablation_guard(bn,decoder)
        assert guard['actual_block_order']=='LLLG' and guard['match']['alpha']==0
        stages,sources,handles={},{},[]
        for index,block in enumerate(decoder.blocks):
            handles.append(block.register_forward_hook(lambda m,a,o,i=index:stages.__setitem__(i,o)))
        for index,head in enumerate(decoder.heads):
            handles.append(head.register_forward_pre_hook(lambda m,a,i=index:sources.__setitem__(i,a[0])))
        taps=[torch.randn(1,256,768,device='cuda') for _ in range(12)]
        recons=decoder(bn(taps))
        score=ablation_match(taps,recons)
        distances=torch.stack([(1-F.cosine_similarity(t,r,dim=-1)).topk(5,dim=-1).values.mean(-1)
                               for t,r in zip(taps,recons)],dim=1)
        assert torch.allclose(score,distances.mean(1),atol=1e-7,rtol=1e-6)
        assert torch.allclose(score,matched_score(taps,recons,k=5,alpha=0),atol=1e-7,rtol=1e-6)
        for stage,routes in decoder.routes.items():
            for head,tap in routes:
                assert sources[head] is stages[stage]
        score.mean().backward()
        assert torch.isfinite(score).all()
        assert all(h.weight.grad is not None and torch.isfinite(h.weight.grad).all() for h in decoder.heads)
        assert bn.fuse[0].weight.grad is not None
        for handle in handles:
            handle.remove()
        report['model_checks'].append({'experiment':exp.name,'seed':seed,'guard':guard,
                                        'initial_state_equal_to_hap_rd':True,'forward_backward':'passed',
                                        'head_route_hooks':'passed','match_equals_mean12_and_alpha0_reference':'passed'})
        del bn,decoder,taps,recons,score,distances,stages,sources
        gc.collect();torch.cuda.empty_cache()
        print('preflight passed',exp.name,flush=True)
    files=['model/lggd_no_laga.py','model/publication_rd.py','model/student_structural.py',
           'model/bottleneck.py','model/bottleneck_depthrank.py','model/bottleneck_depthrank_learnable.py',
           'model/teacher.py','model/loss.py','scripts/da_laga_scene.py','scripts/da_laga_scheduler.py',
           'scripts/preflight_da_laga.py']
    files += [str(Path(rel)/suffix) for rel in plan['experiments'] for suffix in ['config.json','scripts/run_scene.py']]
    report['code_hashes']={rel:sha(ROOT/rel) for rel in files}
    report['status']='passed'
    save_json(owner/'preflight.json',report)
    print('PREFLIGHT PASSED: all three seeds, identical baseline initialization and alpha0 match',flush=True)


if __name__=='__main__':
    main()
