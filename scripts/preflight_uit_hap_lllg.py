#!/usr/bin/env python3
"""Check independent UIT LLLG models, teacher, routes, gradients and PLDD match."""
from __future__ import annotations
import argparse
import gc
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model.hap_lllg import build_hap_lllg,actual_guard,pldd_match,GLMB,GGMB
from model.teacher import build_appearance_teacher
from scripts.uit_hap_lllg_runner import checked_config,data_manifest,read,save_json,sha,state_hash


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    manifest=read(args.manifest);owner=args.manifest.resolve().parent
    torch.set_num_threads(2)
    _,_,data=data_manifest()
    historical=read(ROOT/'experiments/uit/appearance/20261104_variant_matrix_v2_rd_s4_t12_gl768_h512_s42/results/uit.json')
    assert data['split_fingerprint']==historical['split_fingerprint']
    teacher=build_appearance_teacher(layers=list(range(12))).eval()
    teacher_hash=state_hash(teacher)
    assert not any(p.requires_grad for p in teacher.parameters())
    teacher=teacher.cuda()
    with torch.no_grad():features=teacher(torch.zeros(1,3,256,256,device='cuda'))['patches']
    assert len(features)==12 and all(f.shape==(1,256,768) for f in features)
    report={'status':'running','data':data,'teacher':{'frozen':True,'taps':list(range(12)),
             'patch_shape':[1,256,768],'state_sha256':teacher_hash},'torch':torch.__version__,'checks':[]}
    del teacher,features;torch.cuda.empty_cache()
    # Old entrypoints are imported only here as numerical references, never by the runner.
    from model.publication_rd import build_publication_model,publication_match
    for rel in manifest['experiments']:
        exp=ROOT/rel;cfg=checked_config(exp);variant=cfg['variant'];seed=cfg['seed']
        torch.manual_seed(seed)
        bn,decoder=build_hap_lllg(variant,seed)
        torch.manual_seed(seed)
        reference_bn,reference_decoder=build_publication_model(variant,seed)
        for current,reference in [(bn,reference_bn),(decoder,reference_decoder)]:
            assert current.state_dict().keys()==reference.state_dict().keys()
            assert all(torch.equal(v,reference.state_dict()[k]) for k,v in current.state_dict().items())
        bn,decoder=bn.cuda(),decoder.cuda()
        reference_bn,reference_decoder=reference_bn.cuda(),reference_decoder.cuda()
        guard=actual_guard(bn,decoder,variant)
        assert guard['block_order']=='LLLG' and guard['observed_block_types']==['GLMB','GLMB','GLMB','GGMB']
        stages,sources,handles={},{},[]
        for i,block in enumerate(decoder.blocks):
            handles.append(block.register_forward_hook(lambda m,a,o,j=i:stages.__setitem__(j,o)))
        for i,head in enumerate(decoder.heads):
            handles.append(head.register_forward_pre_hook(lambda m,a,j=i:sources.__setitem__(j,a[0])))
        taps=[torch.randn(1,256,768,device='cuda') for _ in range(12)]
        code=bn(taps);predictions=decoder(code);scores=pldd_match(taps,predictions)
        with torch.no_grad():
            ref_code=reference_bn(taps);ref_predictions=reference_decoder(ref_code)
            assert torch.allclose(code,ref_code,atol=1e-6,rtol=1e-5)
            assert all(torch.allclose(a,b,atol=1e-6,rtol=1e-5) for a,b in zip(predictions,ref_predictions))
            assert torch.allclose(scores,publication_match(taps,predictions,variant),atol=1e-7,rtol=1e-6)
        for stage,heads in decoder.routes.items():
            for head in heads:assert sources[head] is stages[stage]
        scores.mean().backward()
        assert torch.isfinite(scores).all() and all(h.weight.grad is not None and torch.isfinite(h.weight.grad).all() for h in decoder.heads)
        assert bn.fuse[0].weight.grad is not None
        if cfg['model']['podp']:
            assert bn.raw_depth_basis.grad is not None and torch.isfinite(bn.raw_depth_basis.grad).all()
            assert torch.count_nonzero(bn.raw_depth_basis.grad)>0
            optimizer=torch.optim.AdamW(list(bn.parameters())+list(decoder.parameters()),lr=5e-4,weight_decay=.05)
            optimizer.step()
            q=bn.effective_depth_basis()
            assert torch.allclose(q@q.T,torch.eye(2,device='cuda'),atol=1e-5)
            del optimizer,q
        for handle in handles:handle.remove()
        # A GLLL accidental construction must be rejected by the actual-module guard.
        original=decoder.blocks
        decoder.blocks=torch.nn.ModuleList([original[3],original[0],original[1],original[2]])
        try:
            actual_guard(bn,decoder,variant)
        except ValueError:
            pass
        else:
            raise AssertionError('guard accepted GLLL')
        decoder.blocks=original
        report['checks'].append({'experiment':exp.name,'variant':variant,'seed':seed,'guard':guard,
                                'initial_state_reference':'passed','numerical_forward_reference':'passed',
                                'head_route_hooks':'passed','all_head_gradients':'passed',
                                'pldd_matched_train_test':'passed','reject_glll':'passed',
                                'podp_gradient_and_post_step_orthogonality':'passed' if cfg['model']['podp'] else 'not_applicable'})
        del bn,decoder,reference_bn,reference_decoder,taps,code,predictions,scores,ref_code,ref_predictions,stages,sources,handles,original
        gc.collect();torch.cuda.empty_cache()
        print('preflight passed',exp.name,flush=True)
    files=['model/hap_lllg.py','model/teacher.py','scripts/uit_hap_lllg_runner.py',
           'scripts/uit_hap_lllg_scheduler.py','scripts/preflight_uit_hap_lllg.py']
    files += [str(Path(rel)/name) for rel in manifest['experiments'] for name in ['config.json','scripts/run_uit.py']]
    report['code_hashes']={rel:sha(ROOT/rel) for rel in files}
    report['status']='passed'
    save_json(owner/'preflight.json',report)
    print('PREFLIGHT PASSED: six independent LLLG configs; teacher, routes, forward/backward and match verified',flush=True)


if __name__=='__main__':
    main()
