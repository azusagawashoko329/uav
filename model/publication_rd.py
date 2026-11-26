"""Explicit publication RD models; historical variant builders remain unchanged.

New decoder/routing and match entrypoints for the DA publication ablation chain.
Low-level OCBE, rank projection and mixer blocks remain shared model components.
"""
from __future__ import annotations
from copy import deepcopy
import torch
from torch import nn
from torch.nn import functional as F
from .bottleneck import OCBE
from .bottleneck_depthrank_learnable import LearnableDepthRank2OCBE
from .bottleneck_depthrank_group_learnable import GroupLearnableDepthRank2OCBE
from .student_structural import GlobalMixerBlock, LocalMixerBlock, GatedFFN

SPECS = {
    'pt_rd': dict(label='PT-RD', depth=12, taps=[5,7,9,11], width=768, hidden=512,
                  order='G'*12, ffn='gelumlp', rank2=False, alpha=0., params=89079104),
    'b_ablation': dict(label='B-ablation', depth=4, taps=[5,7,9,11], width=768, hidden=512,
                       order='GGGG', ffn='gelumlp', rank2=False, alpha=0., params=32376128),
    'c_ablation': dict(label='C-ablation', depth=4, taps=list(range(12)), width=768, hidden=512,
                       order='GGGG', ffn='gelumlp', rank2=False, alpha=0., params=40258880),
    'd_ablation': dict(label='D-ablation', depth=4, taps=list(range(12)), width=768, hidden=512,
                       order='GGGG', ffn='gelumlp', rank2=False, alpha=.5, params=40258880),
    'hap_rd': dict(label='HAP-RD', depth=4, taps=list(range(12)), width=768, hidden=512,
                   order='LLLG', ffn='gate', rank2=False, alpha=.5, params=30248000),
    'f1': dict(label='F1', depth=4, taps=list(range(12)), width=192, hidden=512,
               order='LLLG', ffn='gate', rank2=False, alpha=.5, params=7718336),
    'f2': dict(label='F2', depth=4, taps=list(range(12)), width=192, hidden=128,
               order='LLLG', ffn='gate', rank2=False, alpha=.5, params=4154432),
    'hap_rd_lite': dict(label='HAP-RD-Lite', depth=4, taps=list(range(12)), width=192, hidden=128,
                        order='LLLG', ffn='gate', rank2=True, alpha=.5, params=3761222),
    'hap_rd_lite_q4': dict(label='HAP-RD-Lite-Q4', depth=4, taps=list(range(12)), width=192, hidden=128,
                          order='LLLG', ffn='gate', rank2=True, q_groups=4, alpha=.5, params=3761240),
}

class StandardBlock(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))

    def forward(self, x):
        z = self.norm1(x)
        z, _ = self.attn(z,z,z,need_weights=False)
        x = x + z
        return x + self.mlp(self.norm2(x))

class PublicationDecoder(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = deepcopy(spec)
        self.proj_in = nn.Linear(64, spec['width'])
        blocks = []
        for ch in spec['order']:
            if spec['ffn'] == 'gelumlp':
                if ch != 'G': raise ValueError('standard decoder requires global blocks')
                blocks.append(StandardBlock(spec['width']))
            elif ch == 'G':
                blocks.append(GlobalMixerBlock(spec['width'], heads=8, ffn_ratio=2., ffn_type='gate'))
            else:
                blocks.append(LocalMixerBlock(spec['width'], ffn_ratio=2., kernel_size=3, ffn_type='gate'))
        self.blocks = nn.ModuleList(blocks)
        self.heads = nn.ModuleList(nn.Linear(spec['width'],768) for _ in spec['taps'])
        # Explicit stage -> (head index, teacher tap) routing. Four-tap head indices
        # match historical V0's stage-order heads; 12-tap heads use tap order.
        if len(spec['taps']) == 4:
            self.routes = {spec['depth']//4*(i+1)-1: [(i,spec['taps'][3-i])] for i in range(4)}
        else:
            self.routes = {i:[(tap,tap) for tap in range((3-i)*3,(4-i)*3)] for i in range(4)}

    def forward(self, code):
        x = self.proj_in(code)
        out = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            for head, tap in self.routes.get(i,[]):
                out[tap] = self.heads[head](x)
        return [out[t] for t in self.spec['taps']]

def model_spec(name):
    if name not in SPECS: raise ValueError(f'unknown publication variant {name}')
    return deepcopy(SPECS[name])

def build_publication_model(name, seed):
    spec = model_spec(name)
    if spec['rank2']:
        bottleneck_cls = (GroupLearnableDepthRank2OCBE if spec.get('q_groups', 1) == 4
                          else LearnableDepthRank2OCBE)
        bottleneck = bottleneck_cls(n_taps=12, code_dim=64, fuse_hidden=spec['hidden'],
                                    use_tap_norm=True, basis_init='random_orthonormal', basis_seed=seed)
    else:
        bottleneck = OCBE(n_taps=len(spec['taps']), code_dim=64, fuse_hidden=spec['hidden'], use_tap_norm=True)
    decoder = PublicationDecoder(spec)
    count = sum(p.numel() for p in bottleneck.parameters()) + sum(p.numel() for p in decoder.parameters())
    if spec['params'] and count != spec['params']: raise RuntimeError(f'parameter mismatch {name}: {count}')
    return bottleneck, decoder

def publication_match(taps, recons, name):
    """Returns B scores; training = batch mean, evaluation keeps B scores."""
    spec = model_spec(name)
    if len(taps) != len(spec['taps']) or len(recons) != len(taps): raise ValueError('tap count mismatch')
    per = []
    for t, r in zip(taps,recons):
        if t.shape != r.shape or t.ndim != 3 or t.shape[1:] != (256,768):
            raise ValueError(f'expected Bx256x768 teacher/reconstruction, got {t.shape}, {r.shape}')
        per.append((1-F.cosine_similarity(t,r,dim=-1)).topk(5,dim=-1).values.mean(-1))
    values = torch.stack(per,dim=1)
    if len(taps)==4: return values.mean(1)
    groups = values.reshape(values.size(0),4,3)
    return ((1-spec['alpha'])*groups.mean(-1)+spec['alpha']*groups.max(-1).values).mean(1)

def architecture_guard(bn, dec, name):
    spec=model_spec(name)
    actual_order=''.join('L' if isinstance(b,LocalMixerBlock) else 'G' for b in dec.blocks)
    if actual_order!=spec['order']: raise RuntimeError('wrong actual block order')
    for b in dec.blocks:
        if spec['ffn']=='gelumlp':
            if not isinstance(b,StandardBlock) or not isinstance(b.mlp[1],nn.GELU) or b.mlp[0].out_features!=4*spec['width']:
                raise RuntimeError('actual standard FFN is not ratio-4 GELU')
        else:
            if not isinstance(b.ffn,GatedFFN) or b.ffn.in_proj.out_features!=4*spec['width'] or b.ffn.out_proj.in_features!=2*spec['width']:
                raise RuntimeError('actual gated FFN is not hidden ratio 2')
        if isinstance(b,LocalMixerBlock):
            if b.dwconv.kernel_size!=(3,3) or b.dwconv.groups!=spec['width'] or b.pointwise.out_features!=spec['width']:
                raise RuntimeError('wrong local mixer')
        elif b.attn.num_heads!=8:raise RuntimeError('wrong attention heads')
    if len(bn.tap_norms)!=len(spec['taps']) or not all(isinstance(n,nn.LayerNorm) for n in bn.tap_norms):
        raise RuntimeError('tap normalization missing')
    if sorted(t for routes in dec.routes.values() for head,t in routes)!=sorted(spec['taps']):
        raise RuntimeError('routing does not cover each teacher tap exactly once')
    expected_in=(8 if spec['rank2'] else len(spec['taps']))*768
    if bn.fuse[0].in_features!=expected_in or bn.fuse[0].out_features!=spec['hidden']: raise RuntimeError('wrong bottleneck')
    if dec.proj_in.out_features!=spec['width'] or len(dec.heads)!=len(spec['taps']): raise RuntimeError('wrong decoder')
    if any(h.in_features!=spec['width'] or h.out_features!=768 for h in dec.heads): raise RuntimeError('wrong reconstruction head')
    expected_q_groups = spec.get('q_groups', 1)
    expected_bn_cls = (GroupLearnableDepthRank2OCBE if expected_q_groups == 4
                       else LearnableDepthRank2OCBE)
    if spec['rank2'] != isinstance(bn, expected_bn_cls): raise RuntimeError('wrong Q mode')
    q=None
    if spec['rank2']:
        q=bn.effective_depth_basis().detach().cpu()
        if expected_q_groups == 4:
            if q.shape != (4, 2, 3) or not torch.allclose(q @ q.transpose(1, 2),
                                                           torch.eye(2).expand(4, -1, -1),
                                                           atol=1e-5):
                raise RuntimeError('per-group Q not row-orthonormal')
        elif q.shape!=(2,3) or not torch.allclose(q@q.T,torch.eye(2),atol=1e-5):
            raise RuntimeError('Q not row-orthonormal')
    return {'variant':name,'teacher_taps':spec['taps'],'student_blocks':len(dec.blocks),
            'actual_block_order':actual_order,'ffn':spec['ffn'],'ffn_hidden_ratio':4 if spec['ffn']=='gelumlp' else 2,
            'student_width':spec['width'],'pdfb_hidden':bn.fuse[0].out_features,'code_dim':bn.code_dim,
            'tap_norm':True,'reconstruction_heads':len(dec.heads),'routes':{str(i):v for i,v in dec.routes.items()},
            'rank2':spec['rank2'],'q_shape':list(q.shape) if q is not None else None,
            'q_groups':expected_q_groups if spec['rank2'] else None,
            'q_shared_across_groups':spec['rank2'] and expected_q_groups == 1,
            'q_init':'random_orthonormal' if spec['rank2'] else None,
            'match':{'error':'patch_cosine','k':5,'groups':4 if len(spec['taps'])==12 else None,'alpha':spec['alpha']},
            'trainable_parameters':sum(p.numel() for p in bn.parameters())+sum(p.numel() for p in dec.parameters())}

def load_historical_weights(bn, dec, payload):
    """Only key renaming for equivalent GGGG+GELU historical block wrappers."""
    sd = payload['student_state_dict']
    remapped = {k.replace('.ffn.mlp.','.mlp.'):v for k,v in sd.items()}
    dec.load_state_dict(remapped,strict=True)
    bn.load_state_dict(payload['ocbe_state_dict'],strict=True)
