"""教师消融专用:MADViT 的 DeiT 教师(deit_base_distilled_patch16_224)纯提取器。

背景:组会质疑"RD 的性能增益是不是只来自把教师换成了 DINOv3"。本文件提供一个与
model/teacher.py 的 DINOv3Teacher **接口完全镜像** 的 DeiT 教师,使 train.py/eval.py 里的
OCBE / student / top-k loss / top-k readout / taps / 配方 / 协议 **逐字节不变**,唯一变量 =
backbone 预训练(DeiT:ImageNet 监督;DINOv3:1.7B 自监督)。两者都是 ViT-B/16、embed_dim=768、
12 blocks、~86M —— **容量对等**,所以增益若消失即归因于"教师"、若仍在即归因于"方法"。

权重型号取自 MAD-ViT 复现(/code_experiment/MAD-ViT/code/train_val_VIT_Distillation.py:201/538、
repro/code/mvtrain.py 等):`deit_base_distilled_patch16_224`,timm pretrained。MADViT 代码里那个
latent_dim=192 的随机冻结投影是它**自己方法**的一部分(为对齐 deit_tiny 学生),**不属于"教师权重"**,
此处不引入 —— 本提取器返回 backbone 的 RAW 768 维 block 特征,维度对齐仍由 RD 的 OCBE/student 在
学生侧完成(与 DINOv3 口径一致,见 teacher.py 的 design contract)。

Token 布局:DeiT-distilled num_prefix_tokens=2([CLS, DIST, patch...]),patch = tokens[:, 2:]。
DeiT 用 learned 绝对位置编码(非 RoPE),分辨率固定 **224** → 14×14 = 196 patches,不可改分辨率
(改了 pos_embed 长度不匹配)。embed_dim 768 与 DINOv3 一致 → OCBE 输入维不变。Env: base(timm 1.0.20)。
"""
import torch
import torch.nn as nn
import timm

DEIT_NAME = 'deit_base_distilled_patch16_224'   # MADViT 教师 backbone(见 MAD-ViT 复现)


class DeiTTeacher(nn.Module):
    """FROZEN DeiT-base-distilled 提取器,接口镜像 model.teacher.DINOv3Teacher(零可训练参数)。"""

    def __init__(self, layers=None, pretrained=True):
        super().__init__()
        self.vit = timm.create_model(DEIT_NAME, pretrained=pretrained)
        self.vit.eval()
        for p in self.vit.parameters():
            p.requires_grad_(False)

        self.embed_dim = self.vit.embed_dim                  # 768
        self.num_blocks = len(self.vit.blocks)               # 12
        self.num_prefix = self.vit.num_prefix_tokens         # 2 (cls + distill)
        self.num_register = 0                                # DeiT 无 register token
        self.layers = list(range(self.num_blocks)) if layers is None else list(layers)

        # 预处理从 pretrained_cfg 读(ImageNet mean/std),不硬编码
        cfg = self.vit.pretrained_cfg
        self.img_mean = tuple(cfg.get('mean', (0.485, 0.456, 0.406)))
        self.img_std = tuple(cfg.get('std', (0.229, 0.224, 0.225)))
        self.input_size = tuple(cfg.get('input_size', (3, 224, 224)))

        self._feats = {}
        self._handles = [self.vit.blocks[li].register_forward_hook(self._mk_hook(li))
                         for li in self.layers]

    def _mk_hook(self, li):
        def hook(_m, _inp, out):
            self._feats[li] = out
        return hook

    def train(self, mode=True):
        # 冻结教师必须恒 eval,无论父模块 train() 调用
        return super().train(False)

    @torch.no_grad()
    def forward(self, x):
        """x: (B,3,224,224) 已用 self.img_mean/std 归一化。返回 RAW(未归一)特征,dict 结构同 DINOv3Teacher:
          'patches' : list over self.layers, each (B, 196, 768) — 逐 block patch tokens
          'cls'     : (B, 768)                                  — 最终(post-norm)CLS token
          'tokens'  : list over self.layers, each (B, 198, 768) — 完整 raw block 输出
          'layers'  : tapped block 索引
        L2 归一化/温度/top-k 由 loss/readout 负责(学生侧对齐)。"""
        self._feats.clear()
        ff = self.vit.forward_features(x)                    # (B, 198, 768) post-final-norm
        tokens = [self._feats[li] for li in self.layers]     # raw per-block (B, 198, 768)
        patches = [t[:, self.num_prefix:] for t in tokens]   # (B, 196, 768)
        cls = ff[:, 0]                                        # (B, 768)
        return {'patches': patches, 'cls': cls, 'tokens': tokens, 'layers': list(self.layers)}


def build_deit_teacher(layers=None, pretrained=True):
    """工厂:冻结 MADViT DeiT 教师。`layers` = 要 tap 的 block 索引(消融默认与 baseline 同 [5,7,9,11])。"""
    return DeiTTeacher(layers=layers, pretrained=pretrained)
