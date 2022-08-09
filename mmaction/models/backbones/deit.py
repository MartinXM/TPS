# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from os import X_OK
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from ..builder import BACKBONES
from mmaction.utils import get_root_logger


from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from .vision_transformer import VisionTransformer
from .vision_transformer_tsm import VisionTransformer as TSMVisionTransformer
from .vision_transformer_bayershift import VisionTransformer as BSVisionTransformer
from .vision_transformer_btsm import VisionTransformer as BTSMVisionTransformer


from einops import rearrange

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'deit_tiny_distilled_patch16_224', 'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224', 'deit_base_patch16_384',
    'deit_base_distilled_patch16_384',
]

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (weights from official Google JAX impl)
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_base_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch8_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),

    # patch models, imagenet21k (weights from official Google JAX impl)
    'vit_tiny_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch8_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        num_classes=21843),
    'vit_huge_patch14_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843),

    # SAM trained models (https://arxiv.org/abs/2106.01548)
    'vit_base_patch32_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz'),
    'vit_base_patch16_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz'),

    # deit models (FB weights)
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0),
    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    # ViT ImageNet-21K-P pretraining by MILL
    'vit_base_patch16_224_miil_in21k': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    'vit_base_patch16_224_miil': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm'
            '/vit_base_patch16_224_1k_miil_84_4.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear',
    ),
}



class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]
    
    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_small_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_224(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    model = DistilledVisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model



class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=(2,4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        return x

@BACKBONES.register_module()
class DeiT3D(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=192,
                 depths=12,
                 num_heads=3,
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        # self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed3D(
            # patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            # norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        # self.layers = nn.ModuleList()
        # for i_layer in range(self.num_layers):
        #     layer = BasicLayer(
        #         dim=int(embed_dim * 2**i_layer),
        #         depth=depths[i_layer],
        #         num_heads=num_heads[i_layer],
        #         window_size=window_size,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         qk_scale=qk_scale,
        #         drop=drop_rate,
        #         attn_drop=attn_drop_rate,
        #         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
        #         norm_layer=norm_layer,
        #         downsample=PatchMerging if i_layer<self.num_layers-1 else None,
        #         use_checkpoint=use_checkpoint)
        #     self.layers.append(layer)


        self.model = VisionTransformer(
            patch_size=patch_size, embed_dim=embed_dim, depth=depths, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            norm_layer=partial(norm_layer, eps=1e-6) ) #, **kwargs)
        self.model.default_cfg = _cfg()
        if pretrained2d:
            if embed_dim == 192:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                    map_location="cpu", check_hash=True
                )
            elif embed_dim == 384:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                    map_location="cpu", check_hash=True
                )
            elif embed_dim == 768:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu", check_hash=True
                )
        # model.load_state_dict(checkpoint["model"])
        # print('model:',checkpoint["model"].keys())
        # print('self.state:',self.state_dict().keys())

        for i in checkpoint["model"]:
            # print(i)
            if 'model.'+i in self.state_dict().keys():
                self.state_dict()['model.'+i].copy_(checkpoint["model"][i])

        # self.model = models


        # self.num_features = int(embed_dim * 2**(self.num_layers-1))

        # add a norm layer for each output
        # self.norm = norm_layer(self.num_features)
        # self.avg = nn.AdaptiveAvgPool2d(1)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,self.patch_size[0],1,1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2*self.window_size[1]-1) * (2*self.window_size[2]-1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2*self.window_size[1]-1, 2*self.window_size[2]-1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            pass
            # self.apply(_init_weights)
            # logger = get_root_logger()
            # logger.info(f'load model from: {self.pretrained}')

            # if self.pretrained2d:
                # Inflate 2D model into 3D model.
                # self.inflate_weights(logger)
            # else:
                # Directly load 3D model.
                # load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        # x = self.patch_embed(x)

        # x = self.pos_drop(x)

        # for layer in self.layers:
            # x = layer(x.contiguous())

        n, c, d, h, w = x.size()
        # print('x.input_ before reshape:', x.shape)

        x = rearrange(x, 'n c d h w -> n d c h w')
        x = x.reshape(n*d,c,h,w)
        # x = self.model(x)
        ## DeiT
        # print('x.input:', x.shape)
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # if self.model.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
        # else:
            # x = torch.cat((cls_token, self.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        for block in self.model.blocks:
            x = block(x)
        x = self.model.norm(x)

        # feat = x[:,1:]
        # feat = feat.permute(0,2,1).reshape(n*d,-1,14,14)
        # feat = self.avg(feat)
        # feat = torch.cat((x[:,0].unsqueeze(2),feat.squeeze(3)),2)
        feat = x[:,0].unsqueeze(2)
        # nn, pp, cc = x.shape
        # feat = x.permute(0,2,1).reshape(n,d,cc,pp).permute(0,2,1,3)
        # feat = feat.reshape(n,d,-1,2).permute(0,2,1,3)
        feat = feat.reshape(n,d,-1,1).permute(0,2,1,3)

        
        # print('after pass:', feat.shape)
        # x = rearrange(x, 'n c d h w -> n d h w c')
        # x = self.norm(x)
        # x = rearrange(x, 'n d h w c -> n c d h w')

        return feat

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DeiT3D, self).train(mode)
        self._freeze_stages()

@BACKBONES.register_module()
class DeiT3D_TSM(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=192,
                 depths=12,
                 num_heads=3,
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        # self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed3D(
            # patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            # norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        # self.layers = nn.ModuleList()
        # for i_layer in range(self.num_layers):
        #     layer = BasicLayer(
        #         dim=int(embed_dim * 2**i_layer),
        #         depth=depths[i_layer],
        #         num_heads=num_heads[i_layer],
        #         window_size=window_size,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         qk_scale=qk_scale,
        #         drop=drop_rate,
        #         attn_drop=attn_drop_rate,
        #         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
        #         norm_layer=norm_layer,
        #         downsample=PatchMerging if i_layer<self.num_layers-1 else None,
        #         use_checkpoint=use_checkpoint)
        #     self.layers.append(layer)


        self.model = TSMVisionTransformer(
            patch_size=patch_size, embed_dim=embed_dim, depth=depths, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            norm_layer=partial(norm_layer, eps=1e-6) ) #, **kwargs)
        self.model.default_cfg = _cfg()
        if pretrained2d:
            if embed_dim == 192:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                    map_location="cpu", check_hash=True
                )
            elif embed_dim == 384:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                    map_location="cpu", check_hash=True
                )
            elif embed_dim == 768:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu", check_hash=True
                )
        # model.load_state_dict(checkpoint["model"])
        # print('model:',checkpoint["model"].keys())
        # print('self.state:',self.state_dict().keys())

        for i in checkpoint["model"]:
            # print(i)
            if 'model.'+i in self.state_dict().keys():
                self.state_dict()['model.'+i].copy_(checkpoint["model"][i])

        # self.model = models

        self.pos_embed = nn.Parameter(self.model.pos_embed.repeat(8,1,1))

        # self.num_features = int(embed_dim * 2**(self.num_layers-1))

        # add a norm layer for each output
        # self.norm = norm_layer(self.num_features)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self._freeze_stages()


        # for i, b in enumerate(self.model.blocks):
            # self.model.blocks[i] = TemporalShift(b, n_segment=8, n_div=num_heads)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,self.patch_size[0],1,1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2*self.window_size[1]-1) * (2*self.window_size[2]-1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2*self.window_size[1]-1, 2*self.window_size[2]-1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            pass
            # self.apply(_init_weights)
            # logger = get_root_logger()
            # logger.info(f'load model from: {self.pretrained}')

            # if self.pretrained2d:
                # Inflate 2D model into 3D model.
                # self.inflate_weights(logger)
            # else:
                # Directly load 3D model.
                # load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        # x = self.patch_embed(x)

        # x = self.pos_drop(x)

        # for layer in self.layers:
            # x = layer(x.contiguous())

        n, c, d, h, w = x.size()
        # print('x.input_ before reshape:', x.shape)

        x = rearrange(x, 'n c d h w -> n d c h w')
        x = x.reshape(n*d,c,h,w)
        # x = self.model(x)
        ## DeiT
        # print('x.input:', x.shape)
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # if self.model.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
        # else:
            # x = torch.cat((cls_token, self.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.model.pos_drop(x + self.pos_embed.repeat(n,1,1))
        for block in self.model.blocks:
            x = block(x)
        x = self.model.norm(x)

        # feat = x[:,1:]
        # feat = feat.permute(0,2,1).reshape(n*d,-1,14,14)
        # feat = self.avg(feat)
        # feat = torch.cat((x[:,0].unsqueeze(2),feat.squeeze(3)),2)
        # feat = feat.reshape(n,d,-1,2).permute(0,2,1,3)

        feat = x[:,0].unsqueeze(2)
        feat = feat.reshape(n,d,-1,1).permute(0,2,1,3)

        
        # print('after pass:', feat.shape)
        # x = rearrange(x, 'n c d h w -> n d h w c')
        # x = self.norm(x)
        # x = rearrange(x, 'n d h w c -> n c d h w')

        return feat

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DeiT3D_TSM, self).train(mode)
        self._freeze_stages()



class TemporalShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(TemporalShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, p, c  = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, p, c)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            ## class token only
            out[:, :-1, :, :fold] = x[:, 1:, :, :fold]  # shift left
            out[:, 1:, :, fold: 2 * fold] = x[:, :-1, :, fold: 2 * fold]  # shift right
            out[:, :, :, 2 * fold:] = x[:, :, :, 2 * fold:]  # not shift

        return out.view(nt, p, c)

class BayerShift(nn.Module):
    def __init__(self, net, n_segment=3, n_div=8, inplace=False):
        super(BayerShift, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.fold_div = n_div
        self.inplace = inplace
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using fold div: {}'.format(self.fold_div))

    def forward(self, x):
        x = self.shift(x, self.n_segment, fold_div=self.fold_div, inplace=self.inplace)
        return self.net(x)

    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, p, c  = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, p, c)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            ## class token only
            ## x size: B, T, N, C
            # B, T, N, C = x.size()
            feat = x[:,:,1:,:]
            feat = feat.view(n_batch, n_segment, 14, 14, c)
            # out = torch.zeros_like(feat)
            out = feat[:,:,:,:,:]
            # x0 = feat[:, :, 0::2, 0::2, :]  # B H/2 W/2 C
            # x1 = feat[:, :, 1::2, 0::2, :]  # B H/2 W/2 C
            # x2 = feat[:, :, 0::2, 1::2, :]  # B H/2 W/2 C
            # x3 = feat[:, :, 1::2, 1::2, :]  # B H/2 W/2 C
            out[:, :-1, 0::2, 0::2, :fold] = feat[:, 1:, 0::2, 0::2, :fold]  # top left
            out[:, 1:, 1::2, 1::2, :fold] = feat[:, :-1, 1::2, 1::2, :fold]  # bottom right
            out[:, :, 1::2, 0::2, :fold] = feat[:, :, 1::2, 0::2, :fold]  # rest
            out[:, :, 0::2, 1::2, :fold] = feat[:, :, 0::2, 1::2, :fold]  # rest
            out[:, :, :, :, fold:] = feat[:, :, :, :, fold:]  # rest
            # out[:, :, :, :, fold:] = feat[:, :, :, :, fold:]  # rest
            out = out.view(n_batch, n_segment, 14*14, c)
            out = torch.cat((x[:,:,0,:].unsqueeze(2),out),dim=2)
            # out[:, :, :, 2 * fold:] = x[:, :, :, 2 * fold:]  # not shift

        return out.view(nt, p, c)  

@BACKBONES.register_module()
class DeiT3D_2plus1D(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=192,
                 depths=12,
                 num_heads=3,
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        # self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed3D(
            # patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            # norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        # self.layers = nn.ModuleList()
        # for i_layer in range(self.num_layers):
        #     layer = BasicLayer(
        #         dim=int(embed_dim * 2**i_layer),
        #         depth=depths[i_layer],
        #         num_heads=num_heads[i_layer],
        #         window_size=window_size,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         qk_scale=qk_scale,
        #         drop=drop_rate,
        #         attn_drop=attn_drop_rate,
        #         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
        #         norm_layer=norm_layer,
        #         downsample=PatchMerging if i_layer<self.num_layers-1 else None,
        #         use_checkpoint=use_checkpoint)
        #     self.layers.append(layer)


        self.model = VisionTransformer(
            patch_size=patch_size, embed_dim=embed_dim, depth=depths, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            norm_layer=partial(norm_layer, eps=1e-6) ) #, **kwargs)
        self.model.default_cfg = _cfg()
        if pretrained2d:
            if embed_dim == 192:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                    map_location="cpu", check_hash=True
                )
            elif embed_dim == 384:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                    map_location="cpu", check_hash=True
                )
            elif embed_dim == 768:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu", check_hash=True
                )
        # model.load_state_dict(checkpoint["model"])
        # print('model:',checkpoint["model"].keys())
        # print('self.state:',self.state_dict().keys())

        for i in checkpoint["model"]:
            # print(i)
            if 'model.'+i in self.state_dict().keys():
                self.state_dict()['model.'+i].copy_(checkpoint["model"][i])

        # self.model = models


        # self.num_features = int(embed_dim * 2**(self.num_layers-1))

        # add a norm layer for each output
        # self.norm = norm_layer(self.num_features)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self._freeze_stages()


        for i, b in enumerate(self.model.blocks):
            self.model.blocks[i] = Temporal1D(b, n_segment=8, in_channels=self.model.embed_dim)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,self.patch_size[0],1,1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2*self.window_size[1]-1) * (2*self.window_size[2]-1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2*self.window_size[1]-1, 2*self.window_size[2]-1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            pass
            # self.apply(_init_weights)
            # logger = get_root_logger()
            # logger.info(f'load model from: {self.pretrained}')

            # if self.pretrained2d:
                # Inflate 2D model into 3D model.
                # self.inflate_weights(logger)
            # else:
                # Directly load 3D model.
                # load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        # x = self.patch_embed(x)

        # x = self.pos_drop(x)

        # for layer in self.layers:
            # x = layer(x.contiguous())

        n, c, d, h, w = x.size()
        # print('x.input_ before reshape:', x.shape)

        x = rearrange(x, 'n c d h w -> n d c h w')
        x = x.reshape(n*d,c,h,w)
        # x = self.model(x)
        ## DeiT
        # print('x.input:', x.shape)
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # if self.model.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
        # else:
            # x = torch.cat((cls_token, self.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        for block in self.model.blocks:
            x = block(x)
        x = self.model.norm(x)

        feat = x[:,1:]
        feat = feat.permute(0,2,1).reshape(n*d,-1,14,14)
        feat = self.avg(feat)
        feat = torch.cat((x[:,0].unsqueeze(2),feat.squeeze(3)),2)
        # nn, pp, cc = x.shape
        # feat = x.permute(0,2,1).reshape(n,d,cc,pp).permute(0,2,1,3)
        feat = feat.reshape(n,d,-1,2).permute(0,2,1,3)

        
        # print('after pass:', feat.shape)
        # x = rearrange(x, 'n c d h w -> n d h w c')
        # x = self.norm(x)
        # x = rearrange(x, 'n d h w c -> n c d h w')

        return feat

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DeiT3D_2plus1D, self).train(mode)
        self._freeze_stages()



class Temporal1D(nn.Module):
    def __init__(self, net, n_segment=3, in_channels=192, inplace=False):
        super(Temporal1D, self).__init__()
        self.net = net
        self.n_segment = n_segment
        self.inplace = inplace
        self.conv = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        if inplace:
            print('=> Using in-place shift...')
        print('=> Using Temporal 1D Conv with inchannel: {}'.format(in_channels))

    def forward(self, x):
        nt, p, c  = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, p, c).permute(0,2,3,1)
        x = x.reshape(n_batch*p, c, self.n_segment)
        x = self.conv(x)
        x = x.view(n_batch, p, c, self.n_segment)
        x = x.permute(0,3,1,2).reshape(nt, p, c)
        return self.net(x)



@BACKBONES.register_module()
class DeiT3D_BayerShift(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=192,
                 depths=12,
                 num_heads=3,
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        # self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed3D(
            # patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            # norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        # self.layers = nn.ModuleList()
        # for i_layer in range(self.num_layers):
        #     layer = BasicLayer(
        #         dim=int(embed_dim * 2**i_layer),
        #         depth=depths[i_layer],
        #         num_heads=num_heads[i_layer],
        #         window_size=window_size,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         qk_scale=qk_scale,
        #         drop=drop_rate,
        #         attn_drop=attn_drop_rate,
        #         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
        #         norm_layer=norm_layer,
        #         downsample=PatchMerging if i_layer<self.num_layers-1 else None,
        #         use_checkpoint=use_checkpoint)
        #     self.layers.append(layer)


        self.model = BSVisionTransformer(
            patch_size=patch_size, embed_dim=embed_dim, depth=depths, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            norm_layer=partial(norm_layer, eps=1e-6) ) #, **kwargs)
        self.model.default_cfg = _cfg()
        if pretrained2d:
            if embed_dim == 192:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                    map_location="cpu", check_hash=True
                )
            elif embed_dim == 384:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                    map_location="cpu", check_hash=True
                )
            elif embed_dim == 768:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu", check_hash=True
                )
        # model.load_state_dict(checkpoint["model"])
        # print('model:',checkpoint["model"].keys())
        # print('self.state:',self.state_dict().keys())

        for i in checkpoint["model"]:
            # print(i)
            if 'model.'+i in self.state_dict().keys():
                self.state_dict()['model.'+i].copy_(checkpoint["model"][i])
        

        self.pos_embed = nn.Parameter(self.model.pos_embed.repeat(8,1,1))
        # self.pos_embed = self.model.pos_embed.repeat(8,1,1)


        # self.model = models


        # self.num_features = int(embed_dim * 2**(self.num_layers-1))

        # add a norm layer for each output
        # self.norm = norm_layer(self.num_features)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self._freeze_stages()


        # for i, b in enumerate(self.model.blocks):
            # if i > 5:
            # self.model.blocks[i] = BayerShift(b, n_segment=8, n_div=num_heads)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,self.patch_size[0],1,1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2*self.window_size[1]-1) * (2*self.window_size[2]-1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2*self.window_size[1]-1, 2*self.window_size[2]-1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            pass
            # self.apply(_init_weights)
            # logger = get_root_logger()
            # logger.info(f'load model from: {self.pretrained}')

            # if self.pretrained2d:
                # Inflate 2D model into 3D model.
                # self.inflate_weights(logger)
            # else:
                # Directly load 3D model.
                # load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        # x = self.patch_embed(x)

        # x = self.pos_drop(x)

        # for layer in self.layers:
            # x = layer(x.contiguous())
        if len(x.size()) == 5:
            n, c, d, h, w = x.size()
            # print('x.input_ before reshape:', x.shape)

            x = rearrange(x, 'n c d h w -> n d c h w')
            x = x.reshape(n*d,c,h,w)
            sample_flag = False
        else:
            n, c, h, w = x.size()
            sample_flag = True
        # x = self.model(x)
        ## DeiT
        # print('x.input:', x.shape)
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # if self.model.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
        # else:
            # x = torch.cat((cls_token, self.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        # print(x.shape, self.pos_embed.shape, n, d)
        x = self.model.pos_drop(x + self.pos_embed.repeat(n,1,1))
        for block in self.model.blocks:
            x = block(x)
        x = self.model.norm(x)

        # feat = x[:,1:]
        # feat = feat.permute(0,2,1).reshape(n*d,-1,14,14)
        # feat = self.avg(feat)
        # feat = torch.cat((x[:,0].unsqueeze(2),feat.squeeze(3)),2)
        # feat = feat.reshape(n,d,-1,2).permute(0,2,1,3)

        if sample_flag:
            feat = x[:,0].view(n//8,8,-1).unsqueeze(3).permute(0,2,1,3)
        else:
            feat = x[:,0].unsqueeze(2)
            feat = feat.reshape(n,d,-1,1).permute(0,2,1,3)

        
        # print('after pass:', feat.shape)
        # x = rearrange(x, 'n c d h w -> n d h w c')
        # x = self.norm(x)
        # x = rearrange(x, 'n d h w c -> n c d h w')

        return feat

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DeiT3D_BayerShift, self).train(mode)
        self._freeze_stages()


@BACKBONES.register_module()
class DeiT3D_BTSM(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: (4,4,4).
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: Truee
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer: Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=192,
                 depths=12,
                 num_heads=3,
                 window_size=(2,7,7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        # self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        # self.patch_embed = PatchEmbed3D(
            # patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            # norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        # self.layers = nn.ModuleList()
        # for i_layer in range(self.num_layers):
        #     layer = BasicLayer(
        #         dim=int(embed_dim * 2**i_layer),
        #         depth=depths[i_layer],
        #         num_heads=num_heads[i_layer],
        #         window_size=window_size,
        #         mlp_ratio=mlp_ratio,
        #         qkv_bias=qkv_bias,
        #         qk_scale=qk_scale,
        #         drop=drop_rate,
        #         attn_drop=attn_drop_rate,
        #         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
        #         norm_layer=norm_layer,
        #         downsample=PatchMerging if i_layer<self.num_layers-1 else None,
        #         use_checkpoint=use_checkpoint)
        #     self.layers.append(layer)


        self.model = BTSMVisionTransformer(
            patch_size=patch_size, embed_dim=embed_dim, depth=depths, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            norm_layer=partial(norm_layer, eps=1e-6) ) #, **kwargs)
        self.model.default_cfg = _cfg()
        if pretrained2d:
            if embed_dim == 192:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
                    map_location="cpu", check_hash=True
                )
            elif embed_dim == 384:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
                    map_location="cpu", check_hash=True
                )
            elif embed_dim == 768:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu", check_hash=True
                )
        # model.load_state_dict(checkpoint["model"])
        # print('model:',checkpoint["model"].keys())
        # print('self.state:',self.state_dict().keys())

        for i in checkpoint["model"]:
            # print(i)
            if 'model.'+i in self.state_dict().keys():
                self.state_dict()['model.'+i].copy_(checkpoint["model"][i])
        

        self.pos_embed = nn.Parameter(self.model.pos_embed.repeat(8,1,1))
        # self.pos_weight = nn.Parameter(torch.zeros(depths))
        # self.pos_embed = self.model.pos_embed.repeat(8,1,1)


        # self.model = models


        # self.num_features = int(embed_dim * 2**(self.num_layers-1))

        # add a norm layer for each output
        # self.norm = norm_layer(self.num_features)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self._freeze_stages()


        # for i, b in enumerate(self.model.blocks):
            # if i > 5:
            # self.model.blocks[i] = BayerShift(b, n_segment=8, n_div=num_heads)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """Inflate the swin2d parameters to swin3d.

        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.

        Args:
            logger (logging.Logger): The logger used to print
                debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location='cpu')
        state_dict = checkpoint['model']

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict['patch_embed.proj.weight'] = state_dict['patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,self.patch_size[0],1,1) / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            relative_position_bias_table_current = self.state_dict()[k]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            L2 = (2*self.window_size[1]-1) * (2*self.window_size[2]-1)
            wd = self.window_size[0]
            if nH1 != nH2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if L1 != L2:
                    S1 = int(L1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2*self.window_size[1]-1, 2*self.window_size[2]-1),
                        mode='bicubic')
                    relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            pass
            # self.apply(_init_weights)
            # logger = get_root_logger()
            # logger.info(f'load model from: {self.pretrained}')

            # if self.pretrained2d:
                # Inflate 2D model into 3D model.
                # self.inflate_weights(logger)
            # else:
                # Directly load 3D model.
                # load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        # x = self.patch_embed(x)

        # x = self.pos_drop(x)

        # for layer in self.layers:
            # x = layer(x.contiguous())
        if len(x.size()) == 5:
            n, c, d, h, w = x.size()
            # print('x.input_ before reshape:', x.shape)

            x = rearrange(x, 'n c d h w -> n d c h w')
            x = x.reshape(n*d,c,h,w)
            sample_flag = False
        else:
            n, c, h, w = x.size()
            sample_flag = True
        # x = self.model(x)
        ## DeiT
        # print('x.input:', x.shape)
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # if self.model.dist_token is None:
        x = torch.cat((cls_token, x), dim=1)
        # else:
            # x = torch.cat((cls_token, self.model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        # print(x.shape, self.pos_embed.shape, n, d)
        x = self.model.pos_drop(x + self.pos_embed.repeat(n,1,1))
        # length = len(self.model.blocks)
        for ind, block in enumerate(self.model.blocks):
            x = block(x)
            # if ind == len(self.model.blocks)-1:
                # x = block(x)
            # else:
                # x = block(x) + self.pos_embed.repeat(n,1,1)
        x = self.model.norm(x)

        # feat = x[:,1:]
        # feat = feat.permute(0,2,1).reshape(n*d,-1,14,14)
        # feat = self.avg(feat)
        # feat = torch.cat((x[:,0].unsqueeze(2),feat.squeeze(3)),2)
        # feat = feat.reshape(n,d,-1,2).permute(0,2,1,3)

        if sample_flag:
            feat = x[:,0].view(n//8,8,-1).unsqueeze(3).permute(0,2,1,3)
        else:
            feat = x[:,0].unsqueeze(2)
            feat = feat.reshape(n,d,-1,1).permute(0,2,1,3)

        
        # print('after pass:', feat.shape)
        # x = rearrange(x, 'n c d h w -> n d h w c')
        # x = self.norm(x)
        # x = rearrange(x, 'n d h w c -> n c d h w')

        return feat

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(DeiT3D_BTSM, self).train(mode)
        self._freeze_stages()