import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("/data/gsw/Code/Surgformer")
import utils
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from collections import OrderedDict
import math
from torch.autograd import Function
from torch.nn import LayerNorm as norm_layer
from torch.cuda.amp import autocast


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        num_frames=1,
        num_classes=7
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (num_frames)
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(patch_size[0], patch_size[1]),
            stride=(patch_size[0], patch_size[1]),
        )
        self.norm = norm_layer(embed_dim)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # 直接使用3D卷积来映射时序帧到视频序列tokens，在过程中进行Temporal Sample
        # 对于逐帧计算的Tool以及Phase，怎么处理模型结构的变化？降低视频序列长度并且放弃时序采样

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}
    
    def get_num_layers(self):
        return sum(1 for _ in self.children())


    def forward(self, x):

        
        #x=x.unsqueeze(2)
        B, C, H, W = x.shape
        #x = rearrange(x, "b c t h w -> (b t) c h w")
    
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        with autocast():
            x = self.proj(x).flatten(2)
        x = rearrange(x, "b c k -> b k c", b=B)
        cls_tokens = self.cls_token.expand(B,-1,-1)
        x = torch.cat((cls_tokens, x), dim=1)
        self.norm(x)
        x = x[:,0]
        class_output = self.head(x)
        return class_output,x

@register_model
class VIT(nn.Module):
    def __init__(
        self,
        pretrained,
        pretrain_path,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # For consistency with other models

        # Patch Embedding
        self.patch_embed = PatchEmbeding(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        if pretrained:
            npz_checkpoint = np.load(pretrain_path)
            state_dict_npz = {}
            for key,value in npz_checkpoint.items():
                new_key = key
                new_key = new_key.replace('Transoformer/encoderblock','blocks')
                n
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x):
        with autocast():
            x = self.patch_embed(x)  # Patch embedding
        B, N, C = x.size()

        # Add class token and positional embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Pass through Transformer Blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        
        # Remove class token and reshape back to spatial dimensions
        x = x[:, 1:]  # Remove class token
        H = W = int(math.sqrt(N))  # Assuming square input
        x = x.transpose(1, 2).reshape(B, C, H, W)  # Reshape to (B, C, H, W)
        
        return x

    def get_num_layers(self):
        return len(self.blocks)

    def forward(self, x):
        # x的输入形状为 (B, C, H, W)
        features = x  # 直接使用原始特征
        # 对特征进行全局平均池化用于分类任务
        pooled_features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        class_output = self.head(pooled_features)
        return class_output, features  # 返回分类结果和原始空间维度的特征图


class PatchEmbeding(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # Input: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Block(nn.Module):
    """Transformer Block"""

    def __init__(
        self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
        drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    """Multi-head Self Attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """Feedforward Network"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Stochastic Depth (Drop Path)"""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x / keep_prob * random_tensor

