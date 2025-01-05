# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.utils import to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils.ckpt_convert import swin_converter
from ..utils.transformer import PatchEmbed, PatchMerging
from ..builder import NECKS


class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(init_cfg)

        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None)

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        # print(H)
        # print(W)
        # print(L)
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None):

        super(SwinBlock, self).__init__()

        self.init_cfg = init_cfg
        self.with_cp = with_cp

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=2,
            ffn_drop=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            act_cfg=act_cfg,
            add_identity=True,
            init_cfg=None)

    def forward(self, x, hw_shape):

        def _inner_forward(x):
            # Self-attention
            B, C, H, W = x.shape  # 假設 x 是四維的
            x = x.view(B, C, -1)  # 將空間維度展平，變成 (B, C, H*W)
            x = x.permute(0, 2, 1)  # 將形狀調整為 (B, H*W, C)，這樣每個像素成為序列中的一個位置

            identity = x
            x = self.norm1(x)
            x = self.attn(x, hw_shape)

            x = x + identity

            identity = x
            x = self.norm2(x)
            x = self.ffn(x, identity=identity)

            x = x.permute(0, 2, 1).view(B, C, H, W)  # 恢復到原始的空間維度

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x
    
# SwinTransformerBlock 使用多個 SwinTransformerLayer
class SwinTransformerBlock(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 depth,
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None,
                 in_channels=3,
                 mlp_ratio=4):
        super(SwinTransformerBlock, self).__init__()

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]

        # 構建多層 SwinTransformerLayer
        layers = []
        for i in range(depth):
            layers.append(
                SwinBlock(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * in_channels,
                    window_size=window_size,
                    shift=False if i % 2 == 0 else True,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rates[i],
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    init_cfg=None)
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x, hw_shape):
        for layer in self.layers:
            x = layer(x, hw_shape)
        return x

@NECKS.register_module()
class SwinNeck(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dims, num_heads, depths, window_size, patch_size=4, strides=(4, 2, 2, 2),
                 patch_norm=True,
                 norm_cfg=dict(type='LN'),):
        super(SwinNeck, self).__init__()
        self.up_merge = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 定義 Swin Transformer Blocks
        self.stage1 = SwinTransformerBlock(embed_dims=embed_dims * 8, num_heads=num_heads * 8, depth=depths[3], window_size=window_size, in_channels=in_channels[3])
        self.stage2 = SwinTransformerBlock(embed_dims=embed_dims * 4, num_heads=num_heads * 4, depth=depths[2], window_size=window_size, in_channels=in_channels[2])
        self.stage3 = SwinTransformerBlock(embed_dims=embed_dims * 2, num_heads=num_heads * 2, depth=depths[1], window_size=window_size, in_channels=in_channels[1])
        self.stage4 = SwinTransformerBlock(embed_dims=embed_dims, num_heads=num_heads, depth=depths[0], window_size=window_size, in_channels=in_channels[0])

        # 定義 LayerNorm
        self.norm = nn.LayerNorm(out_channels)

        # print('\n\n\n\n', embed_dims, '\n\n\n\n')

        # 定義跳躍連接的通道壓縮卷積
        self.compress_conv1 = nn.Conv2d(in_channels[3] + embed_dims * 8, embed_dims * 4, kernel_size=1)
        self.compress_conv2 = nn.Conv2d(in_channels[2] + embed_dims * 4, embed_dims * 2, kernel_size=1)
        self.compress_conv3 = nn.Conv2d(in_channels[1] + embed_dims * 2, embed_dims, kernel_size=1)
        self.compress_conv4 = nn.Conv2d(in_channels[0] + embed_dims, embed_dims, kernel_size=1)

        # 1x1卷積將通道數從 96 變為 256
        self.conv1 = nn.Conv2d(embed_dims * 4, out_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(embed_dims * 2, out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(embed_dims, out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(embed_dims, out_channels, kernel_size=1)

    def forward(self, features):

        # Backbone 特徵：C2, C3, C4, C5
        C2, C3, C4, C5 = features[0], features[1], features[2], features[3]

        B, C, H, W = features[3].shape  # 假設 x 是四維的
        hw_shape = (H, W)     
        
        # for i in features:
        #     b, c, h, w = i.shape
        #     print(b, c, h, w)

        # Stage 1: C5 與 Stage1 特徵拼接
        C5_stage = self.stage1(C5, hw_shape)
        C5_merged = torch.cat([C5, C5_stage], dim=1)  # 拼接 Backbone 特徵與 Neck 特徵
        C5_compressed = self.compress_conv1(C5_merged)

        # 更新 hw_shape (如果特徵圖形狀改變)
        hw_shape = (hw_shape[0] * 2, hw_shape[1] * 2)  # 假設上採樣倍數為2

        # Stage 2: 上採樣後與 C4 拼接
        C4_stage = self.stage2(self.up_merge(C5_compressed), hw_shape)
        C4_merged = torch.cat([C4, C4_stage], dim=1)
        C4_compressed = self.compress_conv2(C4_merged)

        # 更新 hw_shape
        hw_shape = (hw_shape[0] * 2, hw_shape[1] * 2)

        # Stage 3: 上採樣後與 C3 拼接
        C3_stage = self.stage3(self.up_merge(C4_compressed), hw_shape)
        C3_merged = torch.cat([C3, C3_stage], dim=1)
        C3_compressed = self.compress_conv3(C3_merged)

        # 更新 hw_shape
        hw_shape = (hw_shape[0] * 2, hw_shape[1] * 2)

        # Stage 4: 上採樣後與 C2 拼接
        C2_stage = self.stage4(self.up_merge(C3_compressed), hw_shape)
        C2_merged = torch.cat([C2, C2_stage], dim=1)
        C2_compressed = self.compress_conv4(C2_merged)

        # 轉換輸出通道到設置的輸出通道數量上
        x4 = self.conv1(C5_compressed)
        x3 = self.conv2(C4_compressed)
        x2 = self.conv3(C3_compressed)
        x1 = self.conv4(C2_compressed)

        xs = [x4, x3, x2, x1]
        output = []

        for x in xs:
            B, C, H, W = x.shape  # 假設 x 是四維的
            x = x.view(B, C, -1)  # 將空間維度展平，變成 (B, C, H*W)
            x = x.permute(0, 2, 1)  # 將形狀調整為 (B, H*W, C)，這樣每個像素成為序列中的一個位置

            # 最終輸出進行 LayerNorm
            x = self.norm(x)
            x = x.permute(0, 2, 1).view(B, C, H, W)  # 恢復到原始的空間維度
            output.append(x)

        # 返回多層特徵圖作為 tuple
        return output[0], output[1], output[2], output[3]
