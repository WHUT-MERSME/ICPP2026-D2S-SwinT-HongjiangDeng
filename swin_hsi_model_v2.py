"""
基于swin_tiny_patch4_window7_224的高光谱图像分类模型
该模型针对高光谱图像特点进行了适配，包括特殊的patch提取和嵌入方式
"""

import math
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import Batch_Mlp, batch_PatchEmbed, PatchEmbed, Mlp, DropPath, ClassifierHead, modified_ClassifierHead, to_2tuple, to_ntuple, trunc_normal_, \
    _assert, use_fused_attn, resize_rel_pos_bias_table, resample_patch_embed
from timm.models._builder import build_model_with_cfg
from timm.models._features_fx import register_notrace_function
from timm.models._manipulate import checkpoint_seq, named_apply
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations
from timm.models.vision_transformer import get_init_weights_vit

from ECMT_uitls.trans_utils import MyTestPlace,MyAt

_int_or_tuple_2_t = Union[int, Tuple[int, int]]

def window_partition(
        x: torch.Tensor,
        window_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0],
               W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size[0], window_size[1], C)
    return windows


@register_notrace_function  # reason: int argument is a Proxy
def window_reverse(windows, window_size: Tuple[int, int], H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W //
                     window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x

def get_relative_position_index(win_h: int, win_w: int):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid(
        [torch.arange(win_h), torch.arange(win_w)]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - \
        coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(
        1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_coords[:, :, 0] += win_h - 1  # shift to start from 0
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)  # Wh*Ww, Wh*Ww


def SpikeSim_Energy(flop, time_steps, ratio,potential_ratio,weight_num):
    N_rd = flop
    N_neuron = flop/weight_num
    E_ad = 0.9 ##0.03
    E_mul = 3.7 ##0.2
    E_mem = 5.00#1.25
    E_rd = N_rd* (E_mem*33/32)*ratio
    E_acc = N_rd * E_ad*ratio

    E_state = (E_mem + E_mul + E_ad + E_ad + E_ad + E_mem)*N_neuron*potential_ratio
    E_offmap = (E_mem/32)*N_neuron*potential_ratio
    E_snn = E_rd + E_acc + E_state + E_offmap
    return E_snn

def ANN_Energy(flop, weight_num,sparsity):
    N_rd = flop
    N_neuron = flop/weight_num
    E_ad = 0.9 ##0.03
    E_mul = 3.7 ##0.2
    E_mem = 5.00#1.25

    E_rd = N_rd* (E_mem*2) *sparsity
    E_acc = N_rd * (E_ad+E_mul) *sparsity
    E_offmap = (E_mem)*N_neuron

    E_snn = E_rd + E_acc +  E_offmap

    return E_snn

class PatchExtract(nn.Module):
    """从输入图像中提取固定大小的patches

    Args:
        patch_size (tuple): patch的高度和宽度
    """
    def __init__(self, patch_size=(2, 2)):
        super().__init__()
        self.patch_size_x = patch_size[0]
        self.patch_size_y = patch_size[1]

    def forward(self, x):
        # if x is None:
        #     # 当输入为None时，不进行任何操作但仍返回None
        #     return None
        # 从输入图像中提取patches
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size_x, self.patch_size_x).\
            unfold(3, self.patch_size_y, self.patch_size_y)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, -1, self.patch_size_x * self.patch_size_y * C)
        return patches

class PatchEmbedding(nn.Module):
    """将提取的patches嵌入到高维特征空间

    Args:
        num_patches (int): patches的数量
        patch_dim (int): 每个patch的维度
        embed_dim (int): 嵌入空间的维度
    """
    def __init__(self, num_patches, patch_dim, embed_dim):
        super().__init__()
        self.num_patches = num_patches
        self.proj = nn.Linear(patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.norm = nn.BatchNorm1d(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        # if x is None:
        #     # 当输入为None时，不进行任何操作但仍返回None
        #     return None
        # 线性投影、位置编码和归一化
        x = self.proj(x)
        x = x + self.pos_embed
        x = self.norm(x)
        return x

class SelectAdaptivePool2d(nn.Module):
    """自定义的自适应池化层，与model-2.log中的结构完全一致

    Args:
        pool_type (str): 池化类型，默认为'avg'
        flatten (nn.Module): 展平操作
    """
    def __init__(self, pool_type='avg', flatten=None):
        super().__init__()
        self.pool_type = pool_type
        self.flatten = flatten

    def __repr__(self):
        """自定义打印方式，确保与model-2.log中的格式一致"""
        return f"SelectAdaptivePool2d (pool_type={self.pool_type}, flatten={self.flatten.__class__.__name__})"

    def forward(self, x):
        # 应用自适应平均池化
        if x is None:
            # 当输入为None时，返回None给pool_if神经元处理
            return None

        if self.pool_type == 'avg':
            x = nn.functional.adaptive_avg_pool2d(x, 1)

        # 如果提供了flatten操作，则应用它
        if self.flatten is not None:
            x = self.flatten(x)
        return x


class WindowAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports shifted and non-shifted windows.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int,
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            qkv_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
    ):
        """
        Args:
            dim: Number of input channels.
            num_heads: Number of attention heads.
            head_dim: Number of channels per head (dim // num_heads if not set)
            window_size: The height and width of the window.
            qkv_bias:  If True, add a learnable bias to query, key, value.
            attn_drop: Dropout ratio of attention weight.
            proj_drop: Dropout ratio of output.
        """
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)  # Wh, Ww
        win_h, win_w = self.window_size
        self.window_area = win_h * win_w
        self.num_heads = num_heads
        head_dim = head_dim or dim // num_heads
        attn_dim = head_dim * num_heads
        self.attn_dim = attn_dim
        self.scale = head_dim ** -0.5
        # NOTE not tested for prime-time yet
        self.fused_attn = use_fused_attn(experimental=True)
        self.q_if = nn.Identity()
        self.k_if = nn.Identity()
        self.v_if = nn.Identity()

        self.stdp_qk = nn.Identity()
        self.softmax_if = nn.Identity()

        self.stdp_av = nn.Identity()
        self.snn_mode = False
        self.tau = 0
        self.timestep = 0
        self.v_queue = list()
        self.v_trace_queue = list()

        # define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_h - 1) * (2 * win_w - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(
            win_h, win_w), persistent=False)

        self.qkv = nn.Linear(dim, attn_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.tp = 0
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.at1 = MyAt()
        self.at2 = MyAt()
        self.softmax = nn.Softmax(dim=-1)
        self.x_in_test = MyTestPlace(place='fc_qkv')
        self.q_out_test = MyTestPlace(place='q')
        self.k_out_test = MyTestPlace(place='k')
        self.v_out_test = MyTestPlace(place='v')
        self.attn_out_test = MyTestPlace(place='s')
        self.x_out_test = MyTestPlace(place='fc_out')

        self.B = 0
        self.N = 0

    def _get_rel_pos_bias(self) -> torch.Tensor:
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        return relative_position_bias.unsqueeze(0)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        if (x is not None):
            B_, N, C = x.shape
            x = sum(self.x_in_test(x))
            self.B = B_
            self.N = N
            qkv = self.qkv(x).reshape(
                B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)

            q = sum(self.q_out_test(q))
            k = sum(self.k_out_test(k))
            v = sum(self.v_out_test(v))

        else:
            q = None
            k = None
            v = None
        q = self.q_if(q)
        k = self.k_if(k)
        v = self.v_if(v)
        if (self.snn_mode and v is not None):
            self.v_queue.append(v)
            self.v_trace_queue.append(self.v_if.trace_v)


        if self.fused_attn:
            assert False, "fused atten not supported"
            attn_mask = self._get_rel_pos_bias()
            if mask is not None:
                num_win = mask.shape[0]
                mask = mask.view(1, num_win, 1, N, N).expand(
                    B_ // num_win, -1, self.num_heads, -1, -1)
                attn_mask = attn_mask + mask.reshape(-1, self.num_heads, N, N)
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            if (q is not None):
                if (self.snn_mode == True):
                    attn = ((q @ self.k_if.trace_v.transpose(-2, -1))+(self.q_if.trace_v @
                            k.transpose(-2, -1))-((self.q_if.trace_pure * q) @ k.transpose(-2, -1))) * self.scale

                    attn = attn + self._get_rel_pos_bias() / ((1-1/(self.tau**self.timestep)) /
                                                              (1-1/self.tau)-1)  # num_head,2*window-1,2*window-1 == 4,13,13
                else:
                    attn = self.at1(q, k.transpose(-2, -1))
                    attn *= self.scale
                    attn = attn + self._get_rel_pos_bias()

            else:
                attn = None

            if (attn is not None):
                if (self.snn_mode):
                    if mask is not None:
                        num_win = mask.shape[0]
                        attn = attn.view(-1, num_win, self.num_heads, self.N, self.N) + mask.unsqueeze(
                            1).unsqueeze(0) / ((1-1/(self.tau**self.timestep))/(1-1/self.tau)-1)
                        attn = attn.view(-1, self.num_heads, self.N, self.N)

                else:
                    if mask is not None:
                        num_win = mask.shape[0]
                        attn = attn.view(-1, num_win, self.num_heads,
                                         self.N, self.N) + mask.unsqueeze(1).unsqueeze(0)
                        attn = attn.view(-1, self.num_heads, self.N, self.N)

            attn = self.stdp_qk(attn)

            if (attn is not None):
                if (self.snn_mode == False):
                    attn = self.softmax(attn)
            else:
                attn = None

            attn = self.softmax_if(attn)

            if (attn is not None):
                if (self.snn_mode):
                    attn = self.attn_drop(attn)
                    tp_v = self.v_queue.pop(0)
                    tp_v_trace = self.v_trace_queue.pop(0)
                    x = ((self.softmax_if.trace_v @ tp_v)+(attn @ tp_v_trace) -
                         (((self.softmax_if.trace_v) * attn) @ tp_v))

                    if (len(self.v_queue) == 0):
                        self.stdp_av.stdp_scale = self.softmax_if.spike_sum.unsqueeze(
                            -1)

                else:
                    attn = self.attn_drop(attn)

                    attn = sum(self.attn_out_test(attn))

                    x = self.at2(attn, v)
            else:
                x = None
            x = self.stdp_av(x)

            if (x is not None):
                x = x.transpose(1, 2).reshape(self.B, self.N, -1)
                x = sum(self.x_out_test(x))
                x = self.proj(x)
                x = self.proj_drop(x)

        return x

    def flops(self, N):
        flop = 0
        flop += N*self.dim * 3*self.dim  # q,k,v
        flop += self.num_heads * N * (self.dim//self.num_heads) * N  # QK^T
        flop += self.num_heads*N*N * (self.dim//self.num_heads)  # attn@V
        flop += N*self.dim*self.dim
        return flop

    def flops_ANN(self, N, input_ratio):
        flop = 0
        flop += ANN_Energy(N*self.dim * self.dim,self.dim,input_ratio)
        flop += ANN_Energy(N*self.dim * self.dim,self.dim,input_ratio)
        flop += ANN_Energy(N*self.dim * self.dim,self.dim,input_ratio)

        flop+=ANN_Energy(self.num_heads * N * (self.dim//self.num_heads) * N,(self.dim//self.num_heads),self.q_if.spike_count_meter.avg*self.k_if.spike_count_meter.avg)

        flop+=ANN_Energy(self.num_heads*N*N*(self.dim//self.num_heads),N ,self.softmax_if.spike_count_meter.avg *self.v_if.spike_count_meter.avg)

        return flop, N*self.dim*self.dim,self.stdp_av.spike_count_meter.avg,N*self.dim*self.dim/self.attn_dim

    def flops_snn(self, N, input_ratio):
        flop = 0
        flop += SpikeSim_Energy(N*self.dim * self.dim,self.timestep,input_ratio,self.q_if.mem_count_meter.avg,self.dim)
        flop += SpikeSim_Energy(N*self.dim * self.dim,self.timestep,input_ratio,self.k_if.mem_count_meter.avg,self.dim)
        flop += SpikeSim_Energy(N*self.dim * self.dim,self.timestep,input_ratio,self.v_if.mem_count_meter.avg,self.dim)
        flop+=SpikeSim_Energy(self.num_heads * N * (self.dim//self.num_heads) * N,self.timestep,self.q_if.spike_count_meter.avg*self.k_if.spike_count_meter.avg,self.softmax_if.mem_count_meter.avg,(self.dim//self.num_heads))
        flop+=SpikeSim_Energy(self.num_heads*N*N*(self.dim//self.num_heads),self.timestep, self.softmax_if.spike_count_meter.avg *self.v_if.spike_count_meter.avg, self.stdp_av.mem_count_meter.avg,N )
        return flop, N*self.dim*self.dim,self.stdp_av.spike_count_meter.avg,N*self.dim*self.dim/self.attn_dim

    def flops_snn2(self, N, input_ratio):
        flop = 0
        flop += N*self.dim * 3*self.dim*input_ratio  # q,k,v
        flop += self.num_heads * N * (self.dim//self.num_heads) * N * \
            self.q_if.spike_count_meter.avg*self.k_if.spike_count_meter.avg  # QK^T
        flop += self.num_heads*N*N * \
            (self.dim//self.num_heads)*self.softmax_if.spike_count_meter.avg * \
            self.v_if.spike_count_meter.avg  # attn@V
        flop += N*self.dim*self.dim * self.stdp_av.spike_count_meter.avg
        return flop


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.
    """

    def __init__(
            self,
            dim: int,
            input_resolution: _int_or_tuple_2_t,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            shift_size: int = 0,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: float = 0.,
            act_layer: Callable = nn.ReLU,
            norm_layer: Callable = nn.BatchNorm1d,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            window_size: Window size.
            num_heads: Number of attention heads.
            head_dim: Enforce the number of channels per head
            shift_size: Shift size for SW-MSA.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        ws, ss = self._calc_window_shift(window_size, shift_size)
        self.window_size: Tuple[int, int] = ws
        self.shift_size: Tuple[int, int] = ss
        self.window_area = self.window_size[0] * self.window_size[1]
        self.mlp_ratio = mlp_ratio
        self.snn_mode = False
        self.timestep = 0

        self.norm1 = nn.BatchNorm2d(dim)
        self.norm1_if = nn.Identity()

        self.C = 0
        self.x_queue = list()
        self.x2_queue = list()
        self.x_shape = 0

        self.Hp = 0
        self.Wp = 0
        self.H = 0
        self.W = 0
        self.has_shift = 0
        self.attn = WindowAttention(
            dim,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.attn_if = nn.Identity()

        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.norm2_if = nn.Identity()

        self.mlp = Batch_Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.final_if = nn.Identity()

        if any(self.shift_size):
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            H = math.ceil(H / self.window_size[0]) * self.window_size[0]
            W = math.ceil(W / self.window_size[1]) * self.window_size[1]
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None)):
                for w in (
                        slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def _calc_window_shift(self, target_window_size, target_shift_size) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        target_window_size = to_2tuple(target_window_size)
        target_shift_size = to_2tuple(target_shift_size)
        window_size = [r if r <= w else w for r, w in zip(
            self.input_resolution, target_window_size)]
        shift_size = [0 if r <= w else s for r, w, s in zip(
            self.input_resolution, window_size, target_shift_size)]
        return tuple(window_size), tuple(shift_size)

    def _attn(self, x):
        if (x is not None):
            B, H, W, C = x.shape
            self.C = C

            # cyclic shift
            has_shift = any(self.shift_size)
            if has_shift:
                shifted_x = torch.roll(
                    x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            else:
                shifted_x = x

            # pad for resolution not divisible by window size
            pad_h = (self.window_size[0] - H %
                     self.window_size[0]) % self.window_size[0]
            pad_w = (self.window_size[1] - W %
                     self.window_size[1]) % self.window_size[1]
            shifted_x = torch.nn.functional.pad(
                shifted_x, (0, 0, 0, pad_w, 0, pad_h))
            Hp, Wp = H + pad_h, W + pad_w
            self.Hp = Hp
            self.Wp = Wp
            self.H = H
            self.W = W
            # partition windows
            # nW*B, window_size, window_size, C
            x_windows = window_partition(shifted_x, self.window_size)
            # nW*B, window_size*window_size, C
            x_windows = x_windows.view(-1, self.window_area, C)
            # self.has_shift
        else:
            x_windows = None
            # W-MSA/SW-MSA

        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        if (attn_windows is not None):
            # merge windows
            attn_windows = attn_windows.view(
                -1, self.window_size[0], self.window_size[1], self.C)
            shifted_x = window_reverse(
                attn_windows, self.window_size, self.Hp, self.Wp)  # B H' W' C
            shifted_x = shifted_x[:, :self.H, :self.W, :].contiguous()
            has_shift = any(self.shift_size)

            # reverse cyclic shift
            if has_shift:
                x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
            else:
                x = shifted_x

        else:
            x = None

        return x

    def forward(self, x):
        if (self.snn_mode and x is not None):
            self.x_queue.append(x)

        if (x is not None):
            B, H, W, C = x.shape
            self.x_shape = x.shape

            x_norm = self.norm1(x.permute(0, 3, 1, 2).contiguous()).permute(
                0, 2, 3, 1).contiguous()
        else:
            x_norm = None
        x_norm = self.norm1_if(x_norm)

        x_norm = self._attn(x_norm)

        if (x_norm is not None):

            if (self.snn_mode):
                x = self.x_queue.pop(0)
                B, H, W, C = x.shape

            x = x + self.drop_path1(x_norm)
            x = x.reshape(B, -1, C)
        else:
            x = None

        x = self.attn_if(x)
        if (x is not None):
            if (self.snn_mode):
                self.x2_queue.append(x)
            # _assert(False, f"x height ({x.shape}) is not even({self.norm2}).")

            # x_norm = self.norm2(x)

            x_norm = self.norm2(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        else:
            x_norm = None
        x_norm = self.norm2_if(x_norm)
        x_norm = self.mlp(x_norm)
        if (x_norm is not None):
            if (self.snn_mode):
                x = self.x2_queue.pop(0)
            x = x + self.drop_path2(x_norm)
            x = x.reshape(self.x_shape)
        else:
            x = None
        x = self.final_if(x)

        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_area
        flops += nW * self.attn.flops(self.window_area)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

    def flops_ANN(self, input_ratio):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += ANN_Energy(self.dim * H * W, 1, input_ratio)
        # W-MSA/SW-MSA
        nW = H * W / self.window_area

        tp_flops, a, b, c = self.attn.flops_ANN(self.window_area, self.norm1_if.spike_count_meter.avg)

        flops += tp_flops * nW
        flops += ANN_Energy(a, c, b) * nW

        # mlp
        flops_tp, mlp_ratio = self.mlp.flops_ANN(
            H, W, self.norm2_if.spike_count_meter.avg)
        flops += flops_tp
        # norm2
        flops += ANN_Energy(self.dim * H * W, 1, mlp_ratio)

        return flops, self.final_if.spike_count_meter.avg

    def flops_snn(self, input_ratio):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += SpikeSim_Energy(self.dim * H * W, self.timestep, input_ratio, self.norm1_if.mem_count_meter.avg, 1)
        # W-MSA/SW-MSA
        nW = H * W / self.window_area

        tp_flops, a, b, c = self.attn.flops_snn(self.window_area, self.norm1_if.spike_count_meter.avg)

        flops += tp_flops * nW
        flops += SpikeSim_Energy(a, self.timestep, b, self.attn_if.mem_count_meter.avg, c) * nW

        # mlp
        flops_tp, mlp_ratio = self.mlp.flops_snn(
            self.norm2_if.spike_count_meter.avg, H, W)
        flops += flops_tp
        # norm2
        flops += SpikeSim_Energy(self.dim * H * W, self.timestep, mlp_ratio, self.final_if.mem_count_meter.avg, 1)

        return flops, self.final_if.spike_count_meter.avg

    def flops_snn2(self, input_ratio):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W * input_ratio
        # W-MSA/SW-MSA
        nW = H * W / self.window_area
        flops += nW * \
                 self.attn.flops_snn2(
                     self.window_area, self.norm1_if.spike_count_meter.avg)
        # mlp
        flops_tp, mlp_ratio = self.mlp.flops_snn2(
            self.norm2_if.spike_count_meter.avg, H, W)
        flops += flops_tp
        # norm2
        flops += self.dim * H * W * mlp_ratio
        return flops, self.final_if.spike_count_meter.avg

class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    """

    def __init__(
            self,
            dim: int,
            out_dim: Optional[int] = None,
            norm_layer: Callable = nn.BatchNorm2d,
    ):
        """
        Args:
            dim: Number of input channels.
            out_dim: Number of output channels (or 2 * dim if None)
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.snn_mode = False
        self.H = 0
        self.W = 0
        self.timestep = 0

        self.out_dim = out_dim or 2 * dim
        self.norm = nn.BatchNorm2d(4 * dim)
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.patch_merging_if = nn.Identity()

    def forward(self, x):
        if (x is not None):
            B, H, W, C = x.shape
            self.H = H
            self.W = W
            self.H = self.H
            _assert(H % 2 == 0, f"x height ({H}) is not even.")
            _assert(W % 2 == 0, f"x width ({W}) is not even.")
            x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)

            x = self.norm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            x = self.reduction(x)
        x = self.patch_merging_if(x)

        return x

    def flops(self):
        flops = 0
        H = self.H
        W = self.W
        flops = H*W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

    def flops_snn2(self, input_ratio):
        flops = 0
        H = self.H
        W = self.W
        flops = H*W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim*input_ratio
        return flops, self.patch_merging_if.spike_count_meter.avg

    def flops_snn(self, input_ratio):
        flops = 0
        H = self.H
        W = self.W
        flops +=SpikeSim_Energy((H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim,self.timestep, input_ratio, self.patch_merging_if.mem_count_meter.avg,(4 * self.dim) )
        return flops, self.patch_merging_if.spike_count_meter.avg

    def flops_ANN(self,input_ratio):
        flops = 0
        H = self.H
        W = self.W
        flops +=ANN_Energy((H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim,(4 * self.dim),input_ratio )
        return flops, self.patch_merging_if.spike_count_meter.avg


class SwinTransformerStage(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    """

    def __init__(
            self,
            dim: int,
            out_dim: int,
            input_resolution: Tuple[int, int],
            depth: int,
            downsample: bool = True,
            num_heads: int = 4,
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[List[float], float] = 0.,
            norm_layer: Callable = nn.BatchNorm2d,
    ):
        """
        Args:
            dim: Number of input channels.
            input_resolution: Input resolution.
            depth: Number of blocks.
            downsample: Downsample layer at the end of the layer.
            num_heads: Number of attention heads.
            head_dim: Channels per head (dim // num_heads if not set)
            window_size: Local window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            proj_drop: Projection dropout rate.
            attn_drop: Attention dropout rate.
            drop_path: Stochastic depth rate.
            norm_layer: Normalization layer.
        """
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = tuple(
            i // 2 for i in input_resolution) if downsample else input_resolution
        self.depth = depth
        self.grad_checkpointing = False
        window_size = to_2tuple(window_size)
        shift_size = tuple([w // 2 for w in window_size])
        self.snn_mode = False
        self.timestep = 0

        # patch merging layer
        if downsample:
            self.downsample = PatchMerging(
                dim=dim,
                out_dim=out_dim,
                norm_layer=norm_layer,
            )
            self.downsample_bool = True

        else:
            assert dim == out_dim
            self.downsample = nn.Identity()
            self.downsample_bool = False

        # build blocks
        self.blocks = nn.Sequential(*[
            SwinTransformerBlock(
                dim=out_dim,
                input_resolution=self.output_resolution,
                num_heads=num_heads,
                head_dim=head_dim,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(
                    drop_path, list) else drop_path,
                norm_layer=norm_layer,
            )
            for i in range(depth)])

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)

        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample_bool:
            flops += self.downsample.flops()
        return flops

    def flops_snn(self, ratio):
        flops = 0
        for blk in self.blocks:
            flops_blk, ratio = blk.flops_snn(ratio)
            flops += flops_blk

        if self.downsample_bool:
            flops_down, ratio = self.downsample.flops_snn(ratio)
            flops += flops_down
        return flops, ratio

    def flops_ANN(self, ratio):
        flops = 0
        for blk in self.blocks:
            flops_blk, ratio = blk.flops_ANN(ratio)
            flops += flops_blk

        if self.downsample_bool:
            flops_down, ratio = self.downsample.flops_ANN(ratio)
            flops += flops_down
        return flops, ratio

    def flops_snn2(self, ratio):
        flops = 0
        for blk in self.blocks:
            flops_blk, ratio = blk.flops_snn2(ratio)
            flops += flops_blk
        if self.downsample_bool:
            flops_down, ratio = self.downsample.flops_snn2(ratio)
            flops += flops_down
        return flops, ratio


class SwinTransformer(nn.Module):
    """ Swin Transformer

    A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    """

    def __init__(
            self,
            img_size: _int_or_tuple_2_t = 224,
            patch_size: int = 4,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: str = 'avg',
            embed_dim: int = 96,
            depths: Tuple[int, ...] = (2, 2),
            num_heads: Tuple[int, ...] = (3, 6),
            head_dim: Optional[int] = None,
            window_size: _int_or_tuple_2_t = 7,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.1,
            norm_layer: Union[str, Callable] = nn.BatchNorm1d,
            weight_init: str = '',
            **kwargs,
    ):
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of input image channels.
            num_classes: Number of classes for classification head.
            embed_dim: Patch embedding dimension.
            depths: Depth of each Swin Transformer layer.
            num_heads: Number of attention heads in different layers.
            head_dim: Dimension of self-attention heads.
            window_size: Window size.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: If True, add a learnable bias to query, key, value.
            drop_rate: Dropout rate.
            attn_drop_rate (float): Attention dropout rate.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg')
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.output_fmt = 'NHWC'
        self.snn_mode = False

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.feature_info = []
        self.timestep = 0

        if not isinstance(embed_dim, (tuple, list)):
            embed_dim = [int(embed_dim * 2 ** i)
                         for i in range(self.num_layers)]

        # split image into non-overlapping patches
        self.patch_embed = batch_PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            norm_layer=norm_layer,
            output_fmt='NHWC',
        )
        self.patch_grid = self.patch_embed.grid_size
        self.num_patches = self.patch_embed.num_patches
        # build layers
        head_dim = to_ntuple(self.num_layers)(head_dim)
        if not isinstance(window_size, (list, tuple)):
            window_size = to_ntuple(self.num_layers)(window_size)
        elif len(window_size) == 2:
            window_size = (window_size,) * self.num_layers
        assert len(window_size) == self.num_layers
        mlp_ratio = to_ntuple(self.num_layers)(mlp_ratio)
        dpr = [x.tolist() for x in torch.linspace(
            0, drop_path_rate, sum(depths)).split(depths)]
        layers = []
        in_dim = embed_dim[0]
        scale = 1
        for i in range(self.num_layers):
            out_dim = embed_dim[i]
            layers += [SwinTransformerStage(
                dim=in_dim,
                out_dim=out_dim,
                input_resolution=(
                    self.patch_grid[0] // scale,
                    self.patch_grid[1] // scale
                ),
                depth=depths[i],
                downsample=i > 0,
                num_heads=num_heads[i],
                head_dim=head_dim[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio[i],
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
            )]
            in_dim = out_dim
            if i > 0:
                scale *= 2
            self.feature_info += [dict(num_chs=out_dim,
                                       reduction=4 * scale, module=f'layers.{i}')]
        self.layers = nn.Sequential(*layers)
        self.sparsity = 1
        self.norm = nn.BatchNorm2d(self.num_features)
        self.last_norm_if = nn.Identity()
        self.head = modified_ClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
            input_fmt=self.output_fmt,
        )
        if weight_init != 'skip':
            self.init_weights(weight_init)

    @torch.jit.ignore
    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        named_apply(get_init_weights_vit(mode, head_bias=head_bias), self)

    @torch.jit.ignore
    def no_weight_decay(self):
        nwd = set()
        for n, _ in self.named_parameters():
            if 'relative_position_bias_table' in n:
                nwd.add(n)
        return nwd

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^patch_embed',  # stem and embed
            blocks=r'^layers\.(\d+)' if coarse else [
                (r'^layers\.(\d+).downsample', (0,)),
                (r'^layers\.(\d+)\.\w+\.(\d+)', None),
                (r'^norm', (99999,)),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for l in self.layers:
            l.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.layers(x)
        if (self.snn_mode == False):
            self.sparsity = torch.count_nonzero((x / x.max() * 256).round()) / (x.flatten()).size()[0]
        if (x is not None):
            x = self.norm(x.permute(0, 3, 1, 2).contiguous()
                          ).permute(0, 2, 3, 1).contiguous()
        x = self.last_norm_if(x)

        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.num_patches // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops * self.sparsity

    def flops_ANN(self, ratio):
        flops = 0
        flops_tp, ratio = self.patch_embed.flops_ANN(ratio)
        flops += flops_tp
        for i, layer in enumerate(self.layers):
            flops_tp, ratio = layer.flops_ANN(ratio)
            flops += flops_tp

        flops += ANN_Energy(self.num_features * self.num_patches // (2 ** self.num_layers),
                            (self.num_patches // (2 ** self.num_layers)), ratio)
        flops += ANN_Energy(self.num_features * self.num_classes, self.num_features,
                            self.head.pool_if.spike_count_meter.avg)
        return flops

    def flops_snn(self, ratio):
        flops = 0
        flops_tp, ratio = self.patch_embed.flops_snn(ratio)
        flops += flops_tp
        for i, layer in enumerate(self.layers):
            flops_tp, ratio = layer.flops_snn(ratio)
            flops += flops_tp

        flops += SpikeSim_Energy(self.num_features * self.num_patches // (2 ** self.num_layers), self.timestep, ratio,
                                 self.head.pool_if.mem_count_meter.avg, (self.num_patches // (2 ** self.num_layers)))
        flops += SpikeSim_Energy(self.num_features * self.num_classes, self.timestep,
                                 self.head.pool_if.spike_count_meter.avg, self.last_norm_if.mem_count_meter.avg,
                                 self.num_features)
        return flops

    def flops_snn2(self, ratio):
        flops = 0
        flops_tp, ratio = self.patch_embed.flops_snn2(ratio)
        flops += flops_tp
        for i, layer in enumerate(self.layers):
            flops_tp, ratio = layer.flops_snn2(ratio)
            flops += flops_tp
        flops += self.num_features * \
                 self.num_patches // (2 ** self.num_layers) * ratio
        flops += self.num_features * self.num_classes * \
                 self.head.pool_if.spike_count_meter.avg
        return flops

class HSISwinModel(nn.Module):
    """针对高光谱图像分类的Swin Transformer模型

    Args:
        num_classes (int): 类别数量
        in_chans (int): 输入通道数
        img_size (int): 输入图像大小
        pretrained (bool): 是否使用预训练权重
        drop_rate (float): dropout率
    """
    def __init__(
            self,
            num_classes=16,
            in_chans=30,
            img_size=25,
            pretrained=True,
            drop_rate=0.03,
    ):
        super().__init__()

        # Patch提取
        self.patch_extract = PatchExtract(patch_size=(2, 2))

        # 计算patch相关参数
        self.patch_size = 2
        self.patches_per_dim = img_size // self.patch_size  # 每个维度上的patch数量 = 12

        # 将提取的patches嵌入到高维特征空间
        self.patch_dim = self.patch_size * self.patch_size * in_chans  # 2*2*30=120
        self.proj = nn.Linear(self.patch_dim, 96)  # 投影到与embed_dim相同的维度
        self.norm = nn.LayerNorm(96)

        # 创建Swin Transformer主干网络，注意patch_size设置为1
        self.backbone = SwinTransformer(
            img_size=self.patches_per_dim,  # 使用patches_per_dim作为输入尺寸 (12)
            patch_size=1,                   # 设置为1，表示不再分割
            in_chans=96,                    # 使用投影后的维度
            num_classes=num_classes,
            embed_dim=96,
            depths=(1, 2),
            num_heads=(3, 6),
            window_size=4,
            mlp_ratio=4.,
            qkv_bias=True,
            attn_drop_rate=drop_rate,
            drop_rate=drop_rate,
            drop_path_rate=drop_rate,
        )

        # 初始化权重
        self.apply(self._init_weights)

    def forward(self, x):
        # 检查输入是否为None
        if x is None:
            return self.backbone(None)

        # 对于正常输入的处理
        # 数据增强（仅在训练时）
        if self.training:
            B, C, H, W = x.shape
            if H > 25:
                h_start = torch.randint(0, H - 25 + 1, (1,))
                w_start = torch.randint(0, W - 25 + 1, (1,))
                x = x[:, :, h_start:h_start + 25, w_start:w_start + 25]
            if torch.rand(1) > 0.5:
                x = torch.flip(x, [3])

        # 使用patch_extract提取patches
        patches = self.patch_extract(x)  # [B, N, patch_dim]

        # 投影和归一化
        patches = self.proj(patches)  # [B, N, 96]

        # patches = patches.permute(0, 2, 1)  # -> [B, 96, N]
        # patches = self.norm(patches)  # 应用 BatchNorm1d(96)
        # patches = patches.permute(0, 2, 1)  # -> [B, N, 96]

        patches = self.norm(patches)  # [B, N, 96]

        # 将patches重新排列成2D图像格式，适配SwinTransformer的patch_embed期望输入格式
        B, N, C = patches.shape
        H_patch = W_patch = int(N ** 0.5)  # 计算patch的高度和宽度 (12x12)

        # 从[B, N, C]重新排列为[B, C, H_patch, W_patch]格式
        # 这是SwinTransformer的patch_embed层期望的NCHW格式
        x_reshaped = patches.permute(0, 2, 1).reshape(B, C, H_patch, W_patch)

        # 直接传递给backbone处理，保留patch_embed层
        x = self.backbone(x_reshaped)

        return x

    def _init_weights(self, m):
        """初始化模型权重"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)