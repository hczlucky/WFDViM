import math
import copy
from functools import partial
from typing import Optional, Callable, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_

from models.wavelet.wfdb import WaveletFrequencyDecouplingBlock

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

try:
    from models.wfdvim.selective_scan import selective_scan_state_flop_jit, selective_scan_fn, ConvLayer
except:
    from models.wfdvim.selective_scan import selective_scan_state_flop_jit, selective_scan_fn

try:
    from Dwconv.dwconv_layer import DepthwiseFunction
except:
    DepthwiseFunction = None

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight /\
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups,
                            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class RepDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
        self.apply(self._init_weights)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [1, 1, 1, 1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device),
                                           [1, 1, 1, 1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight /\
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv

class RepDW_Axias(torch.nn.Module):
    def __init__(self, ed, kernel_max=7, kernel=(1, 7)) -> None:
        super().__init__()
        self.kernel = kernel
        self.kernel_max = kernel_max
        padding = kernel_max // 2
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        if kernel == (1, kernel_max):
            self.conv = Conv2d_BN(ed, ed, (1, kernel_max), 1, (0, padding), groups=ed)
        else:
            self.conv = Conv2d_BN(ed, ed, (kernel_max, 1), 1, (padding, 0), groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
        self.apply(self._init_weights)

    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.no_grad()
    def fuse(self):
        conv = self.conv.fuse()
        conv1 = self.conv1

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        padding = self.kernel_max // 2

        if self.kernel == (1, self.kernel_max):
            conv1_w = torch.nn.functional.pad(conv1_w, [padding, padding])
            identity = torch.nn.functional.pad(torch.ones(conv_w.shape[0], conv_w.shape[1], 1, 1, device=conv_w.device),
                                               [padding, padding])
        else:
            conv1_w = torch.nn.functional.pad(conv1_w, [0, 0, padding, padding])
            identity = torch.nn.functional.pad(torch.ones(conv_w.shape[0], conv_w.shape[1], 1, 1, device=conv_w.device),
                                               [0, 0, padding, padding])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight /\
            (bn.running_var + bn.eps) ** 0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv

class Rep_Inception(torch.nn.Module):
    def __init__(self, dim, kernel_max=7, ratio=0.5) -> None:
        super().__init__()
        gc = int(dim * ratio)
        self.dwconv_h = RepDW_Axias(gc, kernel_max=kernel_max, kernel=(1, kernel_max))
        self.dwconv_w = RepDW_Axias(gc, kernel_max=kernel_max, kernel=(kernel_max, 1))
        self.split = (dim - gc, gc)

    def forward(self, x):
        x_w, x_h = torch.split(x, self.split, dim=1)
        return torch.cat(
            (self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class StateFusion(nn.Module):
    def __init__(self, dim):
        super(StateFusion, self).__init__()
        self.dim = dim
        self.kernel_3 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_1 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_2 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True)

    @staticmethod
    def padding(input_tensor, padding):
        return torch.nn.functional.pad(input_tensor, padding, mode='replicate')

    def forward(self, h):
        if self.training:
            h1 = F.conv2d(self.padding(h, (1, 1, 1, 1)), self.kernel_3, padding=0, dilation=1, groups=self.dim)
            h2 = F.conv2d(self.padding(h, (3, 3, 3, 3)), self.kernel_3_1, padding=0, dilation=3, groups=self.dim)
            h3 = F.conv2d(self.padding(h, (5, 5, 5, 5)), self.kernel_3_2, padding=0, dilation=5, groups=self.dim)
            out = self.alpha[0] * h1 + self.alpha[1] * h2 + self.alpha[2] * h3
            return out
        else:
            if not hasattr(self, "_merge_weight"):
                self._merge_weight = torch.zeros((self.dim, 1, 11, 11), device=h.device)
                self._merge_weight[:, :, 4:7, 4:7] = self.alpha[0] * self.kernel_3
                self._merge_weight[:, :, 2:3, 2:3] = self.alpha[1] * self.kernel_3_1[:, :, 0:1, 0:1]
                self._merge_weight[:, :, 2:3, 5:6] = self.alpha[1] * self.kernel_3_1[:, :, 0:1, 1:2]
                self._merge_weight[:, :, 2:3, 8:9] = self.alpha[1] * self.kernel_3_1[:, :, 0:1, 2:3]
                self._merge_weight[:, :, 5:6, 2:3] = self.alpha[1] * self.kernel_3_1[:, :, 1:2, 0:1]
                self._merge_weight[:, :, 5:6, 5:6] += self.alpha[1] * self.kernel_3_1[:, :, 1:2, 1:2]
                self._merge_weight[:, :, 5:6, 8:9] = self.alpha[1] * self.kernel_3_1[:, :, 1:2, 2:3]
                self._merge_weight[:, :, 8:9, 2:3] = self.alpha[1] * self.kernel_3_1[:, :, 2:3, 0:1]
                self._merge_weight[:, :, 8:9, 5:6] = self.alpha[1] * self.kernel_3_1[:, :, 2:3, 1:2]
                self._merge_weight[:, :, 8:9, 8:9] = self.alpha[1] * self.kernel_3_1[:, :, 2:3, 2:3]
                self._merge_weight[:, :, 0:1, 0:1] = self.alpha[2] * self.kernel_3_2[:, :, 0:1, 0:1]
                self._merge_weight[:, :, 0:1, 5:6] = self.alpha[2] * self.kernel_3_2[:, :, 0:1, 1:2]
                self._merge_weight[:, :, 0:1, 10:11] = self.alpha[2] * self.kernel_3_2[:, :, 0:1, 2:3]
                self._merge_weight[:, :, 5:6, 0:1] = self.alpha[2] * self.kernel_3_2[:, :, 1:2, 0:1]
                self._merge_weight[:, :, 5:6, 5:6] += self.alpha[2] * self.kernel_3_2[:, :, 1:2, 1:2]
                self._merge_weight[:, :, 5:6, 10:11] = self.alpha[2] * self.kernel_3_2[:, :, 1:2, 2:3]
                self._merge_weight[:, :, 10:11, 0:1] = self.alpha[2] * self.kernel_3_2[:, :, 2:3, 0:1]
                self._merge_weight[:, :, 10:11, 5:6] = self.alpha[2] * self.kernel_3_2[:, :, 2:3, 1:2]
                self._merge_weight[:, :, 10:11, 10:11] = self.alpha[2] * self.kernel_3_2[:, :, 2:3, 2:3]

            if DepthwiseFunction is not None:
                out = DepthwiseFunction.apply(h, self._merge_weight, None, 11 // 2, 11 // 2, False)
            else:

                out = F.conv2d(h, self._merge_weight, padding=5, groups=self.dim)
            return out

class StructureAwareSSM(nn.Module):
    def __init__(
            self,
            d_model,
            fusion_type: str = 'original',
            d_state=1,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                     **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn

        if fusion_type == 'original':
            self.state_fusion = StateFusion(self.d_inner)
        else:
            raise NotImplementedError(f"fusion_type {fusion_type} not implemented")

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                bias=True, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)
        if bias:
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            dt_proj.bias._no_reinit = True
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init == "random" or "constant":
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init == "random" or "constant":
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D)
            D._no_weight_decay = True
        else:
            raise NotImplementedError
        return D

    def ssm(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W

        xs = x.view(B, -1, L)

        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)

        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias

        h = self.selective_scan(
            xs, dts,
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        h = rearrange(h, "b d 1 (h w) -> b (d 1) h w", h=H, w=W)
        h = self.state_fusion(h)
        h = rearrange(h, "b d h w -> b d (h w)")

        y = h * Cs
        y = y + xs * Ds.view(-1, 1)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = rearrange(x, 'b h w d -> b d h w').contiguous()
        x = self.act(self.conv2d(x))
        y = self.ssm(x)
        y = rearrange(y, 'b d (h w)-> b h w d', h=H, w=W)
        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y

class FFN(nn.Module):
    def __init__(self, in_dim, mid_dim=None,
                 out_dim=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        mid_dim = mid_dim or in_dim
        self.fc1 = Conv2d_BN(in_dim, mid_dim, 1)
        self.fc2 = Conv2d_BN(mid_dim, out_dim, 1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WFDViMBlock(nn.Module):
    def __init__(
            self,
            dim: int = 0,
            alpha: float = 0.5,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),

            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,

            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            use_checkpoint: bool = False,
            use_pmd=True,
            process_hh=True,
            index=0,
            **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.alpha = alpha
        self.use_checkpoint = use_checkpoint

        self.dim_low = int(dim * alpha)
        self.dim_high = dim - self.dim_low

        if self.dim_low > 0:
            self.low_pre_conv = Rep_Inception(self.dim_low, kernel_max=7)

            self.wavelet_dec = WaveletFrequencyDecouplingBlock(
                self.dim_low,
                use_pmd=use_pmd,
                process_hh=process_hh
            )
            self.spatial_mamba = StructureAwareSSM(
                d_model=self.dim_low,
                d_state=ssm_d_state,
                expand=ssm_ratio,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dt_rank=ssm_dt_rank,
                dropout=ssm_drop_rate,
                fusion_type='original'
            )
            self.mamba_norm = norm_layer(self.dim_low)

        if self.dim_high > 0:
            self.local_conv = RepDW(self.dim_high)

            self.ca = CoordAtt(self.dim_high, reduction=32)

        self.norm = norm_layer(dim)
        self.ffn = FFN(
            in_dim=dim,
            mid_dim=int(dim * mlp_ratio),
            act_layer=mlp_act_layer,
            drop=mlp_drop_rate
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def _forward(self, x):

        prev_high_res = None
        if isinstance(x, (tuple, list)):
            x, prev_high_res = x
        input_x = x

        x_low = x[:, :self.dim_low, :, :]
        x_high = x[:, self.dim_low:, :, :]

        output_low = None
        output_high = None
        High_Res_to_Inject = None

        if self.dim_low > 0:
            x_low_processed = self.low_pre_conv(x_low)
            LL, High_Res, org_size = self.wavelet_dec(x_low_processed)
            High_Res_to_Inject = High_Res
            LL_in = LL.permute(0, 2, 3, 1).contiguous()
            LL_out = self.spatial_mamba(self.mamba_norm(LL_in))
            LL_out = LL_out.permute(0, 3, 1, 2).contiguous()

            output_low = self.wavelet_dec.restore_low(LL_out, org_size)

        if self.dim_high > 0:

            if High_Res_to_Inject is not None:
                if High_Res_to_Inject.shape[1] == x_high.shape[1]:

                    x_high = x_high + High_Res_to_Inject
                else:

                    min_c = min(x_high.shape[1], High_Res_to_Inject.shape[1])
                    part_added = x_high[:, :min_c, :, :] + High_Res_to_Inject[:, :min_c, :, :]

                    if min_c < x_high.shape[1]:
                        part_remain = x_high[:, min_c:, :, :]
                        x_high = torch.cat([part_added, part_remain], dim=1)
                    else:
                        x_high = part_added

            output_high = self.local_conv(x_high)

            output_high = self.ca(output_high)

        if output_low is not None and output_high is not None:
            x_mixed = torch.cat([output_low, output_high], dim=1)
        elif output_low is not None:
            x_mixed = output_low
        else:
            x_mixed = output_high

        x = input_x + self.drop_path(x_mixed)
        x = x + self.drop_path(self.ffn(x))

        return x, High_Res_to_Inject

    def forward(self, x):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, x)
        else:
            return self._forward(x)
