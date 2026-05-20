import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from models.wfdvim.wga import WaveletGuidedAttention
from models.wfdvim.blocks import WFDViMBlock, Conv2d_BN, RepDW, FFN


class MAD(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()

        self.conv2 = Conv2d_BN(in_channels_list[0], out_channels, 1)
        self.conv3 = Conv2d_BN(in_channels_list[1], out_channels, 1)
        self.conv4 = Conv2d_BN(in_channels_list[2], out_channels, 1)

        self.agg_conv = Conv2d_BN(out_channels, out_channels, 3, 1, 1)

        self.predict = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x2, x3, x4):

        feat2 = self.conv2(x2)
        feat3 = self.conv3(x3)
        feat4 = self.conv4(x4)

        feat2_down = F.interpolate(feat2, size=feat3.shape[2:], mode='bilinear', align_corners=True)
        feat4_up = F.interpolate(feat4, size=feat3.shape[2:], mode='bilinear', align_corners=True)

        feat_agg = feat2_down + feat3 + feat4_up
        feat_agg = self.agg_conv(feat_agg)

        pred_coarse = self.predict(feat_agg)

        return pred_coarse

class LocalBlock(nn.Module):
    def __init__(self, dim, hidden_dim=64, drop_path=0., use_layer_scale=True):
        super().__init__()
        self.dwconv = RepDW(dim)
        self.mlp = FFN(dim, hidden_dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = nn.Parameter(torch.ones(dim).unsqueeze(-1).unsqueeze(-1), requires_grad=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        high_res = None
        if isinstance(x, (tuple, list)):
            x, high_res = x

        input = x

        x = self.dwconv(x)
        x = self.mlp(x)

        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)

        return x, high_res

def stem(in_chs, out_chs):
    return nn.Sequential(
        Conv2d_BN(in_chs, out_chs // 2, 3, 2, 1),
        nn.GELU(),
        Conv2d_BN(out_chs // 2, out_chs, 3, 2, 1),
        nn.GELU(),
    )

class Embedding(nn.Module):

    def __init__(self, patch_size=16, stride=2, padding=0, in_chans=3, embed_dim=48):
        super().__init__()
        self.proj = Conv2d_BN(in_chans, embed_dim, ks=patch_size, stride=stride, pad=padding)

    def forward(self, x):
        x = self.proj(x)
        return x

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm, out_dim=None):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale

        if out_dim is None:
            self.out_dim = dim // dim_scale
        else:
            self.out_dim = out_dim

        linear_out_features = self.out_dim * dim_scale * dim_scale

        self.expand = nn.Linear(dim, linear_out_features, bias=False)
        self.norm = norm_layer(self.out_dim)

    def forward(self, x):

        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = self.expand(x)

        from einops import rearrange

        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c',
                      p1=self.dim_scale, p2=self.dim_scale, c=self.out_dim)

        x = self.norm(x)
        return x.permute(0, 3, 1, 2)

class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.expand = nn.Linear(dim, dim * dim_scale * dim_scale, bias=False)
        self.norm = norm_layer(dim)
        self.dim_scale = dim_scale

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.expand(x)
        from einops import rearrange
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=x.shape[-1] // (self.dim_scale ** 2))
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)

def Stage(dim, index, depth, mlp_ratio=4.,
          ssm_d_state=16, ssm_ratio=2.0, ssm_num=1, alpha=0.5,
          drop_path=0., **kwargs):
    blocks = []
    dpr = [x.item() for x in torch.linspace(0, drop_path, depth)] if isinstance(drop_path, float) else drop_path

    for block_idx in range(depth):

        if depth - block_idx <= ssm_num:
            blocks.append(
                WFDViMBlock(
                    dim=dim, alpha=alpha, drop_path=dpr[block_idx],
                    ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, mlp_ratio=mlp_ratio,
                    index=index, **kwargs
                )
            )

        elif index == 2 and block_idx == depth // 2 and ssm_num > 0:
             blocks.append(
                WFDViMBlock(
                    dim=dim, alpha=alpha, drop_path=dpr[block_idx],
                    ssm_d_state=ssm_d_state, ssm_ratio=ssm_ratio, mlp_ratio=mlp_ratio,
                    index=index, **kwargs
                )
            )

        else:
            blocks.append(
                LocalBlock(
                    dim=dim,
                    hidden_dim=int(mlp_ratio * dim),
                    drop_path=dpr[block_idx]
                )
            )

    return nn.Sequential(*blocks)

class WFDViMNet(nn.Module):
    def __init__(self,
                 in_chans=3,
                 num_classes=1,

                 layers=[2, 2, 6, 2],
                 embed_dims=[64, 128, 384, 512],
                 dims_decoder=[512, 384, 128, 64],
                 depths_decoder=[2, 2, 2, 1],

                 ssm_num=1,
                 d_state=1,
                 ssm_ratio=2.0,
                 mlp_ratio=4.0,
                 drop_path_rate=0.2,

                 alphas_enc=[0.25, 0.5, 0.5, 0.75],
                 alphas_dec=[0.75, 0.5, 0.5, 0.25],
                 use_pmd=True,
                 process_hh=True,
                 **kwargs):
        super().__init__()

        if 'dims' in kwargs:
            embed_dims = kwargs['dims']

        if 'depths' in kwargs:
            layers = kwargs['depths']

        self.dims_enc = embed_dims
        self.depths_enc = layers

        if dims_decoder is None:
            self.dims_dec = embed_dims[::-1]
        else:
            self.dims_dec = dims_decoder

        if depths_decoder is None:
            self.depths_dec = layers[::-1]
        else:
            self.depths_dec = depths_decoder

        self.num_enc_layers = len(layers)
        self.num_dec_layers = len(self.depths_dec)

        self.stem = stem(in_chans, self.dims_enc[0])
        self.encoder_stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(layers))]
        enc_cur = 0
        for i in range(self.num_enc_layers):

            if i > 0:
                downsample = Embedding(
                    patch_size=3, stride=2, padding=1,
                    in_chans=self.dims_enc[i - 1], embed_dim=self.dims_enc[i]
                )
                self.downsamples.append(downsample)
            else:
                self.downsamples.append(nn.Identity())

            stage = Stage(
                dim=self.dims_enc[i],
                index=i,
                depth=self.depths_enc[i],

                ssm_d_state=d_state,
                ssm_ratio=ssm_ratio,
                mlp_ratio=mlp_ratio,
                ssm_num=ssm_num,
                alpha=alphas_enc[i] if i < len(alphas_enc) else 0.5,
                drop_path=enc_dpr[enc_cur:enc_cur + self.depths_enc[i]],
                use_pmd=use_pmd,
                process_hh=process_hh,
                **kwargs
            )
            self.encoder_stages.append(stage)
            enc_cur += self.depths_enc[i]

        self.upsamples = nn.ModuleList()

        self.wga_modules = nn.ModuleList()
        self.out_heads = nn.ModuleList()

        pd_in_channels = [self.dims_enc[1], self.dims_enc[2], self.dims_enc[3]]

        pd_in_channels = [self.dims_enc[1], self.dims_enc[2], self.dims_enc[3]]
        pd_out_ch = self.dims_dec[1]
        self.pd_module = MAD(pd_in_channels, out_channels=pd_out_ch)

        for i in range(self.num_dec_layers):

            if i > 0:

                self.upsamples.append(
                    PatchExpand2D(
                        dim=self.dims_dec[i - 1],
                        dim_scale=2,
                        out_dim=self.dims_dec[i]
                    )
                )

                self.wga_modules.append(
                    WaveletGuidedAttention(dim=self.dims_dec[i])
                )

                head_dim = self.dims_dec[i + 1] if i < self.num_dec_layers - 1 else self.dims_dec[i]
                self.out_heads.append(
                    nn.Conv2d(head_dim, num_classes, kernel_size=1)
                )

        self.final_up = Final_PatchExpand2D(dim=self.dims_dec[-1], dim_scale=4)
        self.head = nn.Conv2d(self.dims_dec[-1], num_classes, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        skips = []
        high_freqs = []
        x = self.stem(x)

        for i in range(self.num_enc_layers):
            x = self.downsamples[i](x)
            x, x_high = self.encoder_stages[i](x)
            if i < self.num_enc_layers - 1:
                skips.append(x)
                if x_high is None:
                    high_freqs.append(torch.zeros_like(x))
                else:
                    high_freqs.append(x_high)

        preds = []

        pred_upper = self.pd_module(skips[1], skips[2], x)
        preds.append(pred_upper)

        x = self.upsamples[0](x)

        for i in range(1, self.num_dec_layers):

            skip = skips[-i]
            high_freq = high_freqs[-i]

            if pred_upper.shape[2:] != skip.shape[2:]:
                raise RuntimeError(
                    f"Decoder guidance shape mismatch at layer {i}: "
                    f"pred_upper={tuple(pred_upper.shape[2:])}, skip={tuple(skip.shape[2:])}. "
                    "This variant expects learned feature upsampling to align guidance masks."
                )
            pred_guidance = pred_upper

            skip_refined = self.wga_modules[i - 1](skip, high_freq, pred_guidance)
            x = x + skip_refined

            if i < self.num_dec_layers - 1:
                x = self.upsamples[i](x)
            pred_curr = self.out_heads[i - 1](x)

            preds.append(pred_curr)
            pred_upper = pred_curr

        x = self.final_up(x)
        final_pred = self.head(x)

        if self.training:

            return [final_pred] + preds
        else:
            return final_pred
