import torch
import torch.nn as nn
import torch.nn.functional as F
from models.wfdvim.blocks import CoordAtt, Conv2d_BN


class LaplaceEdgeDetector(nn.Module):
    def __init__(self):
        super(LaplaceEdgeDetector, self).__init__()

        kernel = torch.tensor([[-1, -1, -1],
                               [-1, 8, -1],
                               [-1, -1, -1]], dtype=torch.float32)

        kernel = kernel.view(1, 1, 3, 3)

        self.register_buffer('kernel', kernel)

    def forward(self, x):

        return F.conv2d(x, self.kernel, padding=1)

class WaveletGuidedAttention(nn.Module):
    def __init__(self, dim):
        super(WaveletGuidedAttention, self).__init__()

        self.fusion_conv = Conv2d_BN(dim * 3, dim, ks=3, stride=1, pad=1)

        self.attention_gen = nn.Sequential(
            nn.Conv2d(dim, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.ca = CoordAtt(dim, reduction=32)

        self.laplace = LaplaceEdgeDetector()

    def forward(self, x_skip, x_high_dwt, x_pred_upper):
        residual = x_skip

        pred = torch.sigmoid(x_pred_upper)

        if pred.shape[2:] != x_skip.shape[2:]:
            pred = F.interpolate(pred, size=x_skip.shape[2:], mode='bilinear', align_corners=True)

        background_att = 1 - pred
        feat_reverse = x_skip * background_att

        edge_pred = self.laplace(pred)
        edge_pred = torch.abs(edge_pred)

        feat_boundary = x_skip * edge_pred

        if x_high_dwt.shape[2:] != x_skip.shape[2:]:
            x_high_dwt = F.interpolate(x_high_dwt, size=x_skip.shape[2:], mode='bilinear', align_corners=True)

        if x_high_dwt.shape[1] != x_skip.shape[1]:

            high_freq_mask, _ = torch.max(x_high_dwt, dim=1, keepdim=True)

            feat_high = x_skip * high_freq_mask
        else:

            feat_high = x_skip * x_high_dwt

        fusion = torch.cat([feat_reverse, feat_boundary, feat_high], dim=1)

        fusion = self.fusion_conv(fusion)

        att_map = self.attention_gen(fusion)
        fusion = fusion * att_map

        out = fusion + residual

        out = self.ca(out)

        return out
