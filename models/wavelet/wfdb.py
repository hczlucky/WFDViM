import torch
import torch.nn as nn
import torch.nn.functional as F

from models.wavelet.DWT_layer import DWT_2D, IDWT_2D


class LearnableAnisotropicDiffusion(nn.Module):
    def __init__(self, channels, iterations=2):
        super().__init__()
        self.iterations = iterations
        self.channels = channels
        self.register_buffer(
            "kernel_n",
            torch.tensor([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "kernel_s",
            torch.tensor([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "kernel_w",
            torch.tensor([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "kernel_e",
            torch.tensor([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=torch.float32).view(1, 1, 3, 3),
        )
        self.coef_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, groups=channels, bias=False),
            nn.Sigmoid(),
        )
        self.dt = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        dt = torch.clamp(self.dt, min=0.001, max=0.20)
        kernel_n = self.kernel_n.repeat(self.channels, 1, 1, 1)
        kernel_s = self.kernel_s.repeat(self.channels, 1, 1, 1)
        kernel_w = self.kernel_w.repeat(self.channels, 1, 1, 1)
        kernel_e = self.kernel_e.repeat(self.channels, 1, 1, 1)
        for _ in range(self.iterations):
            x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
            grad_n = F.conv2d(x_pad, kernel_n, groups=self.channels)
            grad_s = F.conv2d(x_pad, kernel_s, groups=self.channels)
            grad_w = F.conv2d(x_pad, kernel_w, groups=self.channels)
            grad_e = F.conv2d(x_pad, kernel_e, groups=self.channels)
            divergence = (
                self.coef_net(torch.abs(grad_n)) * grad_n
                + self.coef_net(torch.abs(grad_s)) * grad_s
                + self.coef_net(torch.abs(grad_w)) * grad_w
                + self.coef_net(torch.abs(grad_e)) * grad_e
            )
            x = x + dt * divergence
        return x


class WaveletFrequencyDecouplingBlock(nn.Module):
    def __init__(self, dim, wavename="bior2.2", use_pmd=True, process_hh=True):
        super().__init__()
        self.dim = dim
        self.use_pmd = use_pmd
        self.process_hh = process_hh
        self.dwt = DWT_2D(wavename=wavename)
        self.idwt = IDWT_2D(wavename=wavename)
        if self.use_pmd:
            pmd_channels = dim * 3 if self.process_hh else dim * 2
            self.pmd = LearnableAnisotropicDiffusion(channels=pmd_channels, iterations=2)
        self.high_refine = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 3, kernel_size=1, groups=dim * 3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, semantic_hint=None):
        ll, lh, hl, hh = self.dwt(x)
        if self.use_pmd:
            if self.process_hh:
                high = self.pmd(torch.cat([lh, hl, hh], dim=1))
                lh, hl, hh = torch.chunk(high, 3, dim=1)
            else:
                high = self.pmd(torch.cat([lh, hl], dim=1))
                lh, hl = torch.chunk(high, 2, dim=1)

        high = torch.cat([lh, hl, hh], dim=1)
        high = high * self.high_refine(high)
        lh, hl, hh = torch.chunk(high, 3, dim=1)
        zero_ll = torch.zeros_like(ll)
        high_recovered = self.idwt(zero_ll, lh, hl, hh)
        if high_recovered.shape[2:] != x.shape[2:]:
            high_recovered = high_recovered[:, :, : x.shape[2], : x.shape[3]]
        return ll, high_recovered, x.shape[2:]

    def restore_low(self, ll_processed, original_size):
        zeros = torch.zeros_like(ll_processed)
        low_recovered = self.idwt(ll_processed, zeros, zeros, zeros)
        height, width = original_size
        if low_recovered.shape[2:] != (height, width):
            low_recovered = low_recovered[:, :, :height, :width]
        return low_recovered
