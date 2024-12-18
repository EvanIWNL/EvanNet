import torch
import torch.nn as nn
import torch.nn.functional as F

from wtconv import WTConv2d


class Stem(nn.Module):  # Upsample
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """

    def __init__(self, img_size=224, in_dim=3, out_dim=768):
        super().__init__()
        self.convs = nn.Sequential(
            WTConv2d(in_dim, out_dim // 4, 3, stride=1, wt_levels=3, wt_type='db4'),  # 224
            nn.BatchNorm2d(out_dim // 4),
            nn.GELU(),
            nn.Conv2d(out_dim // 4, out_dim // 4, 3, 2, 1),  # 112
            nn.BatchNorm2d(out_dim // 4),
            nn.GELU(),
            WTConv2d(out_dim // 4, out_dim // 2, 3, stride=1, wt_levels=3, wt_type='db4'),  # 112
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.Conv2d(out_dim // 2, out_dim // 2, 3, 2, 1),  # 56
            nn.BatchNorm2d(out_dim // 2),
            nn.GELU(),
            WTConv2d(out_dim // 2, out_dim, 3, stride=1, wt_levels=3, wt_type='db4'),  # 56*56*768
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x
