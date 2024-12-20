import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import SimplePatchifier, TwoLayerNN


class ViGBlock(nn.Module):
    def __init__(self, in_features, num_edges=9, head_num=1):
        super().__init__()
        self.k = num_edges
        self.num_edges = num_edges
        self.in_layer1 = TwoLayerNN(in_features)
        self.out_layer1 = TwoLayerNN(in_features)
        self.droppath1 = nn.Identity()  # DropPath(0)
        self.in_layer2 = TwoLayerNN(in_features, in_features*4)
        self.out_layer2 = TwoLayerNN(in_features, in_features*4)
        self.droppath2 = nn.Identity()  # DropPath(0)
        self.multi_head_fc = nn.Conv1d(
            in_features*2, in_features, 1, 1, groups=head_num)

    def forward(self, x):
        B, N, C = x.shape

        sim = x @ x.transpose(-1, -2)
        graph = sim.topk(self.k, dim=-1).indices

        shortcut = x
        x = self.in_layer1(x.reshape(B * N, -1)).view(B, N, -1)

        # aggregation
        neibor_features = x[torch.arange(B).unsqueeze(-1).expand(-1, N).unsqueeze(-1), graph]
        x = torch.stack([x, (neibor_features - x.unsqueeze(-2)).amax(dim=-2)], dim=-1)

        # update
        # Multi-head
        x = self.multi_head_fc(x.view(B * N, -1, 1)).view(B, N, -1)

        x = self.droppath1(self.out_layer1(F.gelu(x).view(B * N, -1)).view(B, N, -1))
        x = x + shortcut

        x = self.droppath2(self.out_layer2(F.gelu(self.in_layer2(
            x.view(B * N, -1)))).view(B, N, -1)) + x

        return x


class VGNN(nn.Module):
    def __init__(self, in_features=3*16*16, out_feature=320, num_patches=196,
                 num_ViGBlocks=16, num_edges=9, head_num=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out_feature//8, stride=2, padding=1, kernel_size=3), # 112
            nn.MaxPool2d(stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_feature//8),
            nn.GELU(),
            nn.Conv2d(in_channels=out_feature//8, out_channels=out_feature // 4, stride=2, padding=1, kernel_size=3), # 56
            nn.MaxPool2d(stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_feature // 4),
            nn.GELU(),
            nn.Conv2d(in_channels=out_feature//4, out_channels=out_feature//2, stride=2, padding=1, kernel_size=3), # 28
            nn.MaxPool2d(stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_feature//2),
            nn.GELU(),
            nn.Conv2d(in_channels=out_feature // 2, out_channels=out_feature, stride=2, padding=1, kernel_size=3), # 14
            nn.MaxPool2d(stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_feature))
        #
        # self.patchifier = SimplePatchifier()
        # # self.patch_embedding = TwoLayerNN(in_features)
        # self.patch_embedding = nn.Sequential(
        #     nn.Linear(in_features, out_feature//2),
        #     nn.BatchNorm1d(out_feature//2),
        #     nn.GELU(),
        #     nn.Linear(out_feature//2, out_feature//4),
        #     nn.BatchNorm1d(out_feature//4),
        #     nn.GELU(),
        #     nn.Linear(out_feature//4, out_feature//8),
        #     nn.BatchNorm1d(out_feature//8),
        #     nn.GELU(),
        #     nn.Linear(out_feature//8, out_feature//4),
        #     nn.BatchNorm1d(out_feature//4),
        #     nn.GELU(),
        #     nn.Linear(out_feature//4, out_feature//2),
        #     nn.BatchNorm1d(out_feature//2),
        #     nn.GELU(),
        #     nn.Linear(out_feature//2, out_feature),
        #     nn.BatchNorm1d(out_feature)
        # )
        self.pose_embedding = nn.Parameter(
            torch.rand(num_patches, out_feature))

        self.blocks = nn.Sequential(
            *[ViGBlock(out_feature, num_edges, head_num)
              for _ in range(num_ViGBlocks)])

    def forward(self, x):

        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        x = x.permute(0, 2, 1)
        # x = self.patchifier(x)
        # B, N, C, H, W = x.shape
        # x = self.patch_embedding(x.view(B * N, -1)).view(B, N, -1)
        x = x + self.pose_embedding

        x = self.blocks(x)

        return x


class Classifier(nn.Module):
    def __init__(self, in_features=3*16*16, out_feature=320,
                 num_patches=196, num_ViGBlocks=16, hidden_layer=1024,
                 num_edges=9, head_num=1, n_classes=10):
        super().__init__()
        self.backbone = VGNN(in_features, out_feature,
                             num_patches, num_ViGBlocks,
                             num_edges, head_num)

        self.predictor = nn.Sequential(
            nn.Linear(out_feature*num_patches, hidden_layer),
            nn.BatchNorm1d(hidden_layer),
            nn.GELU(),
            nn.Linear(hidden_layer, n_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        B, N, C = features.shape
        x = self.predictor(features.view(B, -1))
        return features, x
