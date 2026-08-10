"""LSS-style single-camera occupancy network (BEVDet/FlashOcc recipe, 2D output).

Image -> ResNet-50 -> FPN fuse (stride 16) -> per-pixel depth distribution (LSS lift)
      -> scatter into the 16x16 camera-local grid -> small 2D BEV head -> occupancy.
All cameras share intrinsics, so the frustum->cell tables are precomputed per camera
(pitch/height differ slightly) and the lift is a single index_add_.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .geometry import L, DUMP


class LSSOcc(nn.Module):
    def __init__(self, tables, tables_flip, D=64, C=80, pretrained=True):
        super().__init__()
        self.D, self.C = D, C
        w = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        r = torchvision.models.resnet50(weights=w)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = (
            r.layer1, r.layer2, r.layer3, r.layer4)

        self.lat3 = nn.Conv2d(1024, 256, 1)
        self.lat4 = nn.Conv2d(2048, 256, 1)
        self.fuse = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1),
                                  nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.depth_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, D, 1))
        self.ctx_head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, C, 1))

        def gn(ch):
            return nn.GroupNorm(16, ch)
        self.bev = nn.Sequential(
            nn.Conv2d(C, 128, 3, padding=1), gn(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), gn(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), gn(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1), gn(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1))

        self.register_buffer("tables", torch.from_numpy(tables.astype("int32")),
                             persistent=False)
        self.register_buffer("tables_flip", torch.from_numpy(tables_flip.astype("int32")),
                             persistent=False)

    def backbone(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        f3 = self.layer3(x)
        f4 = self.layer4(f3)
        p = self.lat3(f3) + F.interpolate(self.lat4(f4), size=f3.shape[-2:],
                                          mode="bilinear", align_corners=False)
        return self.fuse(p)

    def forward(self, x, cam_ids, flips):
        """x [B,3,H,W] normalized; cam_ids [B] long; flips [B] bool."""
        B = x.shape[0]
        feat = self.backbone(x)                                  # [B,256,fh,fw]
        dprob = self.depth_head(feat).softmax(dim=1)             # [B,D,fh,fw]
        ctx = self.ctx_head(feat)                                # [B,C,fh,fw]
        w = dprob.unsqueeze(1) * ctx.unsqueeze(2)                # [B,C,D,fh,fw]

        idx = torch.where(flips.view(B, 1), self.tables_flip[cam_ids],
                          self.tables[cam_ids]).long()           # [B,N]
        n_cell = DUMP + 1
        flat = idx + torch.arange(B, device=idx.device).view(B, 1) * n_cell
        acc = w.new_zeros((self.C, B * n_cell), dtype=torch.float32)
        acc.index_add_(1, flat.reshape(-1),
                       w.permute(1, 0, 2, 3, 4).reshape(self.C, -1).float())
        bev = (acc.view(self.C, B, n_cell)[:, :, :DUMP]
               .permute(1, 0, 2).reshape(B, self.C, L, L))
        occ = F.softplus(self.bev(bev)).squeeze(1)               # [B,16,16] >= 0
        return occ
