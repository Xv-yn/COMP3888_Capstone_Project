# model_vitpose.py
import torch
import torch.nn as nn
import timm

class ViTPoseLike(nn.Module):
    def __init__(self, num_keypoints=12, img_size=(256, 192), vit_name="vit_base_patch16_224"):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.img_h, self.img_w = img_size

        # IMPORTANT: create ViT with the SAME size you feed (256x192)
        self.backbone = timm.create_model(
            vit_name,
            pretrained=True,
            num_classes=0,
            img_size=(self.img_h, self.img_w),   # ← key change
        )

        # Read patch size & token grid from the backbone
        # (timm exposes these on the patch_embed module)
        self.patch = self.backbone.patch_embed.patch_size[0]  # 16 for vit_*_patch16_*
        gh, gw = self.backbone.patch_embed.grid_size          # e.g., (16, 12) for 256x192
        self.grid_h, self.grid_w = int(gh), int(gw)

        self.neck = nn.Sequential(
            nn.Conv2d(self.backbone.num_features, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # ×2
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # ×2 => ×4 total
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, num_keypoints, kernel_size=1),
        )

    def forward(self, x):
        B = x.shape[0]
        feats = self.backbone.forward_features(x)  # [B, T(+CLS), C]
        # Drop CLS token if present
        if hasattr(self.backbone, "cls_token") and self.backbone.cls_token is not None:
            # expected tokens = grid_h * grid_w + 1 (CLS)
            if feats.dim() == 3 and feats.shape[1] == (self.grid_h * self.grid_w + 1):
                feats = feats[:, 1:, :]
        # [B, tokens, C] -> [B, C, H, W]
        feats = feats.transpose(1, 2).contiguous().view(B, -1, self.grid_h, self.grid_w)
        feats = self.neck(feats)
        return self.head(feats)
