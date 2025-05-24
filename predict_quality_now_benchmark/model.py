import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfidenceModel(nn.Module):
    def __init__(self, mesh_feat_dim=128, fusion_hidden_dim=256):
        super(ConfidenceModel, self).__init__()

        # SR Image branch (simple CNN encoder)
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1)),  # Output shape: (128, 1, 1)
            nn.Flatten()  # Final shape: (128,)
        )

        # Mesh feature branch (MLP)
        self.mesh_branch = nn.Sequential(
            nn.Linear(mesh_feat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Fusion + prediction (multi-label classification)
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 + 128, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, 3),  # 3 outputs: median / mean / std classification
            nn.Sigmoid()
        )

    def forward(self, image, mesh_feat):
        img_feat = self.image_branch(image)             # [B, 128]
        mesh_feat = self.mesh_branch(mesh_feat)         # [B, 128]
        fused = torch.cat([img_feat, mesh_feat], dim=1) # [B, 256]
        probs = self.fusion_layer(fused)                # [B, 3] with sigmoid output
        return probs
