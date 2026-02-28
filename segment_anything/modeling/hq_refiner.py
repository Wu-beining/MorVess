# models/hq_refiner.py
import torch, torch.nn as nn, torch.nn.functional as F

class HQRefiner(nn.Module):
    """
    HQ-SAM 风格的旁路细化头：
    输入：decoder的 256x256 特征 F_dec_256、编码器早期/末层特征 F_early/F_last（已上采样到 256x256）、
         以及可选的几何先验 D/T/|∇D|。
    另有：hq_token 经 hyper-MLP 生成的动态1x1卷积权重。
    输出：logits_hq (B, 1, 256, 256)
    """
    def __init__(self, c=256, use_geom=True, film=False, geom_in=3):  # D,T,|∇D|
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(c*3 + (geom_in if use_geom and not film else 0), c, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # FiLM 调制（可选）
        self.film_gamma = nn.Linear(c, c) if film else None
        self.film_beta  = nn.Linear(c, c) if film else None
        # 超网络（3层 MLP），输出动态核（1x1 卷积权重）
        self.hyper = nn.Sequential(
            nn.Linear(c, c), nn.ReLU(True),
            nn.Linear(c, c), nn.ReLU(True),
            nn.Linear(c, c)
        )

    def forward(self, hq_token_out, F_dec_256, F_early_256, F_last_256, D=None, T=None, gradD=None):
        B, C, H, W = F_dec_256.shape
        feats = torch.cat([F_dec_256, F_early_256, F_last_256], dim=1)
        if D is not None and T is not None and gradD is not None and self.film_gamma is None:
            feats = torch.cat([feats, D, T, gradD], dim=1)   # 通道拼接几何先验
        fused = self.fuse(feats)  # (B, C, H, W)

        if self.film_gamma is not None:  # FiLM 调制几何先验
            gamma = self.film_gamma(hq_token_out).view(B, C, 1, 1)
            beta  = self.film_beta(hq_token_out).view(B, C, 1, 1)
            fused = fused * (1 + gamma) + beta

        # 动态 1x1 卷积：每个样本一组通道权重
        dyn_w = self.hyper(hq_token_out).view(B, C, 1, 1)
        logits_hq = (fused * dyn_w).sum(dim=1, keepdim=True)  # (B,1,H,W)
        return logits_hq
