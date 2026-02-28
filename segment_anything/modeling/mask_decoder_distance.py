# MA-SAM/segment_anything/modeling/mask_decoder.py

import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder_distance(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        predict_distance: bool = False, # --- [核心修改 1] 新增参数 ---
    ) -> None:
        """
        (文档字符串保持不变)
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        
        
        


        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 32, kernel_size=2, stride=2),
            activation(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 32, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        # --- [新增代码] ---
        # 为距离图添加一个预测头
        # 它的结构和分割掩码的超网络MLP类似
        self.predict_distance = predict_distance
        if self.predict_distance:
            self.distance_prediction_head = MLP(transformer_dim, transformer_dim, transformer_dim // 32, 3)
        # self.distance_prediction_head = MLP(transformer_dim, transformer_dim, transformer_dim // 32, 3)
        # --- [新增代码结束] ---

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # 修改返回类型
        """
        返回:
          torch.Tensor: 预测的分割掩码 (B, Q, H, W)
          torch.Tensor: 预测的iou (B, Q)
          torch.Tensor: 预测的距离图 (B, 1, H, W)  <-- 新增
        """
        # --- 修改接收的变量 ---
        masks, iou_pred, distance_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # (多掩码输出逻辑保持不变)
        if multimask_output:
            mask_slice = slice(1, self.num_multimask_outputs + 1)
            iou_pred = iou_pred[:, mask_slice]
            masks = masks[:, mask_slice, :, :]
        else:
            mask_slice = slice(0, 1)
            iou_pred = iou_pred[:, mask_slice]
            masks = masks[:, mask_slice, :, :]
            
        # --- 修改返回值 ---
        return masks, iou_pred, distance_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # 修改返回类型
        """Predicts masks. See 'forward' for more details."""
        # (Transformer前向传播保持不变)
        # dense_prompt = dense_prompt_embeddings.mean(dim=[2, 3]).unsqueeze(1) 
        # point_embedding = sparse_prompt_embeddings + dense_prompt
        B, C, H, W = image_embeddings.shape
        # B, _, C = image_embeddings.shape
        tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        point_embedding = tokens.unsqueeze(0).expand(B, -1, -1)
        # point_embedding = sparse_prompt_embeddings + dense_prompt_embeddings.sum(dim=1)
        point_tokens, _ = self.transformer(
            image_embeddings,
            image_pe,
            point_embedding,
        )
        iou_token_out = point_tokens[:, 0, :]
        mask_tokens_out = point_tokens[:, 1 : (1 + self.num_mask_tokens), :]
    
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = image_embeddings.shape
        upscaled_embedding = self.output_upscaling(image_embeddings.view(b, c, h, w))
        
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        
        iou_pred = self.iou_prediction_head(iou_token_out)

    
        avg_mask_token_out = torch.mean(mask_tokens_out, dim=1)
        
        distance_hyper_in = self.distance_prediction_head(avg_mask_token_out)
        # 将 [B, C] 转换为 [B, 1, C] 以进行矩阵乘法
        # distance_pred = (distance_hyper_in.unsqueeze(1) @ upscaled_embedding.view(b, c, h * w)).view(b, 1, h, w)
        distance_pred = None
        if self.predict_distance:
            avg_mask_token_out = torch.mean(mask_tokens_out, dim=1)
            distance_hyper_in = self.distance_prediction_head(avg_mask_token_out)
            distance_pred = (distance_hyper_in.unsqueeze(1) @ upscaled_embedding.view(b, c, h * w)).view(b, 1, h, w)



        return masks, iou_pred, distance_pred


class MaskDecoder_multi(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        predict_distance: bool = False,
        predict_thickness: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1

        # mask/iou tokens
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # upscaling from encoder feature (B,C,H/4,W/4) -> (B,C/32,H,W)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 32, kernel_size=2, stride=2),
            activation(),
        )

        # SAM 的超网络 MLP：每个 mask token 生成一组通道权重，再与上采样特征点积得到 mask
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 32, 3) for _ in range(self.num_mask_tokens)]
        )

        # IoU 预测头（和原 SAM 一样）
        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

        # ===== 新增：多任务头 =====
        self.predict_distance = predict_distance
        self.predict_thickness = predict_thickness

        # 距离/边界势场：输出 [0,1]，训练时建议用 Sigmoid+L1
        if self.predict_distance:
            self.distance_prediction_head = MLP(transformer_dim, transformer_dim, transformer_dim // 32, 3)

        # 厚度：我们预测一个标量图，再在 loss 里做 Sigmoid 并与归一化 GT 对齐（见训练脚本）
        if self.predict_thickness:
            self.thickness_prediction_head = MLP(transformer_dim, transformer_dim, transformer_dim // 32, 3)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回：
          masks                 : (B, Q, H, W)
          iou_pred              : (B, Q)
          low_res_distance_logit: (B, 1, H, W) 或 None
          low_res_thickness_logit:(B, 1, H, W) 或 None
        """
        masks, iou_pred, dist_logit, thick_logit = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        if multimask_output:
            mask_slice = slice(1, self.num_multimask_outputs + 1)
            iou_pred = iou_pred[:, mask_slice]
            masks = masks[:, mask_slice, :, :]
        else:
            # mask_slice = slice(0, 1)
            # iou_pred = iou_pred[:, mask_slice]
            # masks = masks[:, mask_slice, :, :]
            masks = masks[:, :2, :, :]
            iou_pred = iou_pred[:, :2]

        return masks, iou_pred, dist_logit, thick_logit

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, C, H, W = image_embeddings.shape

        # 只用 iou/mask token（和原版 SAM 一致）
        tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        point_embedding = tokens.unsqueeze(0).expand(B, -1, -1)

        point_tokens, _ = self.transformer(image_embeddings, image_pe, point_embedding)
        iou_token_out = point_tokens[:, 0, :]
        mask_tokens_out = point_tokens[:, 1 : (1 + self.num_mask_tokens), :]

        # 超网络生成每个 mask 的通道权重
        hyper_in_list = [self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
                         for i in range(self.num_mask_tokens)]
        hyper_in = torch.stack(hyper_in_list, dim=1)

        # 上采样 encoder 特征
        b, c, h, w = image_embeddings.shape
        upscaled_embedding = self.output_upscaling(image_embeddings.view(b, c, h, w))  # (B, C', H, W)
        b, c, h, w = upscaled_embedding.shape

        # (B,Q,C') x (B,C',HW) -> (B,Q,HW) -> (B,Q,H,W)
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # IoU
        iou_pred = self.iou_prediction_head(iou_token_out)

        # ====== 额外回归头：distance & thickness ======
        dist_logit = None
        thick_logit = None

        # 用「平均」mask token 表示图来驱动额外回归（也可选用 iou/mask 某通道）
        avg_mask_token_out = torch.mean(mask_tokens_out, dim=1)

        if self.predict_distance:
            dist_vec = self.distance_prediction_head(avg_mask_token_out)          # (B, C')
            dist_logit = (dist_vec.unsqueeze(1) @ upscaled_embedding.view(b, c, h * w)).view(b, 1, h, w)

        if self.predict_thickness:
            thick_vec = self.thickness_prediction_head(avg_mask_token_out)        # (B, C')
            thick_logit = (thick_vec.unsqueeze(1) @ upscaled_embedding.view(b, c, h * w)).view(b, 1, h, w)

        return masks, iou_pred, dist_logit, thick_logit




class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
