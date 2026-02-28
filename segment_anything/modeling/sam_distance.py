# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F
from icecream import ic

from typing import Any, Dict, List, Tuple

from .image_encoder import ImageEncoderViT
from .mask_decoder_distance import MaskDecoder_distance as MaskDecoder
from .mask_decoder_distance import MaskDecoder_multi as MaskDecoder_Mutli
from .prompt_encoder import PromptEncoder


class Sam_distance(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        predict_distance: bool = False, # --- [核心修改 1] 新增参数 ---
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.predict_distance = predict_distance # --- [核心修改 2] 保存参数 ---
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, batched_input, multimask_output, image_size):
        
        outputs = self.forward_train(batched_input, multimask_output, image_size)
        return outputs

    def forward_train(self, batched_input, multimask_output, image_size):
        b_size, hw_size, d_size = batched_input.shape[0], batched_input.shape[-2], batched_input.shape[1] # [b, d, 3, h, w]
        batched_input = batched_input.contiguous().view(-1, 3, hw_size, hw_size)

        input_images = self.preprocess(batched_input)
        image_embeddings = self.image_encoder(input_images, d_size)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        # low_res_masks, iou_predictions, low_res_distance = self.mask_decoder(
        #     image_embeddings=image_embeddings,
        #     image_pe=self.prompt_encoder.get_dense_pe(),
        #     sparse_prompt_embeddings=sparse_embeddings,
        #     dense_prompt_embeddings=dense_embeddings,
        #     multimask_output=multimask_output,
        # )
        decoder_outputs = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        low_res_masks, iou_predictions, low_res_distance = decoder_outputs
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=(image_size, image_size),
            original_size=(image_size, image_size),
        )
        # distance_maps = self.postprocess_masks(
        #     low_res_distance,
        #     input_size=(image_size, image_size),
        #     original_size=(image_size, image_size),
        # )
        outputs = {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
            # "distance_maps": distance_maps, # <-- 新增的输出项
        }
        if self.predict_distance and low_res_distance is not None:
            distance_maps = self.postprocess_masks(
                low_res_distance,
                input_size=(image_size, image_size),
                original_size=(image_size, image_size),
            )
            outputs["distance_maps"] = distance_maps

        # print(low_res_masks.shape)
        return outputs


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x











class Sam_multi(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: ImageEncoderViT,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder_Mutli,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
        predict_distance: bool = False,
        predict_thickness: bool = False,
    ) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.predict_distance = predict_distance
        self.predict_thickness = predict_thickness
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

    @property
    def device(self) -> Any:
        return self.pixel_mean.device

    def forward(self, batched_input, multimask_output, image_size):
        return self.forward_train(batched_input, multimask_output, image_size)

    def forward_train(self, batched_input, multimask_output, image_size):
        b_size, hw_size, d_size = batched_input.shape[0], batched_input.shape[-2], batched_input.shape[1]
        batched_input = batched_input.contiguous().view(-1, 3, hw_size, hw_size)

        input_images = self.preprocess(batched_input)
        image_embeddings = self.image_encoder(input_images, d_size)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(points=None, boxes=None, masks=None)

        low_res_masks, iou_predictions, low_res_dist, low_res_thick = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        masks = self.postprocess_masks(
            low_res_masks, input_size=(image_size, image_size), original_size=(image_size, image_size)
        )

        outputs = {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
        }

        # 回传低分辨率 logits + 上采样后的可视化结果
        if self.predict_distance and low_res_dist is not None:
            outputs["low_res_distance_logits"] = low_res_dist
            outputs["distance_maps"] = self.postprocess_masks(
                low_res_dist, input_size=(image_size, image_size), original_size=(image_size, image_size)
            )

        if self.predict_thickness and low_res_thick is not None:
            outputs["low_res_thickness_logits"] = low_res_thick
            outputs["thickness_maps"] = self.postprocess_masks(
                low_res_thick, input_size=(image_size, image_size), original_size=(image_size, image_size)
            )

        return outputs

    def postprocess_masks(self, masks: torch.Tensor, input_size: Tuple[int, ...], original_size: Tuple[int, ...]) -> torch.Tensor:
        masks = F.interpolate(
            masks, (self.image_encoder.img_size, self.image_encoder.img_size), mode="bilinear", align_corners=False
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
