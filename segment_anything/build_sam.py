# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn import functional as F
from icecream import ic

from functools import partial

from .modeling import ImageEncoderViT, MaskDecoder_distance, PromptEncoder, Sam_distance, TwoWayTransformer,MaskDecoder_multi,Sam_multi
from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from .modeling import ImageEncoderViT_hq,MaskDecoder_multi_hq,Sam_multi_hq


def build_sam_vit_h(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )


def build_sam_vit_b(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # adopt global attention at [3, 6, 9, 12] transform layer, else window attention layer
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std
    )
    
    
def build_sam_vit_b_distance(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        # adopt global attention at [3, 6, 9, 12] transform layer, else window attention layer
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        predict_distance=True  # 明确指定需要预测距离图
    )
    
    
def build_sam_vit_h_distance(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        predict_distance=True  # 明确指定需要预测距离图
    )
    
    
def build_sam_vit_l_distance(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375],
                    checkpoint=None):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
        num_classes=num_classes,
        image_size=image_size,
        pixel_mean=pixel_mean,
        pixel_std=pixel_std,
        predict_distance=True  # 明确指定需要预测距离图
    )

def build_sam_vit_b_distance_thickness(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375], checkpoint=None):
    return _build_sam(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12, encoder_global_attn_indexes=[2, 5, 8, 11],
        num_classes=num_classes, image_size=image_size, pixel_mean=pixel_mean, pixel_std=pixel_std, checkpoint=checkpoint,
        predict_distance=True, predict_thickness=True,
    )
    
    
def build_sam_vit_b_distance_thickness_hq(image_size, num_classes, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375], checkpoint=None):
    return _build_sam_hq(
    encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12, encoder_global_attn_indexes=[2, 5, 8, 11],
    num_classes=num_classes, image_size=image_size, pixel_mean=pixel_mean, pixel_std=pixel_std, checkpoint=checkpoint,
    predict_distance=True, predict_thickness=True,
)

def _make_backbone(encoder_embed_dim, encoder_depth, encoder_num_heads, encoder_global_attn_indexes, image_size):
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    encoder = ImageEncoderViT(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )
    prompt = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )
    decoder_common_kwargs = dict(
        num_multimask_outputs=1,  # 我们做二类，所以这里给 1（即单前景通道）
        transformer=TwoWayTransformer(depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    return encoder, prompt, decoder_common_kwargs, image_embedding_size, vit_patch_size

def _make_backbone_hq(encoder_embed_dim, encoder_depth, encoder_num_heads, encoder_global_attn_indexes, image_size):
    prompt_embed_dim = 256
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    encoder = ImageEncoderViT_hq(
        depth=encoder_depth,
        embed_dim=encoder_embed_dim,
        img_size=image_size,
        mlp_ratio=4,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        num_heads=encoder_num_heads,
        patch_size=vit_patch_size,
        qkv_bias=True,
        use_rel_pos=True,
        global_attn_indexes=encoder_global_attn_indexes,
        window_size=14,
        out_chans=prompt_embed_dim,
    )
    prompt = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )
    decoder_common_kwargs = dict(
        num_multimask_outputs=1,  # 我们做二类，所以这里给 1（即单前景通道）
        transformer=TwoWayTransformer(depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    return encoder, prompt, decoder_common_kwargs, image_embedding_size, vit_patch_size





def _build_sam(
    *,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    num_classes,
    image_size,
    pixel_mean,
    pixel_std,
    checkpoint=None,
    predict_distance=False,
    predict_thickness=False,
):
    # backbone & prompt
    encoder, prompt, decoder_common_kwargs, image_embedding_size, vit_patch_size = _make_backbone(
        encoder_embed_dim, encoder_depth, encoder_num_heads, encoder_global_attn_indexes, image_size
    )

    # 选择标准版 SAM 或多任务版
    if predict_distance or predict_thickness:
        decoder = MaskDecoder_multi(
            **decoder_common_kwargs, predict_distance=predict_distance, predict_thickness=predict_thickness
        )
        sam = Sam_multi(
            image_encoder=encoder,
            prompt_encoder=prompt,
            mask_decoder=decoder,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            predict_distance=predict_distance,
            predict_thickness=predict_thickness,
        )
    else:
        from .modeling import MaskDecoder, Sam  # 标准解码器
        decoder = MaskDecoder(**decoder_common_kwargs)
        sam = Sam(image_encoder=encoder, prompt_encoder=prompt, mask_decoder=decoder, pixel_mean=pixel_mean, pixel_std=pixel_std)

    sam.train()

    # 预训练权重尽可能匹配加载（忽略新加 head）
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            state_dict = torch.load(f)
        sam.load_state_dict(_safe_load(sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes))

    return sam, image_embedding_size




def _build_sam_hq(
    *,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    num_classes,
    image_size,
    pixel_mean,
    pixel_std,
    checkpoint=None,
    predict_distance=False,
    predict_thickness=False,
):
    # backbone & prompt
    encoder, prompt, decoder_common_kwargs, image_embedding_size, vit_patch_size = _make_backbone_hq(
        encoder_embed_dim, encoder_depth, encoder_num_heads, encoder_global_attn_indexes, image_size
    )

    # 选择标准版 SAM 或多任务版
    if predict_distance or predict_thickness:
        decoder = MaskDecoder_multi_hq(
            **decoder_common_kwargs, predict_distance=predict_distance, predict_thickness=predict_thickness
        )
        sam = Sam_multi_hq(
            image_encoder=encoder,
            prompt_encoder=prompt,
            mask_decoder=decoder,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            predict_distance=predict_distance,
            predict_thickness=predict_thickness,
        )
    else:
        from .modeling import MaskDecoder, Sam  # 标准解码器
        decoder = MaskDecoder(**decoder_common_kwargs)
        sam = Sam(image_encoder=encoder, prompt_encoder=prompt, mask_decoder=decoder, pixel_mean=pixel_mean, pixel_std=pixel_std)

    sam.train()

    # 预训练权重尽可能匹配加载（忽略新加 head）
    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            # state_dict = torch.load(f)
            state_dict = torch.load(checkpoint,map_location = "cpu")
        sam.load_state_dict(_safe_load(sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes))

    return sam, image_embedding_size


def _safe_load(sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
    sam_dict = sam.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in sam_dict and sam_dict[k].shape == v.shape:
            new_state_dict[k] = v

    # 处理 pos_embed 尺寸不匹配的情况
    if "image_encoder.pos_embed" in state_dict:
        pos_embed = state_dict["image_encoder.pos_embed"]
        token_size = image_size // vit_patch_size
        if pos_embed.shape[1] != token_size:
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode="bilinear", align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)
            new_state_dict["image_encoder.pos_embed"] = pos_embed

    sam_dict.update(new_state_dict)
    return sam_dict



sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
    "vit_b_distance": build_sam_vit_b_distance,  
    "vit_l_distance": build_sam_vit_l_distance,  
    "vit_h_distance": build_sam_vit_h_distance,  
    "vit_b_distance_thickness": build_sam_vit_b_distance_thickness,
    "vit_b_distance_thickness_hq":build_sam_vit_b_distance_thickness_hq
}



# def _build_sam(
#         encoder_embed_dim,
#         encoder_depth,
#         encoder_num_heads,
#         encoder_global_attn_indexes,
#         num_classes,
#         image_size,
#         pixel_mean,
#         pixel_std,
#         checkpoint=None,
# ):
#     prompt_embed_dim = 256
#     image_size = image_size
#     vit_patch_size = 16
#     image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
#     sam = Sam(
#         image_encoder=ImageEncoderViT(
#             depth=encoder_depth,
#             embed_dim=encoder_embed_dim,
#             img_size=image_size,
#             mlp_ratio=4,
#             norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
#             num_heads=encoder_num_heads,
#             patch_size=vit_patch_size,
#             qkv_bias=True,
#             use_rel_pos=True,
#             global_attn_indexes=encoder_global_attn_indexes,
#             window_size=14,
#             out_chans=prompt_embed_dim,
#         ),
#         prompt_encoder=PromptEncoder(
#             embed_dim=prompt_embed_dim,
#             image_embedding_size=(image_embedding_size, image_embedding_size),
#             input_image_size=(image_size, image_size),
#             mask_in_chans=16,
#         ),
#         mask_decoder=MaskDecoder(
#             # num_multimask_outputs=3,
#             num_multimask_outputs=num_classes,
#             transformer=TwoWayTransformer(
#                 depth=2,
#                 embedding_dim=prompt_embed_dim,
#                 mlp_dim=2048,
#                 num_heads=8,
#             ),
#             transformer_dim=prompt_embed_dim,
#             iou_head_depth=3,
#             iou_head_hidden_dim=256,
#         ),
#         # pixel_mean=[123.675, 116.28, 103.53],
#         # pixel_std=[58.395, 57.12, 57.375],
#         pixel_mean=pixel_mean,
#         pixel_std=pixel_std
#     )
#     # sam.eval()
#     sam.train()
#     if checkpoint is not None:
#         with open(checkpoint, "rb") as f:
#             state_dict = torch.load(f)
#         try:
#             sam.load_state_dict(state_dict)
#         except:
#             new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes)
#             sam.load_state_dict(new_state_dict)
#     return sam, image_embedding_size






# def _build_sam(
#         encoder_embed_dim,
#         encoder_depth,
#         encoder_num_heads,
#         encoder_global_attn_indexes,
#         num_classes,
#         image_size,
#         pixel_mean,
#         pixel_std,
#         checkpoint=None,
#         predict_distance=False,  # 新增参数，指示是否需要预测距离图
# ):
#     prompt_embed_dim = 256
#     image_size = image_size
#     vit_patch_size = 16
#     image_embedding_size = image_size // vit_patch_size  # Divide by 16 here
#     sam = Sam_distance(
#         image_encoder=ImageEncoderViT(
#             depth=encoder_depth,
#             embed_dim=encoder_embed_dim,
#             img_size=image_size,
#             mlp_ratio=4,
#             norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
#             num_heads=encoder_num_heads,
#             patch_size=vit_patch_size,
#             qkv_bias=True,
#             use_rel_pos=True,
#             global_attn_indexes=encoder_global_attn_indexes,
#             window_size=14,
#             out_chans=prompt_embed_dim,
#         ),
#         prompt_encoder=PromptEncoder(
#             embed_dim=prompt_embed_dim,
#             image_embedding_size=(image_embedding_size, image_embedding_size),
#             input_image_size=(image_size, image_size),
#             mask_in_chans=16,
#         ),
#         mask_decoder=MaskDecoder_distance(
#             # num_multimask_outputs=3,
#             num_multimask_outputs=num_classes,
#             transformer=TwoWayTransformer(
#                 depth=2,
#                 embedding_dim=prompt_embed_dim,
#                 mlp_dim=2048,
#                 num_heads=8,
#             ),
#             transformer_dim=prompt_embed_dim,
#             iou_head_depth=3,
#             iou_head_hidden_dim=256,
#             predict_distance=predict_distance, # --- [核心修改 3] 将参数传递给MaskDecoder ---
#         ),
#         # pixel_mean=[123.675, 116.28, 103.53],
#         # pixel_std=[58.395, 57.12, 57.375],
#         pixel_mean=pixel_mean,
#         pixel_std=pixel_std,
#         predict_distance=predict_distance, # --- [核心修改 4] 将参数传递给Sam模型本身 ---
#     )
#     # sam.eval()
#     sam.train()
#     if checkpoint is not None:
#         with open(checkpoint, "rb") as f:
#             state_dict = torch.load(f)
#         try:
#             sam.load_state_dict(state_dict)
#         except:
#             new_state_dict = load_from(sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes)
#             sam.load_state_dict(new_state_dict)
#     return sam, image_embedding_size


# def load_from(sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
#     ega = encoder_global_attn_indexes
#     sam_dict = sam.state_dict()
#     except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
#     # [新增] 如果我们有新的头，也要从预训练权重中移除，避免冲突
#     if 'distance_prediction_head.0.weight' in sam_dict:
#         except_keys.append('distance_prediction_head')
#     new_state_dict = {k: v for k, v in state_dict.items() if
#                       k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
#     pos_embed = new_state_dict['image_encoder.pos_embed']
#     token_size = int(image_size // vit_patch_size)
#     if pos_embed.shape[1] != token_size:
        
#         pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
#         pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
#         pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
#         new_state_dict['image_encoder.pos_embed'] = pos_embed
#         rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]
#         global_rel_pos_keys = []
#         for rel_pos_key in rel_pos_keys:
#             num = int(rel_pos_key.split('.')[2])
#             if num in encoder_global_attn_indexes:
#                 global_rel_pos_keys.append(rel_pos_key)
        
#         for k in global_rel_pos_keys:
#             rel_pos_params = new_state_dict[k]
#             h, w = rel_pos_params.shape
#             rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
#             rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
#             new_state_dict[k] = rel_pos_params[0, 0, ...]
#     sam_dict.update(new_state_dict)
#     return sam_dict








def load_from(sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
    sam_dict = sam.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k not in sam_dict:
            continue  # 跳过不存在的键
        if sam_dict[k].shape != v.shape:
            continue  # 跳过形状不匹配的键
        new_state_dict[k] = v

    # 可选：处理 pos_embed 和 rel_pos 的插值（如果你 image_size 变了）
    if 'image_encoder.pos_embed' in state_dict:
        pos_embed = state_dict['image_encoder.pos_embed']
        token_size = image_size // vit_patch_size
        if pos_embed.shape[1] != token_size:
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1)
            new_state_dict['image_encoder.pos_embed'] = pos_embed

    sam_dict.update(new_state_dict)
    return sam_dict
