#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于中轴+最大内接球生成“血管厚度图”（直径图，单位：mm）。
读取/保存逻辑与【代码1】保持一致：
- 递归从输入数据集根目录下寻找 label/*.nii.gz
- 输出到：<输出根目录>/<样本目录>/<输出子目录名>/

用法示例（批量，最推荐）：
python generate_thickness.py \
  -i /home/ET/bnwu/MA-SAM/data/parse2022/train \
  -o /home/ET/bnwu/MA-SAM/data/parse2022/parse2022_thickness_map \
  --batch --out_subdir thickness_map

也支持单文件模式（直接把结果写到 -o 指定目录）：
python generate_thickness.py \
  -i /path/to/PA000005.nii.gz -o ./thickness_out
"""

import os
import re
import argparse
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt, gaussian_filter

# ---------------- 核心计算：生成厚度图 ---------------- #

def generate_thickness_map(mask_path,
                           input_root_dir=None,
                           output_root_dir=None,
                           out_subdir_name="thickness_map",
                           smooth_sigma_mm=0.0,
                           save_debug=False):
    """
    读取单个 NIfTI 掩模，生成厚度图并保存。
    - 若提供 input_root_dir 和 output_root_dir，则按【代码1】方式保留层级：
        <output_root_dir>/<relative_sample_dir>/<out_subdir_name>/<basename>_thickness_map.nii.gz
      其中 relative_sample_dir = 相对 input_root_dir 的两级上层（样本目录）。
    - 否则（单文件模式），直接写到 output_root_dir 目录。

    参数：
      mask_path        : 掩模文件路径（.nii 或 .nii.gz）
      input_root_dir   : 输入数据集根目录（批量时必填）
      output_root_dir  : 输出根目录（批量时必填）
      out_subdir_name  : 每个样本目录下的输出子目录名称（默认 thickness_map）
      smooth_sigma_mm  : 仅在掩模内的轻微高斯平滑（单位 mm），默认 0 关闭
      save_debug       : 是否另存骨架半径图/骨架 mask（调试用）
    """
    if not os.path.exists(mask_path):
        print(f"[WARN] 文件不存在，跳过：{mask_path}")
        return

    # ---- 读取并二值化 ----
    mask_img = sitk.ReadImage(mask_path, sitk.sitkFloat32)
    mask_arr = (sitk.GetArrayFromImage(mask_img) > 0).astype(np.uint8)
    spacing = mask_img.GetSpacing()  # (sx, sy, sz)

    # 1) 内部距离（到边界，单位 mm；注意 scipy 轴顺序为 z,y,x）
    internal_dist = distance_transform_edt(mask_arr, sampling=spacing[::-1])

    # 2) 细化得到骨架
    thinner = sitk.BinaryThinningImageFilter()
    skel_img = thinner.Execute(sitk.Cast(mask_img, sitk.sitkUInt8))
    skel_arr = (sitk.GetArrayFromImage(skel_img) > 0).astype(np.uint8)

    # 3) 从骨架处半径外推到所有体素：T(x) = 2 * r(nearest_skeleton(x))
    if skel_arr.sum() == 0:
        thickness = 2.0 * internal_dist * (mask_arr > 0)
        r_skel = np.zeros_like(internal_dist, dtype=np.float32)
    else:
        r_skel = internal_dist * skel_arr  # 骨架半径（mm）
        # 最近骨架索引（利用 edt 的 return_indices）
        arr = (skel_arr == 0).astype(np.uint8)  # 骨架=0，其余=1
        _, inds = distance_transform_edt(arr, sampling=spacing[::-1], return_indices=True)
        z_idx, y_idx, x_idx = inds
        nearest_r = r_skel[z_idx, y_idx, x_idx]
        thickness = np.zeros_like(internal_dist, dtype=np.float32)
        thickness[mask_arr > 0] = 2.0 * nearest_r[mask_arr > 0]

    # 4)（可选）仅在掩模内平滑
    if smooth_sigma_mm and smooth_sigma_mm > 0:
        sigma_vox = [smooth_sigma_mm / s for s in spacing[::-1]]
        thickness = gaussian_filter(thickness, sigma=sigma_vox, mode='nearest') * (mask_arr > 0)

    # ---- 组织输出路径（对齐代码1） ----
    base_name = os.path.basename(mask_path)
    # 兼容诸如 *_label.nii.gz 的命名；若没有 _label 也正常
    base_noext = re.sub(r'\.nii(\.gz)?$', '', base_name, flags=re.I)
    base_noext = base_noext.split('_label')[0]  # 与代码1一致的切法

    if input_root_dir and output_root_dir:
        # 计算相对样本目录： <sample>/label/<file>  -> 取 <sample>
        rel_label_path = os.path.relpath(mask_path, input_root_dir)
        # 去掉最后两级（label/文件名）后，得到样本目录
        relative_sample_dir = os.path.dirname(os.path.dirname(rel_label_path))
        out_dir = os.path.join(output_root_dir, relative_sample_dir, out_subdir_name)
    else:
        # 单文件模式：直接写到 output_root_dir
        out_dir = output_root_dir if output_root_dir else os.path.dirname(mask_path)

    os.makedirs(out_dir, exist_ok=True)

    # 主结果：厚度图
    thick_img = sitk.GetImageFromArray(thickness.astype(np.float32))
    thick_img.CopyInformation(mask_img)
    out_thickness = os.path.join(out_dir, f"{base_noext}_thickness_map.nii.gz")
    sitk.WriteImage(thick_img, out_thickness)
    print("Thickness map saved:", out_thickness)

    if save_debug:
        skel_r_img = sitk.GetImageFromArray(r_skel.astype(np.float32))
        skel_r_img.CopyInformation(mask_img)
        sitk.WriteImage(skel_r_img, os.path.join(out_dir, f"{base_noext}_skeleton_radius_mm.nii.gz"))
        sitk.WriteImage(skel_img,    os.path.join(out_dir, f"{base_noext}_skeleton_mask.nii.gz"))

# ---------------- 入口：与代码1一致的“递归批量” ---------------- #

def main():
    parser = argparse.ArgumentParser(description="从 NIfTI 掩模生成血管厚度图（读取/保存逻辑对齐代码1）")
    parser.add_argument('-i', '--input',  type=str, required=True,
                        help="输入路径：单文件 或 数据集根目录（含若干 <sample>/label/*.nii.gz）")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="输出根目录（批量时会按原层级写入；单文件时直接写到该目录）")
    parser.add_argument('--batch', action='store_true',
                        help="批量模式：从输入根目录递归查找 label/*.nii.gz 并批量处理")
    parser.add_argument('--out_subdir', type=str, default='thickness_map',
                        help="每个样本目录下的输出子目录名（默认 thickness_map）")
    parser.add_argument('--smooth_sigma_mm', type=float, default=0.05,
                        help="仅在掩模内做轻微高斯平滑的 sigma（单位 mm，默认 0 关闭）")
    parser.add_argument('--save_debug', action='store_true',
                        help="额外保存骨架半径图/骨架 mask（调试用）")
    args = parser.parse_args()

    if args.batch:
        if not os.path.isdir(args.input):
            raise ValueError(f"批量模式下，--input 必须是目录：{args.input}")
        files = []
        for root, dirs, fs in os.walk(args.input):
            if os.path.basename(root) == 'label':
                for f in fs:
                    if f.endswith('.nii') or f.endswith('.nii.gz'):
                        files.append(os.path.join(root, f))
        if not files:
            raise RuntimeError(f"未在 {args.input} 下找到任何 label/*.nii[.gz] 文件")

        print(f"共找到 {len(files)} 个 label 掩模，开始处理……")
        for p in files:
            generate_thickness_map(
                p,
                input_root_dir=args.input,
                output_root_dir=args.output,
                out_subdir_name=args.out_subdir,
                smooth_sigma_mm=args.smooth_sigma_mm,
                save_debug=args.save_debug
            )
        print("全部处理完成。")
    else:
        # 单文件模式：直接写到输出目录（不保留层级）
        if not os.path.isfile(args.input):
            raise ValueError(f"单文件模式下，--input 必须是文件：{args.input}")
        os.makedirs(args.output, exist_ok=True)
        generate_thickness_map(
            args.input,
            input_root_dir=None,
            output_root_dir=args.output,
            out_subdir_name=None,           # 单文件模式忽略
            smooth_sigma_mm=args.smooth_sigma_mm,
            save_debug=args.save_debug
        )

if __name__ == '__main__':
    main()
