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
python generate_thickness_process.py \
  -i /home/ET/bnwu/MA-SAM/PA000005.nii.gz -o ./thickness_out
  
  
python generate_thickness_process.py \
  -i /home/ET/bnwu/MA-SAM/data/AIIB23_Train_T1/gt/AIIB23_30.nii.gz -o ./aiib_distance_out
  
  
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
                           save_debug=False,
                           save_all_steps=False):
    """
    读取单个 NIfTI 掩模，生成厚度图并保存。
    新增功能：当 save_all_steps=True 时，保存所有中间步骤的结果。

    参数：
      mask_path        : 掩模文件路径（.nii 或 .nii.gz）
      input_root_dir   : 输入数据集根目录（批量时必填）
      output_root_dir  : 输出根目录（批量时必填）
      out_subdir_name  : 每个样本目录下的输出子目录名称
      smooth_sigma_mm  : 仅在掩模内的轻微高斯平滑（单位 mm），默认 0 关闭
      save_debug       : 是否另存骨架半径图/骨架 mask（调试用）
      save_all_steps   : (新增) 是否保存所有中间步骤的结果
    """
    if not os.path.exists(mask_path):
        print(f"[WARN] 文件不存在，跳过：{mask_path}")
        return

    print(f"\n--- 正在处理单个过程详细过程: {mask_path} ---")

    # ---- 组织输出路径（对齐代码1） ----
    base_name = os.path.basename(mask_path)
    base_noext = re.sub(r'\.nii(\.gz)?$', '', base_name, flags=re.I)
    base_noext = base_noext.split('_label')[0]

    if input_root_dir and output_root_dir:
        rel_label_path = os.path.relpath(mask_path, input_root_dir)
        relative_sample_dir = os.path.dirname(os.path.dirname(rel_label_path))
        out_dir = os.path.join(output_root_dir, relative_sample_dir, out_subdir_name)
    else:
        out_dir = output_root_dir if output_root_dir else os.path.dirname(mask_path)

    os.makedirs(out_dir, exist_ok=True)

    # ---- 预备工作：读取图像并准备保存函数 ----
    mask_img = sitk.ReadImage(mask_path, sitk.sitkFloat32)
    spacing = mask_img.GetSpacing()  # (sx, sy, sz)

    def save_intermediate_step(data_arr, step_name):
        """一个便捷的函数，用于保存中间步骤的 .nii.gz 文件"""
        if save_all_steps:
            img = sitk.GetImageFromArray(data_arr.astype(np.float32))
            img.CopyInformation(mask_img)
            filepath = os.path.join(out_dir, f"{base_noext}_{step_name}.nii.gz")
            sitk.WriteImage(img, filepath)
            print(f"  [步骤] 已保存: {os.path.basename(filepath)}")

    # ---- 开始计算和保存中间步骤 ----

    # 步骤 1: 二值化掩模
    mask_arr = (sitk.GetArrayFromImage(mask_img) > 0).astype(np.uint8)
    save_intermediate_step(mask_arr, "step1_binary_mask")

    # 步骤 2: 计算内部距离图 (D_internal)
    # 这是掩模内每个点到最近边界的距离（单位：mm）
    internal_dist = distance_transform_edt(mask_arr, sampling=spacing[::-1])
    save_intermediate_step(internal_dist, "step2_internal_distance_mm")

    # 步骤 3: 细化算法提取骨架 (S)
    thinner = sitk.BinaryThinningImageFilter()
    # 注意：BinaryThinning要求输入为整数类型
    skel_img = thinner.Execute(sitk.Cast(mask_img > 0, sitk.sitkUInt8))
    skel_arr = (sitk.GetArrayFromImage(skel_img) > 0).astype(np.uint8)
    save_intermediate_step(skel_arr, "step3_skeleton_mask")

    # 步骤 4: 计算骨架上每个点的半径 (r(s))
    # 这是内部距离图在骨架位置的值
    if skel_arr.sum() == 0:
        print("[WARN] 未能提取到骨架，厚度图将基于内部距离图直接生成。")
        # 在没有骨架的罕见情况下，进行简化处理
        r_skel = np.zeros_like(internal_dist, dtype=np.float32)
        nearest_r = internal_dist # 直接使用内部距离作为半径
    else:
        r_skel = internal_dist * skel_arr
        save_intermediate_step(r_skel, "step4_skeleton_radius_mm")

        # 步骤 5: 将骨架半径外推至整个掩模区域
        # 掩模内每个点的值，等于其最近骨架点的半径
        arr = (skel_arr == 0).astype(np.uint8)
        _, inds = distance_transform_edt(arr, sampling=spacing[::-1], return_indices=True)
        z_idx, y_idx, x_idx = inds
        nearest_r = r_skel[z_idx, y_idx, x_idx]
        # 只保留掩模内的值
        nearest_r_masked = nearest_r * (mask_arr > 0)
        save_intermediate_step(nearest_r_masked, "step5_nearest_radius_map_mm")

    # 步骤 6: 生成最终厚度图 (VTM = 2 * r)
    thickness = 2.0 * nearest_r
    thickness_masked = thickness * (mask_arr > 0)
    save_intermediate_step(thickness_masked, "step6_thickness_map_raw_mm")

    # 步骤 7 (可选): 在掩模内进行高斯平滑
    if smooth_sigma_mm and smooth_sigma_mm > 0:
        sigma_vox = [smooth_sigma_mm / s for s in spacing[::-1]]
        thickness_smoothed = gaussian_filter(thickness, sigma=sigma_vox, mode='nearest')
        thickness_final = thickness_smoothed * (mask_arr > 0)
        save_intermediate_step(thickness_final, "step7_thickness_map_smoothed_mm")
    else:
        thickness_final = thickness_masked

    # ---- 保存最终结果（与原始脚本行为保持一致） ----
    final_thick_img = sitk.GetImageFromArray(thickness_final.astype(np.float32))
    final_thick_img.CopyInformation(mask_img)
    out_thickness_final_path = os.path.join(out_dir, f"{base_noext}_thickness_map.nii.gz")
    sitk.WriteImage(final_thick_img, out_thickness_final_path)
    print(f"==> 最终厚度图已保存: {out_thickness_final_path}")

    if save_debug:
        # 调试模式下，确保骨架和半径图也被保存
        skel_r_img = sitk.GetImageFromArray(r_skel.astype(np.float32))
        skel_r_img.CopyInformation(mask_img)
        sitk.WriteImage(skel_r_img, os.path.join(out_dir, f"{base_noext}_skeleton_radius_mm.nii.gz"))
        sitk.WriteImage(skel_img, os.path.join(out_dir, f"{base_noext}_skeleton_mask.nii.gz"))

# ---------------- 入口：与代码1一致的“递归批量” ---------------- #

def main():
    parser = argparse.ArgumentParser(description="从 NIfTI 掩模生成血管厚度图，并可选择保存所有中间步骤。")
    parser.add_argument('-i', '--input',  type=str, required=True,
                        help="输入路径：单文件 或 数据集根目录（含若干 <sample>/label/*.nii.gz）")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="输出根目录（批量时会按原层级写入；单文件时直接写到该目录）")
    parser.add_argument('--batch', action='store_true',
                        help="批量模式：从输入根目录递归查找 label/*.nii.gz 并批量处理")
    parser.add_argument('--out_subdir', type=str, default='thickness_map',
                        help="每个样本目录下的输出子目录名（默认 thickness_map）")
    parser.add_argument('--smooth_sigma_mm', type=float, default=0.05,
                        help="仅在掩模内做轻微高斯平滑的 sigma（单位 mm，默认 0.05）")
    parser.add_argument('--save_debug', action='store_true',
                        help="额外保存骨架半径图/骨架 mask（调试用）")
    parser.add_argument('--save_all_steps', action='store_true',default=True,
                        help="（新增）保存所有计算的中间步骤文件，用于展示和调试")
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
                save_debug=args.save_debug,
                save_all_steps=args.save_all_steps
            )
        print("\n全部处理完成。")
    else:
        # 单文件模式
        if not os.path.isfile(args.input):
            raise ValueError(f"单文件模式下，--input 必须是文件：{args.input}")
        os.makedirs(args.output, exist_ok=True)
        generate_thickness_map(
            args.input,
            input_root_dir=None,
            output_root_dir=args.output,
            out_subdir_name=None,
            smooth_sigma_mm=args.smooth_sigma_mm,
            save_debug=args.save_debug,
            save_all_steps=args.save_all_steps
        )
        print("\n处理完成。")

if __name__ == '__main__':
    main()
