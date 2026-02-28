# -*- coding: utf-8 -*-
"""
此脚本用于为数据集批量生成血管距离图，并将其保存在一个独立的输出目录中，
同时保持原始的目录层次结构。

V4版本：移除了所有命令行参数，改用在脚本内直接指定路径和参数的方式，
以避免环境和命令行使用问题。

请确保已安装必要的库:
pip install SimpleITK numpy scipy tqdm

也支持单文件模式（直接把结果写到 -o 指定目录）：
python generate_distance_process.py \
  -i /home/ET/bnwu/MA-SAM/PA000005.nii.gz -o ./distance_out
  
  
  
python generate_distance_process.py \
  -i /home/ET/bnwu/MA-SAM/data/AIIB23_Train_T1/gt/AIIB23_30.nii.gz -o ./aiib_distance_out

"""
#   
import os
import re
import argparse
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

# ---------------- 核心计算：生成边界势场图 ---------------- #

def generate_potential_map(mask_path,
                           input_root_dir=None,
                           output_root_dir=None,
                           out_subdir_name="potential_map",
                           lambda_param=0.05,
                           save_all_steps=False):
    """
    读取单个 NIfTI 掩模，生成距离图和边界势场图。
    新增功能：当 save_all_steps=True 时，保存所有中间步骤的结果。

    参数：
      mask_path        : 掩模文件路径（.nii 或 .nii.gz）
      input_root_dir   : 输入数据集根目录（批量时必填）
      output_root_dir  : 输出根目录（批量时必填）
      out_subdir_name  : 每个样本目录下的输出子目录名称
      lambda_param     : 边界势场图的指数衰减系数
      save_all_steps   : (新增) 是否保存所有中间步骤的结果
    """
    if not os.path.exists(mask_path):
        print(f"[WARN] 文件不存在，跳过：{mask_path}")
        return

    print(f"\n--- 正在处理: {mask_path} ---")

    # ---- 组织输出路径 ----
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

    # 步骤 2: 计算内部距离图 (Internal Distance Map)
    # 这是掩模内每个点到最近边界的距离（单位：mm）
    internal_dist_arr = distance_transform_edt(mask_arr, sampling=spacing[::-1])
    save_intermediate_step(internal_dist_arr, "step2_internal_distance_mm")
    
    # 步骤 3: 腐蚀掩模
    # 使用 sitk.BinaryErode 创建一个比原始掩模小一圈的掩模
    eroded_mask_img = sitk.BinaryErode(sitk.Cast(mask_img > 0, sitk.sitkUInt8), [1, 1, 1])
    eroded_mask_arr = sitk.GetArrayFromImage(eroded_mask_img)
    save_intermediate_step(eroded_mask_arr, "step3_eroded_mask")

    # 步骤 4: 提取边界 (Boundary Mask)
    # 原始掩模减去腐蚀后的掩模，得到的就是边界
    boundary_arr = mask_arr - eroded_mask_arr
    save_intermediate_step(boundary_arr, "step4_boundary_mask")

    # 步骤 5: 计算到边界的距离图 (Boundary Distance Map)
    # 这是图像中所有点到“边界”的最近距离
    boundary_dist_arr = distance_transform_edt(boundary_arr == 0, sampling=spacing[::-1])
    save_intermediate_step(boundary_dist_arr, "step5_boundary_distance_mm")

    # 步骤 6: 生成边界势场图 (Boundary Potential Map)
    potential_map_arr = np.exp(-lambda_param * boundary_dist_arr)
    save_intermediate_step(potential_map_arr, "step6_boundary_potential_map")

    # ---- 保存最终结果（与原始脚本行为保持一致） ----
    # 1. 内部距离图
    internal_dist_img = sitk.GetImageFromArray(internal_dist_arr.astype(np.float32))
    internal_dist_img.CopyInformation(mask_img)
    out_internal_dist_path = os.path.join(out_dir, f"{base_noext}_internal_distance.nii.gz")
    sitk.WriteImage(internal_dist_img, out_internal_dist_path)
    print(f"==> 最终内部距离图已保存: {out_internal_dist_path}")

    # 2. 边界势场图
    potential_map_img = sitk.GetImageFromArray(potential_map_arr.astype(np.float32))
    potential_map_img.CopyInformation(mask_img)
    out_potential_map_path = os.path.join(out_dir, f"{base_noext}_boundary_potential.nii.gz")
    sitk.WriteImage(potential_map_img, out_potential_map_path)
    print(f"==> 最终边界势场图已保存: {out_potential_map_path}")


# ---------------- 入口：与之前脚本一致的“递归批量” ---------------- #

def main():
    parser = argparse.ArgumentParser(description="从 NIfTI 掩模生成距离图和边界势场图，并可选择保存所有中间步骤。")
    parser.add_argument('-i', '--input',  type=str, required=True,
                        help="输入路径：单文件 或 数据集根目录（含若干 <sample>/label/*.nii.gz）")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="输出根目录（批量时会按原层级写入；单文件时直接写到该目录）")
    parser.add_argument('--batch', action='store_true',
                        help="批量模式：从输入根目录递归查找 label/*.nii.gz 并批量处理")
    parser.add_argument('--out_subdir', type=str, default='potential_map',
                        help="每个样本目录下的输出子目录名（默认 potential_map）")
    parser.add_argument('--lambda', type=float, default=0.05, dest='lambda_param',
                        help="边界势场图的指数衰减系数 (默认 0.05)")
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
            generate_potential_map(
                p,
                input_root_dir=args.input,
                output_root_dir=args.output,
                out_subdir_name=args.out_subdir,
                lambda_param=args.lambda_param,
                save_all_steps=args.save_all_steps
            )
        print("\n全部处理完成。")
    else:
        # 单文件模式
        if not os.path.isfile(args.input):
            raise ValueError(f"单文件模式下，--input 必须是文件：{args.input}")
        os.makedirs(args.output, exist_ok=True)
        generate_potential_map(
            args.input,
            input_root_dir=None,
            output_root_dir=args.output,
            out_subdir_name=None,
            lambda_param=args.lambda_param,
            save_all_steps=args.save_all_steps
        )
        print("\n处理完成。")

if __name__ == '__main__':
    main()
