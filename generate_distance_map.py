# -*- coding: utf-8 -*-
"""
此脚本用于处理 3D NIfTI 格式的血管分割掩模（mask），并生成两种类型的血管距离图：
1. 内部距离图 (Internal Distance Map): 仅在血管内部显示到背景的最短距离，反映血管的中心线和厚度。
2. 边界势场图 (Boundary Potential Map): 在血管边界处值最高，向内外平滑衰减，用于引导网络学习边界。

请确保已安装必要的库:
pip install SimpleITK numpy scipy
"""
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt
import os
import argparse

def generate_distance_maps(mask_path, output_dir, lambda_param=0.05):
    """
    从一个 NIfTI 格式的掩模文件生成血管距离图。

    参数:
    - mask_path (str): 输入的掩模文件路径 (.nii.gz)。
    - output_dir (str): 输出文件的保存目录。
    - lambda_param (float): 边界势场图中指数衰减的系数，可根据需要调整。
                           值越小，势场范围越广；值越大，势场越集中在边界。
    """
    if not os.path.exists(mask_path):
        print(f"错误：输入文件不存在 -> {mask_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    print(f"正在处理文件: {mask_path}")

    # 1. 使用 SimpleITK 加载 NIfTI 文件
    # SimpleITK 可以很好地保留图像的元数据（如spacing, origin, direction）
    mask_image = sitk.ReadImage(mask_path, sitk.sitkFloat32)
    mask_array = sitk.GetArrayFromImage(mask_image)

    # 确保掩模是二值的 (0 或 1)
    mask_array = (mask_array > 0).astype(np.uint8)

    # --- 生成内部距离图 (Internal Distance Map) ---
    # 计算从每个前景点（血管）到最近背景点（非血管）的欧几里得距离。
    # 我们需要获取图像的spacing信息，使得距离计算是物理距离（例如毫米），而不是像素单位。
    spacing = mask_image.GetSpacing()
    
    # distance_transform_edt 会计算每个非零点到最近零点的距离
    # 这正是我们想要的内部距离图
    internal_dist_array = distance_transform_edt(mask_array, sampling=spacing)
    
    # 将结果转换回 SimpleITK 图像
    internal_dist_image = sitk.GetImageFromArray(internal_dist_array)
    internal_dist_image.CopyInformation(mask_image) # 复制所有元数据

    # 保存内部距离图
    base_name = os.path.basename(mask_path).replace(".nii.gz", "")
    output_path_internal = os.path.join(output_dir, f"{base_name}_internal_distance.nii.gz")
    sitk.WriteImage(internal_dist_image, output_path_internal)
    print(f"已保存内部距离图到: {output_path_internal}")


    # --- 生成边界势场图 (Boundary Potential Map) ---
    # 灵感来源于您提供的文献 "Automatic kidney segmentation..."
    
    # a. 提取边界 (原始mask - 腐蚀后的mask)
    # sitk.BinaryErode 需要一个半径参数，这里我们使用一个像素单位的球形结构元素
    # 半径单位是物理单位，所以我们需要根据spacing来定
    # 为了简单和通用，我们用一个像素的腐蚀
    eroded_mask_image = sitk.BinaryErode(sitk.Cast(mask_image, sitk.sitkUInt8), [1, 1, 1])
    eroded_mask_array = sitk.GetArrayFromImage(eroded_mask_image)
    
    boundary_array = mask_array - eroded_mask_array
    
    # b. 计算到边界的距离
    # 我们希望计算每个点到边界的距离，所以边界点应该是0，其他点非0
    # distance_transform_edt 计算的是非0点到0点的距离
    # 所以我们需要反转边界图 (boundary_array == 0)
    boundary_dist_array = distance_transform_edt(boundary_array == 0, sampling=spacing)
    
    # c. 使用指数函数进行归一化，生成势场
    # D(p) = exp(-lambda * dist(p, boundary))
    potential_map_array = np.exp(-lambda_param * boundary_dist_array)
    
    # 将结果转换回 SimpleITK 图像
    potential_map_image = sitk.GetImageFromArray(potential_map_array)
    potential_map_image.CopyInformation(mask_image)

    # 保存边界势场图
    output_path_potential = os.path.join(output_dir, f"{base_name}_boundary_potential.nii.gz")
    sitk.WriteImage(potential_map_image, output_path_potential)
    print(f"已保存边界势场图到: {output_path_potential}")
    print("-" * 30)

if __name__ == '__main__':
    # --- 使用方法 ---
    # 1. 直接在代码中修改文件路径
    # input_mask_file = "data/parse2022/train/PA000005/label/PA000005.nii.gz"
    # output_directory = "pa005_label_distance_map.nii.gz"
    # generate_distance_maps(input_mask_file, output_directory)

    # 2. 使用命令行参数 (推荐)
    # 在终端中运行:
    # python your_script_name.py -i /path/to/mask.nii.gz -o /path/to/output
    # python your_script_name.py -i /path/to/mask_folder -o /path/to/output_folder --batch
    
    parser = argparse.ArgumentParser(description="从NIfTI掩模生成血管距离图")
    parser.add_argument('-i', '--input', type=str, required=True, help="输入的掩模文件或文件夹路径。")
    parser.add_argument('-o', '--output', type=str, required=True, help="输出结果的保存目录。")
    parser.add_argument('-l', '--lambda_param', type=float, default=0.05, help="边界势场图的指数衰减系数。")
    parser.add_argument('--batch', action='store_true', help="如果输入是文件夹，则启用此参数以批量处理所有.nii.gz文件。")
    
    args = parser.parse_args()

    if args.batch:
        if not os.path.isdir(args.input):
            print(f"错误：批量处理模式下，输入路径必须是一个文件夹: {args.input}")
        else:
            nii_files = [f for f in os.listdir(args.input) if f.endswith(".nii.gz")]
            if not nii_files:
                print(f"错误：在文件夹 {args.input} 中未找到.nii.gz文件。")
            else:
                print(f"找到 {len(nii_files)} 个 .nii.gz 文件，开始批量处理...")
                for file_name in nii_files:
                    file_path = os.path.join(args.input, file_name)
                    generate_distance_maps(file_path, args.output, args.lambda_param)
                print("所有文件处理完毕！")
    else:
        if not os.path.isfile(args.input):
            print(f"错误：单个文件处理模式下，输入路径必须是一个文件: {args.input}")
        else:
            generate_distance_maps(args.input, args.output, args.lambda_param)

