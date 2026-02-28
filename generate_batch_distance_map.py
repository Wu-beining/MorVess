# -*- coding: utf-8 -*-
"""
此脚本用于为数据集批量生成血管距离图，并将其保存在一个独立的输出目录中，
同时保持原始的目录层次结构。

V4版本：移除了所有命令行参数，改用在脚本内直接指定路径和参数的方式，
以避免环境和命令行使用问题。

请确保已安装必要的库:
pip install SimpleITK numpy scipy tqdm
"""
import os
import sys
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

def generate_distance_maps(mask_path, input_root_dir, output_root_dir, lambda_param, output_folder_name):
    """
    从一个 NIfTI 掩模文件生成血管距离图，并保存在指定的输出目录结构中。
    (此函数内部逻辑保持不变)
    """
    if not os.path.exists(mask_path):
        tqdm.write(f"警告：输入文件不存在，已跳过 -> {mask_path}")
        return

    try:
        relative_label_path = os.path.relpath(mask_path, input_root_dir)
        relative_sample_dir = os.path.dirname(os.path.dirname(relative_label_path))
        output_dir = os.path.join(output_root_dir, relative_sample_dir, output_folder_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        mask_image = sitk.ReadImage(mask_path, sitk.sitkFloat32)
        mask_array = sitk.GetArrayFromImage(mask_image)
        mask_array = (mask_array > 0).astype(np.uint8)
        spacing = mask_image.GetSpacing()

        # 生成内部距离图
        internal_dist_array = distance_transform_edt(mask_array, sampling=spacing)
        internal_dist_image = sitk.GetImageFromArray(internal_dist_array)
        internal_dist_image.CopyInformation(mask_image)

        # 生成边界势场图
        eroded_mask_image = sitk.BinaryErode(sitk.Cast(mask_image, sitk.sitkUInt8), [1, 1, 1])
        eroded_mask_array = sitk.GetArrayFromImage(eroded_mask_image)
        boundary_array = mask_array - eroded_mask_array
        boundary_dist_array = distance_transform_edt(boundary_array == 0, sampling=spacing)
        
        potential_map_array = np.exp(-lambda_param * boundary_dist_array)
        
        potential_map_image = sitk.GetImageFromArray(potential_map_array)
        potential_map_image.CopyInformation(mask_image)
        
        # 保存结果
        base_name = os.path.basename(mask_path).split('_label.')[0]
        output_path_internal = os.path.join(output_dir, f"{base_name}_internal_distance.nii.gz")
        sitk.WriteImage(internal_dist_image, output_path_internal)
        output_path_potential = os.path.join(output_dir, f"{base_name}_boundary_potential.nii.gz")
        sitk.WriteImage(potential_map_image, output_path_potential)

    except Exception as e:
        tqdm.write(f"处理文件 {mask_path} 时发生错误: {e}")

def main():
    # ===================================================================
    # --- 请在这里修改您的配置 ---
    # ===================================================================
    
    # 1. 输入数据集的根目录路径
    #    请确保路径末尾有斜杠 '/'
    INPUT_DATA_DIR = "data/parse2022/train/"
    
    # 2. 保存所有结果的输出根目录
    OUTPUT_DIR = "parse2022_distance_map"
    
    # 3. 边界势场图的指数衰减系数 (通常不需要修改)
    LAMBDA_PARAM = 0.05
    
    # 4. 在每个样本目录中创建的用于存放结果的文件夹名称 (通常不需要修改)
    OUTPUT_FOLDER_NAME = 'potential_map'

    # ===================================================================
    # --- 配置结束，以下代码无需修改 ---
    # ===================================================================

    if not os.path.isdir(INPUT_DATA_DIR):
        print(f"错误: 输入路径不是一个有效的目录 -> {INPUT_DATA_DIR}")
        sys.exit(1)
        
    print(f"开始在目录 {INPUT_DATA_DIR} 中搜索 label 文件...")
    
    label_files_to_process = []
    for root, dirs, files in os.walk(INPUT_DATA_DIR):
        if os.path.basename(root) == 'label':
            for file in files:
                if file.endswith('.nii.gz'):
                    label_files_to_process.append(os.path.join(root, file))
                    
    if not label_files_to_process:
        print(f"错误: 未能在目录 '{INPUT_DATA_DIR}' 下找到任何 'label/*.nii.gz' 文件。请检查路径和文件结构。")
        sys.exit(1)

    print(f"找到 {len(label_files_to_process)} 个 label 文件。开始批量处理...")
    for file_path in tqdm(label_files_to_process, desc="生成距离图", file=sys.stdout):
        generate_distance_maps(file_path, INPUT_DATA_DIR, OUTPUT_DIR, LAMBDA_PARAM, OUTPUT_FOLDER_NAME)
    print("\n所有文件处理完毕！")

# --- 直接运行主函数 ---
if __name__ == '__main__':
    main()