import shutil
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
from torch.nn import functional as F
import pickle
import random
import pandas as pd
from tqdm import tqdm
import cv2
import SimpleITK as sitk
from scipy.ndimage import zoom

# --- 配置路径 ---
# 假设脚本文件位于项目的某个子目录中
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将 base_dir 指向项目根目录下的 'data' 文件夹
base_dir = os.path.abspath(os.path.join(current_dir, "../data"))


def process_all_data_to_5slice():
    """
    [统一处理函数]
    将3D图像、掩码以及两种距离图卷都处理成2D切片，
    每个样本包含5个切片上下文(前2个, 当前, 后2个)。
    """
    # 输入路径: 包含所有原始nii.gz格式数据的文件夹
    data_pth = os.path.join(base_dir, "parse2022/parse2022_thickness_map")
    # 输出路径: 保存所有处理后的.pkl切片文件
    save_pth = os.path.join(base_dir, "parse2022/train/2D_all_5slice")

    print(f"数据源路径: {data_pth}")
    print(f"输出保存路径: {save_pth}")

    # 获取所有病例文件夹
    patient_dirs = [d for d in os.listdir(data_pth) if d.startswith("PA") and os.path.isdir(os.path.join(data_pth, d))]
    patient_dirs.sort()

    for patient_dir in tqdm(patient_dirs, desc="正在处理病例"):
        case_id = patient_dir
        
        # --- 确保所有目标目录都存在 ---
        # os.makedirs(os.path.join(save_pth, case_id, 'images'), exist_ok=True)
        # os.makedirs(os.path.join(save_pth, case_id, 'masks'), exist_ok=True)
        # os.makedirs(os.path.join(save_pth, case_id, 'boundary_potential'), exist_ok=True)
        # os.makedirs(os.path.join(save_pth, case_id, 'internal_distance'), exist_ok=True)
        os.makedirs(os.path.join(save_pth, case_id, 'thickness_map'), exist_ok=True)
        
        # --- 加载所有四种数据 ---
        # 1. 图像
        img_path = os.path.join(data_pth, patient_dir, "image", f"{patient_dir}.nii.gz")
        if not os.path.exists(img_path):
            print(f"警告: 在 {patient_dir} 中未找到 image 文件，跳过此病例。")
            # continue
        img_arr = nib.load(img_path).get_fdata()

        # 2. 掩码
        mask_path = os.path.join(data_pth, patient_dir, "label", f"{patient_dir}.nii.gz")
        if not os.path.exists(mask_path):
            print(f"警告: 在 {patient_dir} 中未找到 label 文件，跳过此病例。")
            # continue
        mask_arr = nib.load(mask_path).get_fdata()

        # 3. 边界势能图 (智能查找)
        potential_map_folder = os.path.join(data_pth, patient_dir, 'potential_map')
        boundary_files = [f for f in os.listdir(potential_map_folder) if f.endswith('_boundary_potential.nii.gz')]
        if len(boundary_files) != 1:
            print(f"警告: 在 {patient_dir}/potential_map 中找到 {len(boundary_files)} 个边界图，应为1个。跳过。")
            # continue
        boundary_pot_arr = nib.load(os.path.join(potential_map_folder, boundary_files[0])).get_fdata()
        
        # 4. 内部距离图 (智能查找)
        internal_files = [f for f in os.listdir(potential_map_folder) if f.endswith('_internal_distance.nii.gz')]
        if len(internal_files) != 1:
            print(f"警告: 在 {patient_dir}/potential_map 中找到 {len(internal_files)} 个内部距离图，应为1个。跳过。")
            # continue
        internal_dist_arr = nib.load(os.path.join(potential_map_folder, internal_files[0])).get_fdata()
        
        thickness_map_path = os.path.join(data_pth, patient_dir, 'thickness_map', f"{patient_dir}_thickness_map.nii.gz")
        if not os.path.exists(thickness_map_path):
            print(f"警告: 在 {patient_dir} 中未找到 thickness_map 文件，跳过此病例。")
            continue
        thickness_map_arr = nib.load(thickness_map_path).get_fdata()

        # --- 对所有四种数据进行同样的padding和变换操作 ---
        all_arrays = [img_arr, mask_arr, boundary_pot_arr, internal_dist_arr, thickness_map_arr]
        padded_arrays = [np.concatenate((arr[:, :, 0:1], arr[:, :, 0:1], arr, arr[:, :, -1:], arr[:, :, -1:]), axis=-1) for arr in all_arrays]
        
        # 逐切片处理
        for slice_indx in range(2, padded_arrays[0].shape[2]-2):
            slice_num = slice_indx - 2
            
            # 提取5个切片并进行同样的变换
            slice_blocks = [arr[:,:,slice_indx-2 : slice_indx+3] for arr in padded_arrays]
            transformed_blocks = [np.flip(np.rot90(block, k=1, axes=(0, 1)), axis=1) for block in slice_blocks]

            # --- 保存所有四种处理后的数据 ---
            paths_and_data = [
                (os.path.join(save_pth, case_id, 'images', f'2Dimage_{slice_num:04d}.pkl'), transformed_blocks[0]),
                (os.path.join(save_pth, case_id, 'masks', f'2Dmask_{slice_num:04d}.pkl'), transformed_blocks[1]),
                (os.path.join(save_pth, case_id, 'boundary_potential', f'2Dboundary_{slice_num:04d}.pkl'), transformed_blocks[2]),
                (os.path.join(save_pth, case_id, 'internal_distance', f'2Dinternal_{slice_num:04d}.pkl'), transformed_blocks[3]),
                (os.path.join(save_pth, case_id, 'thickness_map', f'2Dthickness_{slice_num:04d}.pkl'), transformed_blocks[4])
            ]

            for path, data in paths_and_data:
                with open(path, 'wb') as file:
                    pickle.dump(data, file)


def get_unified_csv():
    """
    [统一生成函数]
    生成包含全部四种数据路径的 training.csv 和 test.csv 文件
    """
    save_pth = os.path.join(base_dir, "parse2022/train/2D_all_5slice")
    
    training_csv = os.path.join(save_pth, 'training.csv')
    test_csv = os.path.join(save_pth, 'test.csv')

    data_fd_list = [d for d in os.listdir(save_pth) if d.startswith('PA') and os.path.isdir(os.path.join(save_pth, d))]
    
    for _ in range(5):
        random.shuffle(data_fd_list)

    test_fd_list = ['PA000005', 'PA000016', 'PA000024', 'PA000026', 'PA000027', 'PA000036']
    training_fd_list = list(set(data_fd_list) - set(test_fd_list))

    # --- 创建训练集CSV ---
    path_list_all_train = []
    for data_fd in training_fd_list:
        # 以 'images' 文件夹为基准生成路径
        slice_list = os.listdir(os.path.join(save_pth, data_fd, 'images'))
        slice_pth_list = [os.path.join(data_fd, 'images', slice_name) for slice_name in slice_list]
        path_list_all_train.extend(slice_pth_list)
    
    for _ in range(5):
        random.shuffle(path_list_all_train)
    
    # --- 创建包含全部四列的DataFrame ---
    df_train = pd.DataFrame(path_list_all_train, columns=['image_pth'])
    df_train['mask_pth'] = df_train['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'masks/2Dmask_'))
    df_train['boundary_pth'] = df_train['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'boundary_potential/2Dboundary_'))
    df_train['distance_pth'] = df_train['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'internal_distance/2Dinternal_'))
    df_train['thickness_map_pth'] = df_train['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'thickness_map/2Dthickness_'))
    df_train.to_csv(training_csv, index=False)
    print(f"训练集CSV已生成: {training_csv}, 包含 {len(df_train)} 条记录。")

    # --- 创建测试集CSV ---
    path_list_all_test = []
    for data_fd in test_fd_list:
        slice_list = os.listdir(os.path.join(save_pth, data_fd, 'images'))
        slice_list.sort()
        slice_pth_list = [os.path.join(data_fd, 'images', slice_name) for slice_name in slice_list]
        path_list_all_test.extend(slice_pth_list)

    df_test = pd.DataFrame(path_list_all_test, columns=['image_pth'])
    df_test['mask_pth'] = df_test['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'masks/2Dmask_'))
    df_test['boundary_pth'] = df_test['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'boundary_potential/2Dboundary_'))
    df_test['distance_pth'] = df_test['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'internal_distance/2Dinternal_'))
    df_test['thickness_map_pth'] = df_test['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'thickness_map/2Dthickness_'))
    df_test.to_csv(test_csv, index=False)
    print(f"测试集CSV已生成: {test_csv}, 包含 {len(df_test)} 条记录。")


if __name__ == "__main__":
    # 步骤1: 统一处理所有数据
    process_all_data_to_5slice()
    
    # 步骤2: 生成包含所有路径的统一CSV文件
    # get_unified_csv()
