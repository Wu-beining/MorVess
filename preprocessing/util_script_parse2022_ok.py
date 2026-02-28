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


def process_thickness_to_5slice(skip_existing=True):
    """
    读取 parse2022_thickness_map/<case>/thickness_map/<case>_thickness_map.nii.gz
    仅生成 thickness_map 的 5-slice .pkl 到
    parse2022/train/2D_all_5slice/<case>/thickness_map/2Dthickness_XXXX.pkl
    """
    thick_root = os.path.join(base_dir, "parse2022/parse2022_thickness_map")
    save_pth   = os.path.join(base_dir, "parse2022/train/2D_all_5slice")

    cases = [d for d in os.listdir(thick_root)
             if d.startswith("PA") and os.path.isdir(os.path.join(thick_root, d))]
    cases.sort()

    for case_id in tqdm(cases, desc="thickness → 5-slice"):
        # 厚度图路径
        t_path = os.path.join(thick_root, case_id, "thickness_map", f"{case_id}_thickness_map.nii.gz")
        if not os.path.exists(t_path):
            print(f"[WARN] {case_id} 缺少厚度图：{t_path}，跳过。")
            continue

        # 读取并构造 5-slice
        t_arr = nib.load(t_path).get_fdata()
        # padding：前2/后2 复制边界切片 —— 与你现有四路一致
        t_pad = np.concatenate((t_arr[:, :, 0:1], t_arr[:, :, 0:1],
                                t_arr,
                                t_arr[:, :, -1:], t_arr[:, :, -1:]), axis=-1)

        out_dir = os.path.join(save_pth, case_id, "thickness_map")
        os.makedirs(out_dir, exist_ok=True)

        for z in range(2, t_pad.shape[2]-2):
            sid = z - 2
            block = t_pad[:, :, z-2:z+3]
            # 与你原来一致的几何变换：先 rot90(k=1, 轴 0-1)，再左右 flip
            block = np.flip(np.rot90(block, k=1, axes=(0, 1)), axis=1)

            out_pkl = os.path.join(out_dir, f"2Dthickness_{sid:04d}.pkl")
            if skip_existing and os.path.exists(out_pkl):
                continue
            tmp = out_pkl + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(block, f)
            os.replace(tmp, out_pkl)

    print("✓ thickness 5-slice 生成完成。")



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
    # process_thickness_to_5slice()
    
    # 步骤2: 生成包含所有路径的统一CSV文件
    get_unified_csv()
