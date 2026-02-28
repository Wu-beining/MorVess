import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic
import pandas as pd
import pickle
import PIL.Image
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
import cv2

# --- 全局参数 ---
HU_min, HU_max = -500, 500
# 使用您数据集的统计值
data_mean_parse = 53.4296
data_std_parse = 68.6344

def read_pkl_file(path):
    """通用函数，用于从.pkl文件读取数据"""
    with open(path, 'rb') as file:
        data = pickle.load(file)
        return data

# --- 数据增强函数 (已修改以处理多输入) ---

def random_rot_flip(sample):
    """对样本中的所有数据应用同样的旋转和翻转"""
    k = np.random.randint(0, 4)
    axis = np.random.randint(0, 2)
    
    # 对字典中的每个数据项应用相同的变换
    for key in sample.keys():
        sample[key] = np.rot90(sample[key], k, axes=(0, 1))
        sample[key] = np.flip(sample[key], axis=axis).copy()
    return sample

def random_rotate(sample):
    """对样本中的所有数据应用同样的随机旋转"""
    angle = np.random.randint(-15, 15)
    for key in sample.keys():
        # 对图像使用高质量的三次插值
        # 对其他图（掩码、距离图）使用最近邻插值，以避免产生新的像素值
        order = 3 if key == 'image' else 0
        sample[key] = ndimage.rotate(sample[key], angle, order=order, reshape=False)
    return sample

def adjust_light(image):
    """只对图像进行光照调整"""
    image = image*255.0
    gamma = random.random() * 3 + 0.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    for slice_indx in range(image.shape[2]):
        img_curr = image[:,:,slice_indx]
        img_curr = cv2.LUT(np.array(img_curr).astype(np.uint8), table).astype(np.uint8)
        image[:,:,slice_indx] = img_curr
    image = image/255.0
    return image

class RandomGenerator(object):
    """
    数据增强类，现在可以处理包含四种数据类型的样本字典。
    """
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res
        # 您可以保留原来的其他数据增强函数，并在此处类似地调用它们
        # ...

    def __call__(self, sample):
        # sample 是一个字典: {'image':..., 'label':..., 'boundary':..., 'distance':...}
        
        # --- 关键修改：对整个sample字典应用几何变换 ---
        if random.random() > 0.5:
            sample = random_rot_flip(sample)
        if random.random() > 0.5:
            sample = random_rotate(sample)
        
        # --- 仅对图像应用光照和颜色变换 ---
        if random.random() > 0.5:
            sample['image'] = adjust_light(sample['image'])

        # ...在这里可以添加更多的数据增强调用...

        # --- 缩放和类型转换 ---
        image = sample['image']
        label = sample['label']
        boundary = sample['boundary']
        distance = sample['distance']
        thickness = sample['thickness']

        x, y, z = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            # 对图像使用高质量插值，对其他图使用最近邻插值
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=0)
            boundary = zoom(boundary, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=0)
            distance = zoom(distance, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=1)
            thickness = zoom(thickness, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=1)

        # 为低分辨率损失准备标签
        label_h, label_w, label_d = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w, 1.0), order=0)
        
        # 转换为Tensor并调整维度
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32)).permute(2, 0, 1)
        boundary = torch.from_numpy(boundary.astype(np.float32)).permute(2, 0, 1)
        distance = torch.from_numpy(distance.astype(np.float32)).permute(2, 0, 1)
        thickness = torch.from_numpy(thickness.astype(np.float32)).permute(2, 0, 1)
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32)).permute(2, 0, 1)
        
        # 返回一个包含所有数据的字典
        return {
            'image': image, 
            'label': label.long(), 
            'low_res_label': low_res_label.long(),
            'boundary': boundary,
            'distance': distance,
            'thickness':thickness
        }

class dataset_reader_parse(Dataset):
    def __init__(self, base_dir, split, num_classes, transform=None):
        self.transform = transform 
        self.split = split
        self.data_dir = base_dir
        self.num_classes = num_classes

        if split == "train":
            # --- [修改点 1] 读取包含所有路径的CSV文件 ---
            csv_path = os.path.join(self.data_dir, 'training.csv')
            df = pd.read_csv(csv_path)
            
            # 使用os.path.join来构建绝对路径，更稳健
            self.image_paths = [os.path.join(self.data_dir, p) for p in df["image_pth"]]
            self.mask_paths = [os.path.join(self.data_dir, p) for p in df["mask_pth"]]
            self.boundary_paths = [os.path.join(self.data_dir, p) for p in df["boundary_pth"]]
            self.distance_paths = [os.path.join(self.data_dir, p) for p in df["distance_pth"]]
            self.thickness_paths = [os.path.join(self.data_dir, p) for p in df["thickness_map_pth"]]
            
            
        # 您可以为 "test" split 添加类似的逻辑
        # else: ...

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.split == "train":
            # --- [修改点 2] 加载所有四种数据 ---
            image = read_pkl_file(self.image_paths[idx])
            mask = read_pkl_file(self.mask_paths[idx])
            boundary_map = read_pkl_file(self.boundary_paths[idx])
            distance_map = read_pkl_file(self.distance_paths[idx])
            thickness_map = read_pkl_file(self.thickness_paths[idx])

            # --- 图像预处理 ---
            image = np.clip(image, HU_min, HU_max)
            image = (image - data_mean_parse) / data_std_parse
            
            # 确保所有数据都是 float32
            image = image.astype(np.float32)
            mask = mask.astype(np.float32)
            boundary_map = boundary_map.astype(np.float32)
            distance_map = distance_map.astype(np.float32)
            thickness_map = thickness_map.astype(np.float32)
            
            # --- [修改点 3] 将所有数据打包成一个字典，传递给transform ---
            sample = {
                'image': image, 
                'label': mask,
                'boundary': boundary_map,
                'distance': distance_map,
                'thickness': thickness_map
            }

            if self.transform:
                sample = self.transform(sample)

            sample['case_name'] = self.image_paths[idx].split('/')[-3] # 获取病例ID
            return sample
        
        # 如果有测试集，需要添加相应的逻辑
        return None

def test_dataset_loader():
    """
    测试函数，用于验证dataset_reader_parse是否能成功加载和处理数据。
    """
    print("\n--- 开始测试数据加载器 ---")

    data_dir = '/home/ET/bnwu/MA-SAM/data/parse2022/train/2D_all_5slice'
    
    # 检查CSV文件是否存在
    if not os.path.exists(os.path.join(data_dir, 'training.csv')):
        print(f"错误: 在路径 {data_dir} 下找不到 training.csv。请先运行数据预处理脚本。")
        return

    output_size = [224, 224]
    low_res = [112, 112]
    num_classes = 2 # 假设是背景+血管

    # --- 2. 实例化Dataset和Transform ---
    db_train = dataset_reader_parse(
        base_dir=data_dir, 
        split='train', 
        num_classes=num_classes,
        transform=RandomGenerator(output_size=output_size, low_res=low_res)
    )
    
    print(f"数据集找到 {len(db_train)} 个训练样本。")
    if len(db_train) == 0:
        print("数据集中没有样本，测试无法继续。")
        return

    # --- 3. 取出第一个样本进行测试 ---
    print("\n正在获取第一个样本...")
    sample = db_train[0]
    
    # --- 4. 验证样本内容 ---
    print("成功获取样本！正在检查内容...")
    
    # 检查返回的是否是字典
    assert isinstance(sample, dict), "返回的样本不是一个字典！"
    print(f"样本类型: {type(sample)}")
    
    # 检查所有预期的键是否存在
    expected_keys = ['image', 'label', 'low_res_label', 'boundary', 'distance', 'case_name','thickness']
    for key in expected_keys:
        assert key in sample, f"样本中缺少键: '{key}'"
    print(f"样本包含的键: {list(sample.keys())}")
    
    # 打印每个数据项的详细信息
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            print(f"  - {key}:")
            print(f"    - 类型: {value.dtype}")
            print(f"    - 形状: {value.shape}")
            # 检查形状是否符合预期 (C, H, W)
            if key != 'low_res_label':
                assert value.shape[1] == output_size[0] and value.shape[2] == output_size[1], f"{key}的尺寸不正确！"
            else:
                assert value.shape[1] == low_res[0] and value.shape[2] == low_res[1], f"{key}的尺寸不正确！"
        else:
            print(f"  - {key}: {value}")

    print("\n--- 数据加载器测试通过！---")
    print("所有数据项均已成功加载、处理并返回正确的格式。")


if __name__ == "__main__":
    # 运行测试函数
    test_dataset_loader()
