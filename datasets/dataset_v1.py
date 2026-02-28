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
import re # Import regular expressions
import SimpleITK as sitk # 导入SimpleITK库用于读取.nii.gz文件

#================================================================================
# Constants and Helper Functions
#================================================================================

# --- Constants for HU windowing and normalization ---
HU_min, HU_max = -500, 500

# Normalization stats for AIIB2023
aiib_mean = 58.08913621726867
aiib_std = 71.16849465558357

# Normalization stats for PARSE2022
parse_mean = 53.4296
parse_std = 68.6344

def read_image(path):
    """
    MODIFIED: Reads a .nii.gz file using SimpleITK and returns a numpy array.
    The array is permuted from SimpleITK's (d, h, w) to the project's (h, w, d).
    修改: 使用SimpleITK读取.nii.gz文件，并返回numpy数组。
    数组维度会从 (d, h, w) 转换为 (h, w, d) 以匹配项目约定。
    """
    sitk_img = sitk.ReadImage(path)
    # Get numpy array from image and permute axes
    np_img = sitk.GetArrayFromImage(sitk_img)
    np_img = np_img.transpose(1, 2, 0)
    return np_img

#================================================================================
# Data Augmentation Functions (Unaltered)
#================================================================================

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k, axes=(0, 1))
    label = np.rot90(label, k, axes=(0, 1))
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label

def random_rotate(image, label):
    angle = np.random.randint(-15, 15)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def convert_to_PIL(img: np.array) -> PIL.Image:
    '''
    img should be normalized between 0 and 1
    '''
    img = np.clip(img, 0, 1)
    return PIL.Image.fromarray((img * 255).astype(np.uint8))

def convert_to_PIL_label(label):
    return PIL.Image.fromarray(label.astype(np.uint8))

def convert_to_np(img: PIL.Image) -> np.array:
    return np.array(img).astype(np.float32) / 255

def convert_to_np_label(label):
    return np.array(label).astype(np.float32)

def random_erasing(
    imgs,
    label,
    scale_z=(0.02, 0.33),
    scale=(0.02, 0.05),
    ratio=(0.3, 3.3),
    apply_all: int = 0,
    rng: np.random.Generator = np.random.default_rng(0),
):

    # determine the box
    imgshape = imgs.shape
    
    # nx and ny
    while True:
        se = rng.uniform(scale[0], scale[1]) * imgshape[0] * imgshape[1]
        re = rng.uniform(ratio[0], ratio[1])
        nx = int(np.sqrt(se * re))
        ny = int(np.sqrt(se / re))
        if nx < imgshape[1] and ny < imgshape[0]:
            break

    # determine the position of the box
    sy = rng.integers(0, imgshape[0] - ny + 1)
    sx = rng.integers(0, imgshape[1] - nx + 1)

    filling = rng.uniform(0, 1, size=[ny, nx])
    filling = filling[:,:,np.newaxis]
    filling = np.repeat(filling, imgshape[-1], axis=-1)

    # erase
    imgs[sy:sy + ny, sx:sx + nx, :] = filling
    label[sy:sy + ny, sx:sx + nx, :] = 0.

    return imgs, label

def posterize(img, label, v):
    v = int(v)
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = PIL.ImageOps.posterize(img_curr, bits=v)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr
    return img, label

def contrast(img, label, v):
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = PIL.ImageEnhance.Contrast(img_curr).enhance(v)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr
    return img, label

def brightness(img, label, v):
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = PIL.ImageEnhance.Brightness(img_curr).enhance(v)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr
    return img, label

def sharpness(img, label, v):
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = PIL.ImageEnhance.Sharpness(img_curr).enhance(v)
        img_curr = convert_to_np(img_curr)
        img[:,:,slice_indx] = img_curr
    return img, label

def identity(img, label, v):
    return img, label

def adjust_light(image, label):
    image = image*255.0
    gamma = random.random() * 3 + 0.5
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    for slice_indx in range(image.shape[2]):
        img_curr = image[:,:,slice_indx]
        img_curr = cv2.LUT(np.array(img_curr).astype(np.uint8), table).astype(np.uint8)
        image[:,:,slice_indx] = img_curr
    image = image/255.0
    return image, label

def shear_x(img, label, v):
    shear_mat = [1, v, -v * img.shape[1] / 2, 0, 1, 0]
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, shear_mat, resample=PIL.Image.BILINEAR)
        img[:,:,slice_indx] = convert_to_np(img_curr)
    for slice_indx in range(label.shape[2]):
        label_curr = label[:,:,slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, shear_mat, resample=PIL.Image.NEAREST)
        label[:,:,slice_indx] = convert_to_np_label(label_curr)
    return img, label

def shear_y(img, label, v):
    shear_mat = [1, 0, 0, v, 1, -v * img.shape[0] / 2]
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, shear_mat, resample=PIL.Image.BILINEAR)
        img[:,:,slice_indx] = convert_to_np(img_curr)
    for slice_indx in range(label.shape[2]):
        label_curr = label[:,:,slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, shear_mat, resample=PIL.Image.NEAREST)
        label[:,:,slice_indx] = convert_to_np_label(label_curr)
    return img, label

def translate_x(img, label, v):
    translate_mat = [1, 0, v * img.shape[1], 0, 1, 0]
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, translate_mat, resample=PIL.Image.BILINEAR)
        img[:,:,slice_indx] = convert_to_np(img_curr)
    for slice_indx in range(label.shape[2]):
        label_curr = label[:,:,slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, translate_mat, resample=PIL.Image.NEAREST)
        label[:,:,slice_indx] = convert_to_np_label(label_curr)
    return img, label

def translate_y(img, label, v):
    translate_mat = [1, 0, 0, 0, 1, v * img.shape[0]]
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, translate_mat, resample=PIL.Image.BILINEAR)
        img[:,:,slice_indx] = convert_to_np(img_curr)
    for slice_indx in range(label.shape[2]):
        label_curr = label[:,:,slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, translate_mat, resample=PIL.Image.NEAREST)
        label[:,:,slice_indx] = convert_to_np_label(label_curr)
    return img, label

def scale(img, label, v):
    for slice_indx in range(img.shape[2]):
        img_curr = img[:,:,slice_indx]
        img_curr = convert_to_PIL(img_curr)
        img_curr = img_curr.transform(img_curr.size, PIL.Image.AFFINE, [v, 0, 0, 0, v, 0], resample=PIL.Image.BILINEAR)
        img[:,:,slice_indx] = convert_to_np(img_curr)
    for slice_indx in range(label.shape[2]):
        label_curr = label[:,:,slice_indx]
        label_curr = convert_to_PIL_label(label_curr)
        label_curr = label_curr.transform(label_curr.size, PIL.Image.AFFINE, [v, 0, 0, 0, v, 0], resample=PIL.Image.NEAREST)
        label[:,:,slice_indx] = convert_to_np_label(label_curr)
    return img, label

class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res
        seed = 42
        self.rng = np.random.default_rng(seed)
        self.p = 0.5
        self.n = 2
        self.scale = (0.8, 1.2, 2)
        self.translate = (-0.2, 0.2, 2)
        self.shear = (-0.3, 0.3, 2)
        self.posterize = (4, 8.99, 2)
        self.contrast = (0.7, 1.3, 2)
        self.brightness = (0.7, 1.3, 2)
        self.sharpness = (0.1, 1.9, 2)
        self.create_ops()

    def create_ops(self):
        ops = [
            (shear_x, self.shear), (shear_y, self.shear), (scale, self.scale),
            (translate_x, self.translate), (translate_y, self.translate),
            (posterize, self.posterize), (contrast, self.contrast),
            (brightness, self.brightness), (sharpness, self.sharpness),
            (identity, (0, 1, 1)),
        ]
        self.ops = [op for op in ops if op[1][2] != 0]

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if random.random() > 0.5: image, label = random_rot_flip(image, label)
        if random.random() > 0.5: image, label = random_rotate(image, label)
        if random.random() > 0.5: image, label = adjust_light(image, label)
        if random.random() > 0.5: image, label = random_erasing(imgs=image, label=label, rng=self.rng)
        
        inds = self.rng.choice(len(self.ops), size=self.n, replace=False)
        for i in inds:
            op = self.ops[i]
            aug_func, aug_params = op[0], op[1]
            v = self.rng.uniform(aug_params[0], aug_params[1])
            image, label = aug_func(image, label, v)

        x, y, z = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y, 1.0), order=0)
        
        label_h, label_w, label_d = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w, 1.0), order=0)
        
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        label = torch.from_numpy(label.astype(np.float32)).permute(2, 0, 1)
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32)).permute(2, 0, 1)
        
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample


class RandomCrop:
    """
    对图像和标签进行随机裁剪。
    """
    def __init__(self, spatial_size):
        self.spatial_size = spatial_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        
        img_shape = label.shape
        
        # 确保裁剪尺寸不大于图像尺寸
        roi_size = [min(s, d) for s, d in zip(self.spatial_size, img_shape)]

        # 计算随机裁剪的起始点
        starts = [
            random.randint(0, d - s) if d > s else 0
            for s, d in zip(roi_size, img_shape)
        ]

        slicing = tuple(slice(s, s + rs) for s, rs in zip(starts, roi_size))

        image = image[slicing]
        label = label[slicing]


        # slicing_label = tuple(slice(s, s + rs) for s, rs in zip(starts, roi_size))
        # slicing_img = (slice(None),) + slicing_label

        sample['image'] = image
        sample['label'] = label
        
        # 如果由于边界效应导致裁剪尺寸小于目标尺寸，则进行填充
        current_shape = sample['label'].shape
        pad_needed = [max(0, s - d) for s, d in zip(self.spatial_size, current_shape)]
        if any(p > 0 for p in pad_needed):
            pad_width_img = [(0, 0)]
            pad_width_label = []
            for p in pad_needed:
                pad_before = p // 2
                pad_after = p - pad_before
                pad_width_img.append((pad_before, pad_after))
                pad_width_label.append((pad_before, pad_after))
            
            sample['image'] = np.pad(sample['image'], pad_width_img, mode='constant', constant_values=0)
            sample['label'] = np.pad(sample['label'], pad_width_label, mode='constant', constant_values=0)

        return sample



#================================================================================
# AIIB2023 Dataset Class
#================================================================================
class AIIB2023Dataset(Dataset):
    def __init__(self, base_dir, split, num_classes, transform=None):
        """
        Args:
            base_dir (str): Path to the AIIB2023 dataset directory (e.g., './AIIB23_Train_T1').
            split (str): 'train' or 'val'.
            num_classes (int): Number of segmentation classes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.split = split
        self.base_dir = base_dir
        self.num_classes = num_classes
        self.sample_list = []

        # --- Define your validation set here ---
        # Extract the numeric part of the filename for the validation IDs
        val_ids = ['30', '31', '32', '33', '34', '36', '37'] # Example IDs, please adjust

        img_dir = os.path.join(base_dir, 'img')
        all_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.nii.gz')])

        train_files = []
        val_files = []

        for f in all_files:
            # Extract the ID from filename like 'AIIB23_30.nii.gz' -> '30'
            match = re.search(r'_(\d+)\.', f)
            if match:
                file_id = match.group(1)
                if file_id in val_ids:
                    val_files.append(f)
                else:
                    train_files.append(f)

        if self.split == 'train':
            self.sample_list = train_files
            print(f"Found {len(self.sample_list)} training cases in AIIB2023.")
        elif self.split == 'val':
            self.sample_list = val_files
            print(f"Found {len(self.sample_list)} validation cases in AIIB2023.")
        else:
            raise ValueError(f"Split '{self.split}' is not valid. Choose 'train' or 'val'.")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx]
        img_path = os.path.join(self.base_dir, 'img', case_name)
        gt_path = os.path.join(self.base_dir, 'gt', case_name)

        # --- Image Loading and Processing ---
        image = read_image(img_path) # Now reads .nii.gz
        image = np.clip(image, HU_min, HU_max)
        image = (image - HU_min) / (HU_max - HU_min) * 255.0
        image = np.float32(image)
        image = (image - aiib_mean) / aiib_std
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # --- Label Loading ---
        label = read_image(gt_path) # Now reads .nii.gz
        label = np.float32(label)
        
        if self.num_classes == 12:
            label[label == 13] = 12

        sample = {'image': np.float32(image), 'label': np.float32(label)}
        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = case_name
        return sample

#================================================================================
# PARSE2022 Dataset Class
#================================================================================
class PARSE2022Dataset(Dataset):
    def __init__(self, base_dir, split, num_classes, transform=None):
        """
        Args:
            base_dir (str): Path to the PARSE2022 dataset directory (e.g., './train').
            split (str): 'train' or 'val'.
            num_classes (int): Number of segmentation classes.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.split = split
        self.base_dir = base_dir
        self.num_classes = num_classes
        self.sample_list = []

        # --- Validation set as specified in your image ---
        val_ids = ['PA000005', 'PA000016', 'PA000024', 'PA000026', 'PA000027', 'PA000036']
        
        all_case_folders = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
        
        train_cases = []
        val_cases = []

        for case_id in all_case_folders:
            if case_id in val_ids:
                val_cases.append(case_id)
            else:
                train_cases.append(case_id)

        if self.split == 'train':
            self.sample_list = train_cases
            print(f"Found {len(self.sample_list)} training cases in PARSE2022.")
        elif self.split == 'val':
            self.sample_list = val_cases
            print(f"Found {len(self.sample_list)} validation cases in PARSE2022.")
        else:
            raise ValueError(f"Split '{self.split}' is not valid. Choose 'train' or 'val'.")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_id = self.sample_list[idx]
        case_dir = os.path.join(self.base_dir, case_id)
        
        file_name = f"{case_id}.nii.gz" 
        img_path = os.path.join(case_dir, 'image', file_name)
        label_path = os.path.join(case_dir, 'label', file_name)

        # --- Image Loading and Processing ---
        image = read_image(img_path) # Now reads .nii.gz
        image = np.clip(image, HU_min, HU_max)
        image = (image - HU_min) / (HU_max - HU_min) * 255.0
        image = np.float32(image)
        image = (image - parse_mean) / parse_std
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # --- Label Loading ---
        label = read_image(label_path) # Now reads .nii.gz
        label = np.float32(label)

        if self.num_classes == 12:
            label[label == 13] = 12

        sample = {'image': np.float32(image), 'label': np.float32(label)}
        if self.transform:
            sample = self.transform(sample)
        
        sample['case_name'] = case_id
        return sample




train_dataset1 = PARSE2022Dataset(
    base_dir='/home/ET/bnwu/MA-SAM/data/parse2022/train',
    split='train',
    num_classes=1,
    transform= RandomCrop(spatial_size=(128, 128, 128))
)





# Create the dataloader
train_loader1 = torch.utils.data.DataLoader(
    train_dataset1,
    batch_size=4,
    shuffle=True,
    num_workers=8
)


train_dataset1 = AIIB2023Dataset(
    base_dir='/home/ET/bnwu/MA-SAM/data/AIIB23_Train_T1',
    split='train',
    num_classes=1,
    transform=RandomCrop(spatial_size=(128, 128, 128))
)

# Create the dataloader
train_loader1 = torch.utils.data.DataLoader(
    train_dataset1,
    batch_size=4,
    shuffle=True,
    num_workers=8
)