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
current_dir = os.path.dirname(os.path.abspath(__file__))
# base_dir = '/mnt/weka/wekafs/rad-megtron/cchen'
base_dir = os.path.abspath(os.path.join(current_dir, "../data"))


def get_all_5slice():
    """
    Process 3D MRI volumes into 2D slices with 5-slice context (2 before, current, 2 after)
    """
    save_pth = os.path.join(base_dir, "parse2022/train/2D_all_5slice")
    data_pth = os.path.join(base_dir, "parse2022/train")

    # Get all patient directories
    patient_dirs = [d for d in os.listdir(data_pth) if d.startswith("PA") and os.path.isdir(os.path.join(data_pth, d))]
    patient_dirs.sort()

    cnt = 0
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        case_id = patient_dir  # e.g., PA000005
        
        # Create output directories if they don't exist
        if not os.path.exists(os.path.join(save_pth, case_id)):
            os.makedirs(os.path.join(save_pth, case_id))
            os.mkdir(os.path.join(save_pth, case_id, 'images'))
            os.mkdir(os.path.join(save_pth, case_id, 'masks'))
        
        # Load image
        img_file_path = os.path.join(data_pth, patient_dir, "image", f"{patient_dir}.nii.gz")
        img_obj = nib.load(img_file_path)
        img_arr = img_obj.get_fdata()

        # Load mask
        mask_file_path = os.path.join(data_pth, patient_dir, "label", f"{patient_dir}.nii.gz")
        mask_obj = nib.load(mask_file_path)
        mask_arr = mask_obj.get_fdata()

        # Pad first and last slices to create 5-slice context
        img_arr = np.concatenate((img_arr[:, :, 0:1], img_arr[:, :, 0:1], img_arr, img_arr[:, :, -1:], img_arr[:, :, -1:]), axis=-1)
        mask_arr = np.concatenate((mask_arr[:, :, 0:1], mask_arr[:, :, 0:1], mask_arr, mask_arr[:, :, -1:], mask_arr[:, :, -1:]), axis=-1)

        # Process each slice
        for slice_indx in range(2, img_arr.shape[2]-2):
            # Extract 5 slices centered at current slice
            slice_arr = img_arr[:,:,slice_indx-2: slice_indx+3]
            slice_arr = np.flip(np.rot90(slice_arr, k=1, axes=(0, 1)), axis=1)

            # Process mask the same way
            mask_arr_2D = mask_arr[:,:,slice_indx-2: slice_indx+3]
            mask_arr_2D = np.flip(np.rot90(mask_arr_2D, k=1, axes=(0, 1)), axis=1)

            # Save to pickle files
            with open(os.path.join(save_pth, case_id, 'images', f'2Dimage_{slice_indx-2:04d}.pkl'), 'wb') as file:
                pickle.dump(slice_arr, file)

            with open(os.path.join(save_pth, case_id, 'masks', f'2Dmask_{slice_indx-2:04d}.pkl'), 'wb') as file:
                pickle.dump(mask_arr_2D, file)

        cnt += 1


def get_csv():
    """
    Generate training and test CSV files with image and mask paths
    """
    save_pth = os.path.join(base_dir, "parse2022/train/2D_all_5slice")
    
    training_csv = os.path.join(save_pth, 'training.csv')
    test_csv = os.path.join(save_pth, 'test.csv')

    data_fd_list = [d for d in os.listdir(save_pth) if d.startswith('PA') and os.path.isdir(os.path.join(save_pth, d))]
    
    # Shuffle multiple times for better randomization
    random.shuffle(data_fd_list)
    random.shuffle(data_fd_list)
    random.shuffle(data_fd_list)
    random.shuffle(data_fd_list)
    random.shuffle(data_fd_list)

    # Specific test set patients
    test_fd_list = ['PA000005', 'PA000016', 'PA000024', 'PA000026', 'PA000027', 'PA000036']

    # Training set is everything else
    training_fd_list = list(set(data_fd_list) - set(test_fd_list))

    # Create training CSV
    path_list_all = []
    for data_fd in training_fd_list:
        slice_list = os.listdir(os.path.join(save_pth, data_fd, 'images'))
        slice_pth_list = [os.path.join(data_fd, 'images', slice_name) for slice_name in slice_list]
        path_list_all.extend(slice_pth_list)

    # Shuffle multiple times
    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    random.shuffle(path_list_all)
    
    # Create DataFrame
    df = pd.DataFrame(path_list_all, columns=['image_pth'])
    df['mask_pth'] = df['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'masks/2Dmask_'))

    df.to_csv(training_csv, index=False)

    # Create test CSV
    path_list_all = []
    for data_fd in test_fd_list:
        slice_list = os.listdir(os.path.join(save_pth, data_fd, 'images'))
        slice_list.sort()
        slice_pth_list = [os.path.join(data_fd, 'images', slice_name) for slice_name in slice_list]
        path_list_all.extend(slice_pth_list)

    df = pd.DataFrame(path_list_all, columns=['image_pth'])
    df['mask_pth'] = df['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'masks/2Dmask_'))

    df.to_csv(test_csv, index=False)


def get_data_statistics():
    """
    Calculate dataset statistics (mean and std) for normalization
    """
    HU_min = -500
    HU_max = 500

    data_pth = os.path.join(base_dir, "parse2022/train")
    patient_dirs = [d for d in os.listdir(data_pth) if d.startswith("PA") and os.path.isdir(os.path.join(data_pth, d))]
    patient_dirs.sort()

    mean_val_list = []
    std_val_list = []
    psum_list = []
    psum_sq_list = []
    count_list = []
    
    for patient_dir in tqdm(patient_dirs, desc="Calculating statistics"):
        # Load image
        img_file_path = os.path.join(data_pth, patient_dir, "image", f"{patient_dir}.nii.gz")
        img_obj = nib.load(img_file_path)
        img_arr = img_obj.get_fdata()

        # Preprocessing: clamp values and normalize
        img_arr = np.clip(img_arr, HU_min, HU_max)
        img_arr = (img_arr - HU_min) / (HU_max - HU_min) * 255.0

        # Calculate statistics
        psum = np.sum(img_arr)
        psum_sq = np.sum(img_arr ** 2)
        count = img_arr.size

        mean_val = np.mean(img_arr)
        std_val = np.std(img_arr)

        mean_val_list.append(mean_val)
        std_val_list.append(std_val)
        psum_list.append(psum)
        psum_sq_list.append(psum_sq)
        count_list.append(count)

    # Calculate global statistics
    psum_tot = sum(psum_list)
    psum_sq_tot = sum(psum_sq_list)
    count_tot = sum(count_list)

    total_mean = psum_tot / count_tot
    total_var = (psum_sq_tot / count_tot) - (total_mean ** 2)
    total_std = np.sqrt(total_var)

    print(f"Dataset mean: {total_mean:.4f}, std: {total_std:.4f}")
    
    return total_mean, total_std


if __name__ == "__main__":
    # get_all_5slice()
    get_csv()
    get_data_statistics()