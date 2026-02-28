# import shutil
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import nibabel as nib
# from torch.nn import functional as F
# import pickle
# import random
# import pandas as pd
# from tqdm import tqdm
# import cv2
# import SimpleITK as sitk
# from scipy.ndimage.interpolation import zoom
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # base_dir = '/mnt/weka/wekafs/rad-megtron/cchen'
# base_dir = os.path.abspath(os.path.join(current_dir, "../data"))


# def get_all_5slice():

#     save_pth = base_dir + "/AIIB23_Train_T1/2D_all_5slice"
#     img_pth = base_dir + "/AIIB23_Train_T1/img"
#     mask_pth = base_dir + "/AIIB23_Train_T1/gt"

#     img_files = [f for f in os.listdir(img_pth) 
#                 if f.startswith("AIIB23_") and f.endswith(".nii.gz")]
#     img_files.sort()
#     cnt = 0
#     for img_file in img_files:
#         case_id = img_file.split('_')[1].split('.')[0]
#         case_save_dir = os.path.join(save_pth,case_id)
#         os.makedirs(os.path.join(case_save_dir,'images'),
#                     exist_ok= True)
#         os.makedirs(os.path.join(case_save_dir,'masks'),
#                     exist_ok= True)
#         img_arr = nib.load(os.path.join(img_pth,img_file)).get_fdata()
#         # mask_file = img_file.replace(".nii.gz","_seg.nii.gz")
#         # mask_file = img_file.replace("AIIB23_", "AIIB23_").replace(".nii.gz", "_seg.nii.gz")  
#         mask_file = img_file
#         mask_arr = nib.load(os.path.join(mask_pth,mask_file)).get_fdata()
        
#     # data_fd_list = os.listdir(data_pth)
#     # data_fd_list = [data_fd for data_fd in data_fd_list if data_fd.startswith("img") and data_fd.endswith(".nii.gz")]
#     # data_fd_list.sort()
#     # cnt = 0
#     # for data_fd_indx, data_fd in enumerate(data_fd_list):
#     #     case_id = data_fd[3:7]
#     #     if not os.path.exists(save_pth+'/'+case_id):
#     #         os.makedirs(save_pth+'/'+case_id)
#     #         os.mkdir(save_pth+'/'+case_id+'/images')
#     #         os.mkdir(save_pth+'/'+case_id+'/masks')
    
#     #     img_obj = nib.load(data_pth + '/' + data_fd)
#     #     img_arr = img_obj.get_fdata()

#     #     #load mask
#     #     mask_obj = nib.load(data_pth.replace("/img", "/label") + '/' + data_fd.replace("img", "label"))
#     #     mask_arr = mask_obj.get_fdata()
#         for data in [img_arr,mask_arr]:
#             data = np.concatenate([
#                 data[:, :, 0:1], data[:, :, 0:1],  # 头部填充
#                 data,  # 原始数据
#                 data[:, :, -1:], data[:, :, -1:]   # 尾部填充
#             ], axis=-1)
            
#         # img_arr = np.concatenate((img_arr[:, :, 0:1], img_arr[:, :, 0:1], img_arr, img_arr[:, :, -1:], img_arr[:, :, -1:]), axis=-1)
#         # mask_arr = np.concatenate((mask_arr[:, :, 0:1], mask_arr[:, :, 0:1], mask_arr, mask_arr[:, :, -1:], mask_arr[:, :, -1:]), axis=-1)

#         for slice_indx in range(2, img_arr.shape[2]-2):
            
#             # img_slice = img_arr[:,:,slice_idx - 2:slice_idx + 3]
#             # img_slice = np.flip(np.rot90(img_file,k=1,axes=(0,1)),axis=1)
            
            
#             slice_arr = img_arr[:,:,slice_indx-2: slice_indx+3]
#             slice_arr = np.flip(np.rot90(slice_arr, k=1, axes=(0, 1)), axis=1)

#             mask_arr_2D = mask_arr[:,:,slice_indx-2: slice_indx+3]
#             mask_arr_2D = np.flip(np.rot90(mask_arr_2D, k=1, axes=(0, 1)), axis=1)

#             with open(save_pth+'/'+case_id+'/images'+'/2Dimage_'+'{:04d}'.format(slice_indx-2)+'.pkl', 'wb') as file:
#                 pickle.dump(slice_arr, file)

#             with open(save_pth+'/'+case_id+'/masks'+'/2Dmask_'+'{:04d}'.format(slice_indx-2)+'.pkl', 'wb') as file:
#                 pickle.dump(mask_arr_2D, file)

#         cnt += 1

# def get_csv():
#     save_pth = base_dir + "/AIIB23_Train_T1/2D_all_5slice"
    
#     training_csv = save_pth+'/training.csv'
#     test_csv = save_pth+'/test.csv'
#     img_pth = base_dir + "/AIIB23_Train_T1/img"
#     mask_pth = base_dir + "/AIIB23_Train_T1/gt"

#     # 获取所有图像文件
#     img_files = [f for f in os.listdir(img_pth) 
#                 if f.startswith("AIIB23_") and f.endswith(".nii.gz")]
#     img_files.sort()

#     # 获取所有掩码文件
#     mask_files = [f for f in os.listdir(mask_pth) 
#                 if f.startswith("AIIB23_") and f.endswith(".nii.gz")]
#     mask_files.sort()
    
#     assert img_files == mask_files, "图像和掩码文件名不一致"

#     # data_fd_list = os.listdir(save_pth)
#     # data_fd_list = [data_fd for data_fd in data_fd_list if
#     #                 data_fd.startswith('00') and '.' not in data_fd]
    
#     random.shuffle(img_files)
#     random.shuffle(img_files)
#     random.shuffle(img_files)
#     random.shuffle(img_files)
#     random.shuffle(img_files)
#     test_case_ids = ['39', '105', '93', '107', '83', '91', 
#                  '68', '98', '140', '87', '36', '60', 
#                  '159', '63', '137', '92', '157', '113', 
#                  '138', '122', '71', '117', '79', '34']
    
#     # test_fd_list = ['0035', '0036', '0037', '0038', '0039', '0040']

#     # training_fd_list = list(set(data_fd_list)-set(test_fd_list))

#     # path_list_all = []
#     # for data_fd in training_fd_list:
#     #     slice_list = os.listdir(save_pth+'/'+data_fd+'/images')
#     #     slice_pth_list = [data_fd+'/images/'+slice for slice in slice_list]
#     #     path_list_all = path_list_all + slice_pth_list

#     # random.shuffle(path_list_all)
#     # random.shuffle(path_list_all)
#     # random.shuffle(path_list_all)
#     # random.shuffle(path_list_all)
#     # random.shuffle(path_list_all)
#     # df = pd.DataFrame(path_list_all, columns=['image_pth'])
#     # df['mask_pth'] = path_list_all
#     # df['mask_pth'] = df['mask_pth'].apply(lambda x: x.replace('/images/2Dimage_', '/masks/2Dmask_'))

#     # df.to_csv(training_csv, index=False)

#     # path_list_all = []
#     # for data_fd in test_fd_list:
#     #     slice_list = os.listdir(save_pth+'/'+data_fd+'/images')
#     #     slice_list.sort()
#     #     slice_pth_list = [data_fd+'/images/'+slice for slice in slice_list]
#     #     path_list_all = path_list_all + slice_pth_list

#     # df = pd.DataFrame(path_list_all, columns=['image_pth'])
#     # df['mask_pth'] = path_list_all
#     # df['mask_pth'] = df['mask_pth'].apply(lambda x: x.replace('/images/2Dimage_', '/masks/2Dmask_'))

#     # df.to_csv(test_csv, index=False)

#     # 将文件分为训练集和测试集
#     training_files = []
#     test_files = []
#     for file in img_files:
#         case_id = file.split('_')[1].split('.')[0]
#         if case_id in test_case_ids:
#             test_files.append(file)
#         else:
#             training_files.append(file)

#     # 生成训练集 CSV
#     path_list_all = []
#     for file in training_files:
#         case_id = file.split('_')[1].split('.')[0]
#         case_dir = os.path.join(save_pth, case_id)
#         img_slices = os.listdir(os.path.join(case_dir, 'images'))
#         img_slices.sort()
#         img_paths = [os.path.join(case_id, 'images', slice) for slice in img_slices]
#         path_list_all.extend(img_paths)

#     random.shuffle(path_list_all)
#     df = pd.DataFrame(path_list_all, columns=['image_pth'])
#     df['mask_pth'] = df['image_pth'].apply(lambda x: x.replace('/images/2Dimage_', '/masks/2Dmask_'))
#     df.to_csv(training_csv, index=False)

#     # 生成测试集 CSV
#     path_list_all = []
#     for file in test_files:
#         case_id = file.split('_')[1].split('.')[0]
#         case_dir = os.path.join(save_pth, case_id)
#         img_slices = os.listdir(os.path.join(case_dir, 'images'))
#         img_slices.sort()
#         img_paths = [os.path.join(case_id, 'images', slice) for slice in img_slices]
#         path_list_all.extend(img_paths)

#     df = pd.DataFrame(path_list_all, columns=['image_pth'])
#     df['mask_pth'] = df['image_pth'].apply(lambda x: x.replace('/images/2Dimage_', '/masks/2Dmask_'))
#     df.to_csv(test_csv, index=False)

# def get_data_statistics():
#     HU_min = -500
#     HU_max = 500

#     data_pth = base_dir + "/synapseCT/Training/img"
    
#     img_pth = base_dir + "/AIIB23_Train_T1/img"
#     mask_pth = base_dir + "/AIIB23_Train_T1/gt"
    
#     # data_fd_list = os.listdir(img_pth)
#     # data_fd_list = [data_fd for data_fd in data_fd_list if data_fd.startswith("img")]
#     # data_fd_list.sort()
#     img_files = [f for f in os.listdir(img_pth) 
#                 if f.startswith("AIIB23_") and f.endswith(".nii.gz")]
#     img_files.sort()

#     mean_val_list = []
#     std_val_list = []
#     psum_list = []
#     psum_sq_list = []
#     count_list = []
#     for img_file in tqdm(img_files):
        
#         img_obj = nib.load(img_pth + '/' + img_file)
#         img_arr = img_obj.get_fdata()
        
#         mask_file = img_file
#         mask_obj = nib.load(os.path.join(mask_pth,mask_file))
#         mask_arr = mask_obj.get_fdata()
        
#         # preprocessing
#         img_arr[img_arr<=HU_min] = HU_min
#         img_arr[img_arr>=HU_max] = HU_max
#         img_arr = (img_arr-HU_min)/(HU_max-HU_min)*255.0

#         psum = np.sum(img_arr)
#         psum_sq = np.sum(img_arr ** 2)

#         mean_val = np.mean(img_arr)
#         std_val = np.std(img_arr)

#         mean_val_list.append(mean_val)
#         std_val_list.append(std_val)

#         psum_list.append(psum)
#         psum_sq_list.append(psum_sq)

#         count_list.append(img_arr.shape[0]*img_arr.shape[1]*img_arr.shape[2])

#     psum_tot = 0.0
#     psum_sq_tot = 0.0
#     count_tot = 0.0
#     for i in range(len(psum_list)):
#         psum_tot += psum_list[i]
#         psum_sq_tot += psum_sq_list[i]
#         count_tot += count_list[i]

#     total_mean = psum_tot / count_tot # 50.21997497685108
#     total_var  = (psum_sq_tot / count_tot) - (total_mean ** 2)
#     total_std  = np.sqrt(total_var) # 68.47153712416372

#     print(total_mean, total_std)
#     # AIIB23 58.08913621726867 71.16849465558357

# if __name__=="__main__":
#     # get_all_5slice()
#     get_csv()
#     # get_data_statistics()


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
    get_all_5slice()
    # get_csv()
    # get_data_statistics()