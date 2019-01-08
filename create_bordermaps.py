"""=========================== LIBRARIES ======================"""
import numpy as np, os, time, shutil
import datetime, pickle as pkl, cv2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt

import Network_library_TF_Keras as network
import helper_functions as hf
import scipy.misc as sm, scipy.ndimage as ndi

from tqdm import tqdm, trange
import argparse


"""======================= SET DATA PATHS & GENERATE WEIGHTMAPS ===================="""
parser = argparse.ArgumentParser()
parser.add_argument('--path_to_training_masks', default='', type=str, help='Path to car segmentation ground truth masks.')
parser.add_argument('--save_path', default='', type=str, help='Folder to place computed boundary masks in.')
opt = parser.parse_args()

### Set paths
mask_path   = opt.path_to_training_masks
save_path   = opt.save_path

### Get Ground Truth Segmentation Masks
mask_names = os.listdir(mask_path)
mask_files = [mask_path+"/"+x for x in os.listdir(mask_path)]
mask_files.sort()
mask_names.sort()

### Start Generating boundary weight masks by putting
### High weights on boundaries
for i,(mask_file, mask_name) in enumerate(tqdm(zip(mask_files, mask_names), desc='Generating boundary masks... ')):
    # Read Masks
    mask        = sm.imread(mask_file)[:,:,0]/255.
    struct_elem = np.ones([19,19])

    # Get outer and inner area. Center area is set as additional boundary weight.
    outer_border = ndi.binary_dilation(mask, struct_elem)-mask
    inner_border = mask-ndi.binary_erosion(mask, struct_elem)
    total_border = (outer_border+inner_border>0).astype(np.uint8)

    # Save mask
    np.save(save_path+"/"+mask_name.split(".")[0],total_border)
