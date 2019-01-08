"""================================================="""
"""================= LIBRARIES ====================="""
"""================================================="""
import cv2, time, datetime, shutil, pickle as pkl, numpy as np, os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tqdm import tqdm

import Network_library_TF_Keras as network
import helper_functions as hf

import scipy.misc as sm
from scipy import ndimage as ndi

import matplotlib.pyplot as plt

import tensorflow as tf

import keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.utils.np_utils import to_categorical

from functools import partial


"""======================================================================"""
"""================== SET UP TEST DATA GENERATOR ========================"""
"""======================================================================"""
data_path   = "path_to_test_data"
data_names  = os.listdir(data_path)
data_files  = [data_path+"/"+x for x in os.listdir(data_path)]
data_files.sort()
data_names.sort()


### Network weights for a three network ensemble
network_weights1 = "path_to_network_weights_1"
network_weights2 = "path_to_network_weights_2"
network_weights3 = "path_to_network_weights_3"


### Settings for each of the network, which share the same architecture
network_settings= {"mode":"2D","branches":[2,2,2,2],
                  "f_base":32,"k_size":3,"use_BN":False,
                  "dropout":0.2, "use_reg":1e-4, "connection_type":"concat",
                  "n_classes":1, "deconvolve":True,
                  "TF_in":True}

###NOTE: You can also decide to load each network subsequently per test volume to reduce memory usage
segmentation_network1     = network.Load_and_Restore_unnamed(network_weights1,"Unet_2D",network_settings,channels=3)
segmentation_network2     = network.Load_and_Restore_unnamed(network_weights2,"Unet_2D",network_settings,channels=3)
segmentation_network3     = network.Load_and_Restore_unnamed(network_weights3,"Unet_2D",network_settings,channels=3)

output_file = open("test_output_unet_ensemble.csv", "w")
output_file.write("img,rle_mask\n")


### Test dataset mean and standard deviation
mu = np.array([174.4074, 176.1759, 178.0482])
sd = np.array([62.5227, 63.3184, 62.2663])

###
a_col = []
b_col = []


for i,(test_image, test_name) in enumerate(tqdm(zip(data_files, data_names), desc='Computing Test Masks... ')):
    ### Generate Segmentation Predictions per network
    segmentation_image1 = segmentation_network1.run(np.expand_dims(np.pad((cv2.imread(test_image)-mu)/sd,((0,0),(0,2),(0,0)),mode="reflect"),0))
    segmentation_image2 = segmentation_network2.run(np.expand_dims(np.pad((cv2.imread(test_image)-mu)/sd,((0,0),(0,2),(0,0)),mode="reflect"),0))
    segmentation_image3 = segmentation_network3.run(np.expand_dims(np.pad((cv2.imread(test_image)-mu)/sd,((0,0),(0,2),(0,0)),mode="reflect"),0))

    ### Compute ensemble average
    segmentation_image = np.round((segmentation_image1+segmentation_image2+segmentation_image3))

    ### Convert Mask to run length encoding and save
    rle_format = hf.convert_to_run_length_encoding(segmentation_image[0,:,:-2,:].astype(Np.uint8))
    output_file.write(test_name+","+rle_format+"\n")


output_file.close()
