"""================ Libraries ======================"""
import numpy as np, os, scipy.misc as sm, time, shutil, datetime, pickle as pkl, cv2
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt
from scipy import ndimage as ndi

import network_library as network
import helper_functions as hf

import tensorflow as tf, keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.utils.np_utils import to_categorical

from functools import partial
from tqdm import tqdm, trange
import argparse


"""============= TRAINING SETUP ============="""
parser = argparse.ArgumentParser()
parser.add_argument('--path_to_input_images', default='', type=str, help='Path to folder containing training images.')
parser.add_argument('--path_to_ground_truth_masks', default='', type=str, help='Path to folder containing ground truth masks.')
parser.add_argument('--path_to_boundary_masks', default='', type=str, help='Path to precomputed boundary masks.')
opt = parser.parse_args()

# Weight mask parameters
border_mask_path = opt.path_to_boundary_masks
weighting = 1.1

# List of image and mask paths
data_path   = opt.path_to_input_images
gt_path     = opt.path_to_boundary_masks
data_files  = np.array([data_path+"/"+x for x in os.listdir(data_path)])
gt_files    = np.array([gt_path  +"/"+x for x in os.listdir(gt_path)])
data_files.sort()
gt_files.sort()

# Shuffle paths
seed = 1
rng = np.random.RandomState(seed)
rand_idxs = rng.choice(np.arange(len(data_files)), len(data_files), replace=False)
data_files = data_files[rand_idxs]
gt_files   = gt_files[rand_idxs]


# Training Setup - Division into Training and Validation Data
n_epochs       = 45
run_validation = True
train_val_div  = 0.8 #Training-Validation-Division
train_data  = data_files[:int(len(data_files)*train_val_div)]
train_gt    = gt_files[:int(len(data_files)*train_val_div)]
val_data    = data_files[int(len(data_files)*train_val_div):]
val_gt      = gt_files[int(len(data_files)*train_val_div):]

# Set boundary mask paths
border_files    = np.array([border_mask_path  +"/"+x for x in os.listdir(border_mask_path)])
border_files    = border_files[rand_idxs]
train_bm    = border_files[:int(len(data_files)*train_val_div)]
val_bm      = border_files[int(len(data_files)*train_val_div):]



"""================= NETWORK SETUP ================"""
# Reset any existing graph
tf.reset_default_graph()

# Set session
sess = tf.Session()
K.set_session(sess)

# Set all necessary placeholders
lr = tf.placeholder(tf.float32,shape=[])
img         = tf.placeholder(tf.float32, shape=(None, None, None, 3))
true_seg    = tf.placeholder(tf.float32, shape=(None, None, None, 1))
weights     = tf.placeholder(tf.float32, shape=(None, None, None, 1))

# Network Setup Parameters for UNet or Tiramisu to use for Segmentation
unet_params = {"mode":"2D","branches":[2,2,3,3],
               "f_base":32,"k_size":3,"use_BN":False,
               "dropout":0.2, "use_reg":1e-4, "connection_type":"concat",
               "n_classes":1, "deconvolve":True,
               "TF_in":True,"tf_placeholder":img}

tiramisu_params = {"mode":"2D", "upmode":"deconv", "verbose":2, "denseblocks_down":[4,4,6,6], "denseblocks_up":None, "blockbottom":[6], "f_base":32, "k_size":3, "k":12,
                              "use_reg":1e-4, "dropout":0.2, "act":"sigmoid", "use_BN":False, "n_classes":1, "tf_placeholder":img}


### Set up segmentation network
pred_seg,mpool_len = network.get_standard_unet(**unet_params)


### Count trainable parameters - only works when reset was performed.
total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parametes = 1
    for dim in shape:
        variable_parametes *= dim.value
    total_parameters += variable_parametes
print "Total number of parameters: ",total_parameters
print "----"



### Loss and Optimizer Setup
loss = network.weighted_focal_binary_crossentropy(K.flatten(true_seg), K.flatten(pred_seg), K.flatten(weights), weight_type="pixelwise", gamma=2)
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)


### Evaluation Metrics
correct_prediction  = tf.equal(tf.round(K.flatten(pred_seg)), K.flatten(true_seg))
accuracy            = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
dice_coeff          = network.dice_coeff(K.flatten(true_seg), K.flatten(pred_seg))
val_accuracy        = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
val_dice_coeff      = network.dice_coeff(true_seg, pred_seg)


### Checkpoint name for saving
now = datetime.datetime.now()
dy  = now.strftime("%d%m%Y")
dh  = now.strftime("%H%M%S")
Net_name = "Unet"+"_rundate"+dy+"_starttime"+dh

### Logging Setup
check = True
run_num = 1
# Set not existing saving folder
while check:
    sublog  ="run"+str(run_num)
    log_name="log_unet"
    log_path=os.getcwd()+"/log-files/"+log_name+"/"+sublog
    if os.path.exists(log_path):
        run_num+=1
    else:
        check=False


### Logs - Set summary writer
a=tf.summary.scalar("loss", loss)
b=tf.summary.scalar("accuracy", accuracy)
c=tf.summary.scalar("dice",dice_coeff)
summary_train   = tf.summary.merge([a,b,c])
if run_validation:
    e=tf.summary.scalar("val_acc", val_accuracy)
    f=tf.summary.scalar("val_dice",val_dice_coeff)
    summary_val = tf.summary.merge([e,f])
writer = tf.summary.FileWriter(log_path, graph = tf.get_default_graph())



### Set TF saver, Initialize variables
saver       = tf.train.Saver()
tf.add_to_collection("Net_model",pred_seg)
var_init_global = tf.global_variables_initializer()
sess.run(var_init_global)
summary_count, summary_count_val=0,0


### Evaluation metric saving dict
eval_metrics    = {"best_val_acc":0, "best_val_dice":0, "train_loss_per_epoch":[], "train_acc_per_epoch":[], "train_dice_per_epoch":[], "val_acc_per_epoch":[], "val_dice_per_epoch":[]}


### Learning rate scheduling
lr_start = 5e-5
lr_collect = network.lrScheduler(lr_start, 0, n_epochs, decay=0.955, turn=1., split_epoch=10, split_ratio=0.33, lrlist=[], mode="interv")


"""============= START TRAINING ============"""
start_time_train = time.time()
slice_size = [768,768]
verbose    = 1000
slicewise_reps = 2

best_val_dice = 0
for epoch in trange(n_epochs, desc='Epoch progress... ', position=0):
    start       = time.time()

    ### Data Yield Function
    training_data_provider = hf.data_provision(data_list=train_data, gt_list=train_gt, weight_list=train_bm, slice_size=slice_size, seed=epoch, \
                                               slicewise_reps=slicewise_reps, augment=0, weighting=weighting)

    ### Temporary Logging lists
    epoch_tr_loss, epoch_tr_dice, epoch_tr_acc = []


    ### Start training iterations
    for i,(training_img,gt,weight_map) in enumerate(tqdm(training_data_provider),desc='Training iterations... ', position=1):
        feed_dict_train = {img: training_img, true_seg: gt, lr:lr_collect[epoch], weights: weight_map, K.learning_phase():1}
        feed_dict_eval  = {img: training_img, true_seg: gt, weights: weight_map, K.learning_phase():0}


        sess.run(optimizer, feed_dict=feed_dict_train)
        summary_t,acc,cost,dice     = sess.run([summary_train,accuracy,loss,dice_coeff], feed_dict=feed_dict_eval)

        writer.add_summary(summary_t, summary_count)

        epoch_tr_loss.append(cost)
        epoch_tr_acc.append(acc)
        epoch_tr_dice.append(dice)

        summary_count+=1


    ### Run validation
    if run_validation:
        epoch_val_acc, epoch_val_dice  = [],[]

        ### Validation data provider
        validation_data_provider = hf.data_provision(data_list=val_data, gt_list=val_gt, weight_list=val_bm, weighting=weighting)

        ### Start validation iterations
        for i,(validation_img, val_gt_frm, vwmap) in enumerate(tqdm(validation_data_provider, desc='Validation iterations... ', position=1)):
            feed_dict_test = {img: validation_img, true_seg: val_gt_frm, weights: vwmap, K.learning_phase():0}
            summary_v,val_acc,val_dice = sess.run([summary_val,val_accuracy,val_dice_coeff],   feed_dict=feed_dict_test)

            #Write to tensorboard summary
            writer.add_summary(summary_v, summary_count_val)

            summary_count_val+=1

            epoch_val_acc.append(val_acc)
            epoch_val_dice.append(val_dice)

        ### Save best validation weights
        if np.mean(epoch_val_dice)>best_val_dice:
            best_val_dice = np.mean(epoch_val_dice)
            best_epoch = epoch
            test = saver.save(sess, Net_name+"_best_dice")

    eval_metrics["train_loss_per_epoch"].append(np.mean(epoch_tr_loss))
    eval_metrics["train_acc_per_epoch"].append(np.mean(epoch_tr_acc))
    eval_metrics["train_dice_per_epoch"].append(np.mean(epoch_tr_dice))
    if run_validation:
        eval_metrics["val_acc_per_epoch"].append(np.mean(epoch_val_acc))
        eval_metrics["val_dice_per_epoch"].append(np.mean(epoch_val_dice))
