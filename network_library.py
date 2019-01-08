"""========================================================="""
"""=================== Load libraries ======================"""
"""========================================================="""
import numpy as np
import keras
import keras.layers.core as core
from keras.datasets import mnist
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Flatten, Input, Activation
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D, ZeroPadding2D, Conv2DTranspose, GlobalAveragePooling2D
from keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, UpSampling3D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.regularizers import l2
import tensorflow as tf
import scipy.ndimage as ndi
from keras.utils import np_utils
from keras_contrib.layers import Deconvolution3D
from keras.preprocessing import image as image_prep
K.set_image_dim_ordering("tf")




"""================================================="""
"""================== UNet  2D/3D =================="""
"""================================================="""
### Create a fully convolutional Unet in functional style that is in its structure
### fully adaptable to the given input parameters:
### FLAGS:
###     mode: 2D or 3D
###     branches: style of the network, length corresponds to number of maxpooling layers, numbers
###               to the amount of convolutional layers before maxpooling
###     fbase: Initial number of filters that is incremented accordingly
###     ksize: Kernel size, is kept constant
###     dropout: If not equal zero, dropout is performed after every maxpooling operation
###     double_before_pooling: Double the filter number before maxpooling as suggested by Szegedy
###     use_reg:    use L2-regularization with factor use_reg
###     n_classes:  number of output classes
###     use_BN:     use Batch Normalization
###     deconvolve: Use deconvolution or Upsampling
###     TF_in:      use tensorflow style or Keras style
###     tf_placeholder: image placeholder if using tensorflow style
###     input_shape: Input to Input_Layer when using Keras style
def get_standard_unet(mode="2D",branches=[2,2,2,2], f_base=32, k_size=3, dropout=0, double_before_pooling = False, connection_type="None",
                      bn_len=2, use_reg=0, n_classes=2, Inject_classification = False, use_BN=False, deconvolve=False, size_inference_3D=False,
                      input_shape=(None,None,None), TF_in = False, tf_placeholder=None):
    if TF_in:
        model      = tf_placeholder
        do_drop     = dropout
    else:
        model = Input(shape=input_shape)

    #### Downward pass ###
    horizontal_pass = []
    skip_list = []


    for i in range(len(branches)):

        model = conv_pool_down(mode, model, f_base*2**i, k_size, branches[i], dropout, use_reg, TF_in, use_BN=use_BN, dbp = double_before_pooling)

        if i!=0 and mode!="3D" and connection_type=="concat":
            model = keras.layers.concatenate([model,skip], axis=-1)
        if i!=0 and mode!="3D" and connection_type=="residual":
            model = keras.layers.merge.Add([model,skip],axis=-1)

        horizontal_pass.append(model)

        if mode=="2D":
            if dropout:
                model = Dropout(dropout)(model)
            model = MaxPooling2D(pool_size=(2,2),padding="same")(model)
            skip  = model
        if mode=="3D":
            if dropout:
                model = Dropout(dropout)(model)
            model = MaxPooling3D(pool_size=(2,2,2),padding="same")(model)


    #### Bottleneck
    model = conv_pool_down(mode, model, f_base*2**len(branches), k_size, bn_len, dropout, use_reg, TF_in, use_BN=use_BN, dbp=double_before_pooling)



    ### Adjust for residual or densenet-style UNet
    if mode!="3D" and connection_type=="concat":
        model = keras.layers.concatenate([model,skip], axis=-1)
    if mode!="3D" and connection_type=="residual":
        model = keras.layers.merge.Add([model,skip],axis=-1)


    ### Option to double filters before pooling
    if double_before_pooling:
        fbase_use = f_base*2**(len(branches)+1)
    else:
        fbase_use = f_base*2**(len(branches)-1)


    ### Set Upsampling/Transposed Convolution Options
    if mode=="2D":
        if not deconvolve:
            model = UpSampling2D((2,2))(model)
        else:
            model = Conv2DTranspose(filters = fbase_use, kernel_size=(2,2), strides=2, padding="same")(model)
    if mode=="3D":
        if not deconvolve:
            model = UpSampling3D(size=(2,2,2))(model)
        else:
            #### contrib_layer implementation of 3D deconvolution
            out_shape, in_shape = get_up_shapes(fbase_use, model.get_shape()[1:-1].as_list(), used_pool_size = (2,2,2))
            model               = Deconvolution3D(filters=fbase_use, kernel_size=(2,2,2), strides=(2,2,2), output_shape=out_shape, input_shape=in_shape)(model)


    #### Upsampling branch
    for i in range(len(branches)):
        model      = keras.layers.concatenate([model,horizontal_pass[-(i+1)]], axis=-1)
        skip       = model

        if double_before_pooling:
            fbase_use /= 2

        if i==len(branches)-1:
            model = conv_pool_up(mode, model, fbase_use, k_size, branches[i],use_reg, use_BN=use_BN, deconvolve=deconvolve,s_i_3D = size_inference_3D,\
                                 dbp = double_before_pooling, skip=skip, con_type = connection_type, halt=True)
        else:
            model = conv_pool_up(mode, model, fbase_use, k_size, branches[i],use_reg, use_BN=use_BN, deconvolve=deconvolve,s_i_3D = size_inference_3D,\
                                 dbp = double_before_pooling, skip=skip, con_type=connection_type, halt=False)

        if not double_before_pooling:
            fbase_use /= 2


    ### Output Layer
    if mode=="2D":
        model = Conv2D(n_classes,kernel_size=(1,1),strides=1)(model)
        model = Activation("sigmoid")(model)
    if mode=="3D":
        model = Conv3D(n_classes,kernel_size=(1,1,1),strides=1)(model)
        model = Activation("sigmoid")(model)


    if TF_in:
        if Inject_classification:
            return model,inject,len(branches)
        else:
            return model,len(branches)
    else:
        return Model(inputs=inputs, outputs=model),len(branches)

def conv_pool_down(mode, model, no_f_base, f_size, conv_depth, dropout, use_reg,TF_in=False, do_drop=0, use_BN=False, dbp = False):
    ### Function to set up a down-convolutional block
    for i in range(conv_depth):
        if i==conv_depth-1 and dbp:
            no_f_base*=2
        if mode=="2D":
            model = Conv2D(no_f_base, kernel_size=f_size,strides=1,kernel_initializer="he_uniform",padding="same",
                           kernel_regularizer=keras.regularizers.l2(use_reg))(model)
            if use_BN:
                model = BatchNormalization()(model)
            model = Activation("relu")(model)

        if mode=="3D":
            model = Conv3D(no_f_base, kernel_size=f_size,strides=1,kernel_initializer="he_uniform",padding="same",
                           kernel_regularizer=keras.regularizers.l2(use_reg))(model)
            if use_BN:
                model = BatchNormalization()(model)
            model = Activation("relu")(model)

    return model


def conv_pool_up(mode, model, no_f_base, f_size, conv_depth, use_reg, use_BN=False, deconvolve=False,s_i_3D=False, dbp = False, skip=None, con_type="None", halt=False):
    ### Function to set up the up-convolutional block
    for i in range(conv_depth):
        if mode=="2D":
            model = Conv2D(no_f_base, kernel_size=f_size, strides=1, kernel_initializer="he_uniform", padding="same",
                           kernel_regularizer=keras.regularizers.l2(use_reg))(model)
        if mode=="3D":
            model = Conv3D(no_f_base, kernel_size=f_size, strides=1, kernel_initializer="he_uniform", padding="same",
                           kernel_regularizer=keras.regularizers.l2(use_reg))(model)
        if use_BN:
            model = BatchNormalization()(model)
        model = Activation("relu")(model)


    if mode!="3D" and con_type=="concat":
        model = keras.layers.concatenate([model,skip], axis=-1)
    if mode!="3D" and con_type=="residual":
        model = keras.layers.merge.Add([model,skip],axis=-1)

    ### Case difference if last block option
    if not halt:
        if dbp:
            fbase_use  = no_f_base
        else:
            fbase_use  = no_f_base/2
        if mode=="2D":
            if not deconvolve:
                model = UpSampling2D(size=(2,2))(model)
            else:
                model = Conv2DTranspose(filters = fbase_use, kernel_size=(2,2), strides=2, padding="same")(model)
        if mode=="3D":
            if not deconvolve:
                model = UpSampling3D(size=(2,2,2))(model)
            else:
                out_shape, in_shape = get_up_shapes(fbase_use, model.get_shape()[1:-1].as_list(), used_pool_size = (2,2,2))
                model               = Deconvolution3D(filters=fbase_use, kernel_size=(2,2,2), strides=(2,2,2), output_shape=out_shape, input_shape=in_shape)(model)

    return model



def get_up_shapes(n_filters, input_shape, used_pool_size, mode="tf"):
    ### Function to estimate output shapes
    in_shape = input_shape
    out_shape   = np.multiply(input_shape, used_pool_size)
    if mode=="tf":
        return tuple([None]+[x for x in out_shape]+[n_filters]),tuple([None]+[x for x in in_shape]+[n_filters])
    elif mode=="th":
        return tuple([None, n_filters]+[x for x in out_shape]+[n_filters]),tuple([None]+[x for x in in_shape])



"""==========================================="""
"""================= UNet easy ==============="""
"""==========================================="""
#### Unet in Keras' Sequential style either in 2D or 3D
def construct_basic_unet(tf_placeholder, mode="2D"):
    if mode=="2D":
        inputs = tf_placeholder
        conv1 = Conv2D(32, kernel_size=(3,3), strides=1, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, kernel_size=(3,3), strides=1, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, kernel_size=(3,3), strides=1,  activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, kernel_size=(3,3), strides=1, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, kernel_size=(3,3), strides=1, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, kernel_size=(3,3), strides=1, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, kernel_size=(3,3), strides=1, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, kernel_size=(3,3), strides=1, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, kernel_size=(3,3), strides=1, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, kernel_size=(3,3), strides=1, activation='relu', padding='same')(conv5)

        up6 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(conv5), conv4],axis=-1)
        conv6 = Conv2D(256, kernel_size=(3,3), strides=1, activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, kernel_size=(3,3), strides=1, activation='relu', padding='same')(conv6)

        up7 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
        conv7 = Conv2D(128, kernel_size=(3,3), strides=1, activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, kernel_size=(3,3), strides=1, activation='relu', padding='same')(conv7)

        up8 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
        conv8 = Conv2D(64, kernel_size=(3,3), strides=1, activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, kernel_size=(3,3), strides=1, activation='relu', padding='same')(conv8)

        up9 = keras.layers.concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
        conv9 = Conv2D(32, kernel_size=(3,3), strides=1, activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, kernel_size=(3,3), strides=1, activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, kernel_size=(1,1), strides=1, activation='sigmoid')(conv9)
        return conv10,4

    elif mode=="3D":

        downsize_filters_factor=1
        pool_size=(2,2,2)
        n_labels = 1
        dropout=0.2

        inputs = tf_placeholder
        conv1 = Conv3D(int(32/downsize_filters_factor), (3, 3, 3), activation='relu',padding='same')(inputs)
        conv1 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',padding='same')(conv1)
        pool1 = MaxPooling3D(pool_size=pool_size)(conv1)
        drop1 = Dropout(dropout)(pool1)

        conv2 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',padding='same')(drop1)
        conv2 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',padding='same')(conv2)
        pool2 = MaxPooling3D(pool_size=pool_size)(conv2)

        drop2 = Dropout(dropout)(pool2)

        conv3 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',padding='same')(drop2)
        conv3 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu',padding='same')(conv3)
        pool3 = MaxPooling3D(pool_size=pool_size)(conv3)

        drop3 = Dropout(dropout)(pool3)

        conv4 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu',padding='same')(drop3)
        conv4 = Conv3D(int(512/downsize_filters_factor), (3, 3, 3), activation='relu',padding='same')(conv4)

        up5 = keras.layers.concatenate([UpSampling3D(size=(2, 2, 2))(conv4), conv3], axis=-1)
        conv5 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up5)
        conv5 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu',padding='same')(conv5)

        up6 = keras.layers.concatenate([UpSampling3D(size=(2, 2, 2))(conv5), conv2], axis=-1)
        conv6 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu',padding='same')(conv6)

        up7 = keras.layers.concatenate([UpSampling3D(size=(2, 2, 2))(conv6), conv1], axis=-1)
        conv7 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu',padding='same')(conv7)
        conv8 = Conv3D(n_labels, (1, 1, 1))(conv7)
        act = Activation('sigmoid')(conv8)

        return act,3




"""================================================="""
"""============= Tiramisu/DenseNet 2D/3D ==========="""
"""================================================="""
#### Implementation of the Densely-connected/Hundred-layer Tiramisu network
#### NOTE: Tiramisu and DenseNet are heavily related; all layers listed below can also be used
####       for densenet creation.

##### Construct Tiramisu:
##### FLAGS:
#####   -mode: 2D or 3D
#####   -upmode: deconvolution("deconv") or upsampling ("upsampling")
#####   -verbose: output verbosity, the lower, the more output
#####   -denseblocks_down: Number of denseblocks of given sizes for feature extraction
#####   -denseblocks_up: Number of denseblocks of given size for image recreation; if None, denseblocks_down is mirrored
#####   -blockbottom: Bottleneck denseblock size
#####   -use_reg/dropout/use_BN/n_classes/tf_placeholder: Same as UNet implementation
#####   -act: Output activation
def construct_tiramisu(mode, upmode, verbose, denseblocks_down=[4,5,7,10,12], denseblocks_up=None, blockbottom=[15], f_base=32, k_size=3, k=16,
                              use_reg=1e-4, dropout=0, act="sigmoid", use_BN=False, n_classes=1, tf_placeholder=None):
    if denseblocks_up is None:
        denseblocks_up = denseblocks_down
    if mode=="2D":
        model = Conv2D(filters=f_base, kernel_size=k_size, use_bias=False, padding="same", activation="linear", kernel_initializer="he_uniform", kernel_regularizer=keras.regularizers.l2(use_reg))(tf_placeholder)
    if mode=="3D":
        model = Conv3D(filters=f_base, kernel_size=k_size, use_bias=False, padding="same", activation="linear", kernel_initializer="he_uniform", kernel_regularizer=keras.regularizers.l2(use_reg))(tf_placeholder)

    ######  Feature extraction path
    nfilters = f_base
    skip_list = []
    for blocks_down in xrange(len(denseblocks_down)):
        block_to_downsample   = denseblock_ex(mode, model, use_BN, denseblocks_down[blocks_down], k, dropout, use_reg)
        nfilters             += denseblocks_down[blocks_down]*k
        conc_block            = keras.layers.concatenate(block_to_downsample,axis=-1)
        model                 = keras.layers.concatenate([conc_block, model],axis=-1)
        skip_list.append(model)
        model = densenet_transition(mode, model, nfilters, use_BN, dropout, use_reg)
    skip_list   = skip_list[::-1]

    #Note about the used m number: for upsampling, it is NOT simply the filter number, but rather a sum comprising
    #the filters in the downsampling part (skip_list), the number of feature maps from the upsampled block (denseblocks[-(blocks_up+1)] * growth_rate)
    #and the number of feature maps in the new block (denseblocks[blocks_up] * k), where in this definition denseblocks also includes
    #blockbottom.

    ###### blockbottom
    block_to_upsample   = []
    for bottomblock in xrange(len(blockbottom)):
        block_to_upsample   = denseblock_ex(mode, model, use_BN, blockbottom[bottomblock], k, dropout, use_reg)
        filter_stack_up     = k*blockbottom[bottomblock]
        if len(blockbottom)>1 and bottomblock<len(blockbottom)-1:
            model = transition_down_ex(mode, keras.layers.concatenate(block_to_upsample, axis=-1), use_BN, filter_stack_up, dropout, use_reg)

    ###### Upsampling
    for blocks_up in xrange(len(denseblocks_up)):
        model = transition_up_ex(mode, upmode, skip_list[blocks_up], block_to_upsample, filter_stack_up, dropout, use_reg)
        block_to_upsample   = denseblock_ex(mode, model, use_BN, denseblocks_up[-(blocks_up+1)], k, dropout, use_reg)
        model = keras.layers.concatenate(block_to_upsample)
        filter_stack_up = k*denseblocks_up[-(blocks_up+1)]


    model = OutLayer_ex(mode, model, n_classes, use_reg, act=act)

    return model, len(denseblocks_down)




"""=================================================================================="""
"""=============== All functional layers for Tiramisu and DenseNet =================="""
"""=================================================================================="""
def densenet_transition(mode, model, n_filter, use_BN, dropout, use_reg):
    ### Transition Layer between densenet blocks
    if use_BN:
        model = BatchNormalization(axis=-1, beta_regularizer=keras.regularizers.l2(use_reg), gamma_regularizer=keras.regularizers.l2(use_reg))(model)
    model = Activation("relu")(model)
    if mode=="2D":
        model = Conv2D(n_filter, kernel_size=1, strides=1, kernel_initializer="he_uniform", padding="same", kernel_regularizer=keras.regularizers.l2(use_reg))(model)
    if mode=="3D":
        model = Conv3D(n_filter, kernel_size=1, strides=1, kernel_initializer="he_uniform", padding="same", kernel_regularizer=keras.regularizers.l2(use_reg))(model)
    if dropout:
        model = Dropout(dropout)(model)
    if mode=="2D":
        model = AveragePooling2D((2,2),strides=(2,2))(model)
    if mode=="3D":
        model = AveragePooling3D((2,2,2),strides=(2,2,2))(model)
    return model

def tiramisu_layer(mode, model, n_filter, use_BN, dropout, use_reg, ksize=3, is_bottom=False):
    ### Standard Dense/Tiramisu layer with BN, Dropout and Relu activation
    if use_BN:
        model = BatchNormalization(axis=-1, beta_regularizer=keras.regularizers.l2(use_reg),gamma_regularizer=keras.regularizers.l2(use_reg))(model)
    if is_bottom:
        suggested_filter = n_filter*4
        if mode=="2D":
            model = Conv2D(filters=suggested_filter, kernel_size=ksize, strides=1, activation="relu", kernel_initializer="he_uniform", use_bias=False, padding="same", kernel_regularizer=keras.regularizers.l2(use_reg))(model)
        if mode=="3D":
            model = Conv3D(filters=suggested_filter, kernel_size=ksize, strides=1, activation="relu", kernel_initializer="he_uniform", use_bias=False, padding="same", kernel_regularizer=keras.regularizers.l2(use_reg))(model)
        if dropout:
            model = Dropout(dropout)(model)
        if use_BN:
            model = BatchNormalization(axis=-1, beta_regularizer=keras.regularizers.l2(use_reg),gamma_regularizer=keras.regularizers.l2(use_reg))(model)
    if mode=="2D":
        model = Activation("relu")(model)
        model = Conv2D(n_filter, kernel_size=3,strides=1,kernel_initializer="he_uniform",padding="same",kernel_regularizer=keras.regularizers.l2(use_reg))(model)
    if mode=="3D":
        model = Activation("relu")(model)
        model = Conv3D(n_filter, kernel_size=3,strides=1,kernel_initializer="he_uniform",padding="same",kernel_regularizer=keras.regularizers.l2(use_reg))(model)
    if dropout!=0:
        model = Dropout(dropout)(model)
    return model

#NOTE: Denseblocks and mini BNReluConv-layer are the same as in the DenseNet.
def transition_down_ex(mode, model, n_filters, use_BN, dropout, use_reg):
    ### Pooling Layer for both Tiramisu and DenseNet
    model  = tiramisu_layer(mode, model, n_filters, use_BN, dropout, use_reg, ksize=1)
    if mode=="2D":
        model = MaxPooling2D((2,2), strides=(2,2))(model)
    if mode=="3D":
        model = MaxPooling3D((2,2,2), strides=(2,2,2))(model)
    return model

def transition_up_ex(mode, upmode, skip_connection, block_to_upsample, n_filters_keep, dropout, use_reg):
    ### Upsampling/Transposed Convolution Layer
    mslice = keras.layers.concatenate(block_to_upsample,axis=-1)
    if mode == "2D":
        if upmode=="deconv":
            mslice  = Conv2DTranspose(filters = n_filters_keep, kernel_size=3, strides=2, activation="linear", use_bias=False, padding="same", kernel_initializer="he_uniform", kernel_regularizer=keras.regularizers.l2(use_reg))(mslice)
        if upmode=="upsampling":
            mslice  = UpSampling2D(size=(2,2))(mslice)
            mslice  = Conv2D(n_filters_keep, kernel_size=3, padding="same", activation="relu", use_bias=False, kernel_initializer="he_uniform", kernel_regularizer=keras.regularizers.l2(use_reg))(mslice)
    if mode == "3D":
        if upmode=="deconv":
            mslice  = NK.Conv3DTranspose(n_filters_keep, kernel_size=3, strides=2, use_bias=False, padding="same", kernel_initializer="he_uniform", kernel_regularizer=keras.regularizers.l2(use_reg))(mslice)
    return keras.layers.concatenate([mslice, skip_connection], axis=-1)


def denseblock_ex(mode, model, use_BN, n_layers, k, dropout, use_reg, is_bottom=False):
    ### Standard Denseblock, also used for Tiramisu
    block_to_upsample = []
    for j in xrange(n_layers):
        mslice  = tiramisu_layer(mode, model, k, use_BN, dropout, use_reg, is_bottom)
        block_to_upsample.append(mslice)
        model   = keras.layers.concatenate([model, mslice])
    return block_to_upsample

def OutLayer_ex(mode, model, n_filters, use_reg, act="sigmoid"):
    ### Output Layer
    if act=="sigmoid":
        if mode=="2D":
            model = Conv2D(filters=n_filters,kernel_size=1, strides=1, use_bias=False, padding="same", activation="sigmoid", kernel_regularizer=keras.regularizers.l2(use_reg))(model)
        if mode=="3D":
            model = Conv3D(filters=n_filters,kernel_size=1, strides=1, use_bias=False, padding="same", activation="sigmoid", kernel_regularizer=keras.regularizers.l2(use_reg))(model)
    if act=="softmax":
        model       = Conv2D(filters=n_filters,kernel_size=(1,1), strides=1, use_bias=False, padding="same", activation="linear", kernel_regularizer=keras.regularizers.l2(use_reg))(model)
        temp_shape  = model.get_shape()
        model       = core.Reshape((-1,n_filters))(model)
        model       = Activation(act)(model)
        model       = core.Reshape((int(temp_shape[1]),int(temp_shape[2]),n_filters))(model)
    return model






"""======================================"""
"""============ Misc functions =========="""
"""======================================"""
### Verbose number limited printing
def verbose_print(text, verbose_limit, verbose):
    if verbose>=verbose_limit:
        print(text)

### Flatten layer in single dimension
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    layer_flat = tf.reshape(layer, [-1, num_features])
    return layer_flat, num_features

### Save training session
def save(session, checkpoint_file="checkpoint.chk"):
    saver = tf.train.Saver()
    saver.save(session, checkpoint_file)

### Create a learning rate schedule by either exponentially decaying
### or following a reverse sigmoid trend
def lrScheduler(m=1e-5,ep=0,ep_max=100,decay=0.001,split_epoch=10,split_ratio=0.1,turn=1,lrlist=[],mode="sig"):
    if ep==ep_max:
        return lrlist
    else:
        lrlist.append(m)
        if mode=="sig":
            return lrScheduler(m/(turn+decay*ep),ep+1, ep_max, decay, turn,lrlist,mode)
        elif mode=="dec":
            return lrScheduler(m*decay,ep+1,ep_max,decay,turn,lrlist,mode)
        elif mode=="interv":
            if ep%split_epoch==0 and ep!=0:
                return lrScheduler(m*split_ratio, ep+1, ep_max, lrlist=lrlist, split_epoch=split_epoch, split_ratio=split_ratio, mode=mode)
            else:
                return lrScheduler(m, ep+1, ep_max, lrlist=lrlist, split_epoch=split_epoch, split_ratio=split_ratio, mode=mode)



class Load_and_Restore_unnamed():
    """  Importing and running isolated TF graph """
    def __init__(self, loc, network_type, parameters, channels=1):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess  = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            K.set_learning_phase(0)
            lr = tf.placeholder(tf.float32,shape=[])
            weights = tf.placeholder(tf.float32, shape=(2,2))
            #Load Network for Liver Segmentation
            self.img      = tf.placeholder(tf.float32, shape=(None, None, None, channels))
            self.true_seg = tf.placeholder(tf.float32, shape=(None, None, None, 1))
            # img         = tf.placeholder(tf.float32, shape=(1, 128,128, 1))
            # true_seg    = tf.placeholder(tf.float32, shape=(1, 128,128, 1))

            if network_type=="Tiramisu_2D":
                self.pred_seg, self.mpool_len  = construct_tiramisu(tf_placeholder=self.img, **parameters)
            if network_type=="Unet_2D":
                self.pred_seg, self.mpool_len = get_standard_unet(tf_placeholder=self.img, **parameters)
            # saver = tf.train.import_meta_graph(loc + '.meta',
            #                                    clear_devices=True)
            saver = tf.train.Saver()
            saver.restore(self.sess, loc)

            self.total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                self.total_parameters += variable_parameters
            # Get activation function from saved collection
            # You may need to change this in case you name it differently
            # self.pred_seg = tf.get_collection("Net_name")

    def run(self, data):
        """ Running the activation function previously imported """
        # The 'x' corresponds to name of input placeholder
        return self.sess.run(self.pred_seg, feed_dict={self.img: data})





class Load_and_Restore_named():
    """  Importing and running isolated TF graph """
    def __init__(self, loc, network_type, network_name):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess  = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            K.set_learning_phase(0)
            saver = tf.train.import_meta_graph(loc + '.meta',clear_devices=True)
            saver.restore(self.sess, loc)
            # Get activation function from saved collection
            # You may need to change this in case you name it differently
            self.pred_seg = tf.get_collection(network_name)

    def run(self, data):
        """ Running the activation function previously imported """
        # The 'x' corresponds to name of input placeholder
        return self.sess.run(self.pred_seg, feed_dict={"img": data})








"""========================================================="""
"""================= Loss functions/Metrics ================"""
"""========================================================="""
### Dice Coefficient
def dice_coeff(flattened_true_label, flattened_pred_label, eps=1., mode="tf", red_eps1 = 1.):
    if mode=="tf":
        overlap                 = K.sum(flattened_true_label*flattened_pred_label)
        dice                    = tf.clip_by_value(tf.div((2.*overlap+red_eps1*eps),(K.sum(flattened_true_label)+K.sum(flattened_pred_label)+eps)),1e-8,0.99999999)
    if mode=="np":
        overlap                 = np.sum(flattened_true_label*flattened_pred_label)
        dice                    = np.clip(((2.*overlap+red_eps1*eps)/(np.sum(flattened_true_label)+np.sum(flattened_pred_label)+eps)),1e-8,0.99999999)
    return dice

### Dice Loss
def dice_loss(true_label, pred_label, eps=1., red_eps1=1., mode="tf"):
    return -dice_coeff(true_label, pred_label, eps, mode, red_eps1)

### Binary Crossentropy Loss
def binary_crossentropy(true_label, pred_label, is_logit=False):
    bc = K.mean(K.binary_crossentropy(pred_label, true_label, from_logits=is_logit),axis=-1)
    return bc

### Categorical Crossentropy Loss
def categorical_crossentropy_4_seg(true_label, pred_label):
    return K.categorical_crossentropy(K.flatten(true_label), K.flatten(pred_label))

### Kullback Leibler Divergence Loss
def kullback_leibler_divergence_4_seg(true_label, pred_label):
    flattened_true_label = K.clip(true_label, K.epsilon(), 1)
    flattened_pred_label = K.clip(pred_label, K.epsilon(), 1)
    return K.sum(flattened_true_label * K.log(flattened_true_label / flattened_pred_label), axis=-1)

### Jaccard Coefficient
def jaccard_coeff(flattened_true_label, flattened_pred_label, mode="tf", eps=1.):
    if mode=="tf":
        intersection            = K.sum(flattened_true_label*flattened_pred_label)+eps
        jaccard                 = tf.clip_by_value(tf.div(intersection,(K.sum(tf.square(flattened_true_label))+K.sum(tf.square(flattened_pred_label))-1.*intersection)+eps),1e-10,0.9999999)
    if mode=="np":
        intersection            = np.sum(flattened_true_label*flattened_pred_label)+eps
        jaccard                 = np.clip(intersection/np.clip((np.sum(np.square(flattened_true_label))+np.sum(np.square(flattened_pred_label))-1.*intersection)+eps,1e-8,None),1e-8,0.999999)
    return jaccard

### Tversky-Coefficient based loss with FP/FN-weighting parameters alpha and beta
def tversky_loss(flattened_true_label, flattened_pred_label, mode="tf", alpha=0.5, beta=0.5, eps=1.,red_eps1=1.):
    if mode=="tf":
        overlap                 = K.sum(flattened_true_label*flattened_pred_label)*1.
        tversky                 = tf.clip_by_value(tf.div((overlap+red_eps1*eps),(overlap+alpha*K.sum(flattened_pred_label*(1.-flattened_true_label))+beta*K.sum((1.-flattened_pred_label)*flattened_true_label)+eps)),1e-8,0.999999)
    if mode=="np":
        overlap                 = np.sum(flattened_true_label*flattened_pred_label)*1.
        tversky                 = np.clip(np.divide((overlap+red_eps1*eps),(overlap+alpha*np.sum(flattened_pred_label*(1.-flattened_true_label))+beta*np.sum((1.-flattened_pred_label)*flattened_true_label)+eps)),1e-8,0.999999)
    return -tversky

### Pixel-weighted Tversky-loss
def wbc_tversky(flattened_true_label, flattened_pred_label, mode="tf", alpha=0.5, beta=0.5, eps=1., red_eps1=1., weights=None):
    idx_getter  = K.cast(flattened_true_label,tf.int32)
    classf_weights  = K.gather(K.flatten(weights),idx_getter)
    return 10*K.mean(tf.multiply(classf_weights,K.binary_crossentropy(flattened_pred_label,flattened_true_label)),axis=-1)+tversky_loss(flattened_true_label, flattened_pred_label ,alpha=alpha, beta=beta, eps=eps ,red_eps1=red_eps1)


### Combination of binary crossentropy and jaccard loss
def bc_jac_loss(true_label, pred_label, eps=10., mode="tf", type="prod", pfactor = 1.):
    if type=="prod":
        return binary_crossentropy(true_label,pred_label)*(1-jaccard_coeff(true_label,pred_label,mode,eps))
    elif type=="sum":
        return binary_crossentropy(true_label,pred_label)-pfactor*jaccard_coeff(true_label,pred_label,mode,eps)

#### Jaccard loss
def jaccard_distance(flattened_true_label, flattened_pred_label, mode="tf"):
    #Feed flattened output in
    j_dist                  = -jaccard_coeff(flattened_true_label, flattened_pred_label, mode)
    return 1+j_dist

### Confusion matrix
def confusion_matrix(flattened_true_label, flattened_pred_label, return_list = False, mode="tf"):
    idx,freq                  = np.unique(2.*K.round(flattened_pred_label)+flattened_true_label, return_counts=True)
    if return_list:
        return freq
    else:
        freq_vec = np.arange(4)
        print idx
        for m,j in enumerate(idx):
            freq_vec[idx]=freq[m]
        confusion_mat           = freq_vec.reshape(2,2)
        return confusion_mat

### Pixelweise weighted binary crossentropy loss
def weighted_binary_crossentropy_pw(y_true, y_pred, weights):
    return K.mean(tf.multiply(weights,K.binary_crossentropy(y_pred,y_true)), axis=-1)

### label-weighted binary crossentropy loss
def weighted_binary_crossentropy_lw(y_true, y_pred, weights):
    idx_getter  = K.cast(y_true,tf.int32)
    classf_weights  = K.gather(K.flatten(weights),idx_getter)
    return K.mean(tf.multiply(classf_weights,K.binary_crossentropy(y_pred,y_true)),axis=-1)

### Case-weighted binary crossentropy loss
def weighted_binary_crossentropy_cw(y_true, y_pred, weights):
    idx         = np.arange(4)
    weights_f   = K.flatten(weights)
    y_pred_label= K.round(y_pred)
    idx_getter  = 2.*y_pred_label+y_true
    idx_getter  = K.cast(idx_getter,tf.int32)
    classf_weights  = K.gather(weights_f,idx_getter)
    return K.mean(tf.multiply(classf_weights,K.binary_crossentropy(y_pred,y_true)),axis=-1)


### Focal loss with case weighting
def weighted_focal_binary_crossentropy(y_true, y_pred, weights, weight_type="None", gamma=2, from_logits=False):
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1.-eps)
    if weight_type=="None":
        return K.mean(-1.*(K.pow(1.-y_pred, gamma)*y_true*K.log(y_pred)+(1.-y_true)*K.pow(y_pred, gamma)*K.log(1.-y_pred)) ,axis=-1)
    elif weight_type=="pixelwise":
        return K.mean(tf.multiply(weights,-1.*(K.pow(1.-y_pred, gamma)*y_true*K.log(y_pred)+(1.-y_true)*K.pow(y_pred, gamma)*K.log(1.-y_pred))), axis=-1)
    elif weight_type=="labelwise":
        idx_getter  = K.cast(y_true,tf.int32)
        label_weights  = K.gather(K.flatten(weights),idx_getter)
        return K.mean(tf.multiply(label_weights,-1.*(K.pow(1.-y_pred, gamma)*y_true*K.log(y_pred)+(1.-y_true)*K.pow(y_pred, gamma)*K.log(1.-y_pred)), axis=-1))
    elif weight_type=="classwise":
        idx         = np.arange(4)
        weights_f   = K.flatten(weights)
        y_pred_label= K.round(y_pred)
        idx_getter  = 2.*y_pred_label+y_true
        idx_getter  = K.cast(idx_getter,tf.int32)
        class_weights  = K.gather(weights_f,idx_getter)
        return K.mean(tf.multiply(class_weights,-1.*(K.pow(1.-y_pred, gamma)*y_true*K.log(y_pred)+(1.-y_true)*K.pow(y_pred, gamma)*K.log(1.-y_pred)), axis=-1))







""""""""""""""""""""""""""""""""""""
""" Basic Image Augmentation     """
""""""""""""""""""""""""""""""""""""
def truncated_normal(shape, max_dev=3.5):
    return np.clip(np.random.normal(0.0, 1.0, shape), -max_dev, max_dev)/max_dev



def rotate(x, theta, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image_prep.transform_matrix_offset_center(rotation_matrix, h, w)
    x = image_prep.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def shift(x, wshift, hshift, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hshift * h
    ty = wshift * w
    translation_matrix = np.array([[1, 0, tx],
                                   [0, 1, ty],
                                   [0, 0, 1]])
    transform_matrix = translation_matrix  # no need to do offset
    x = image_prep.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def shear(x, shear, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image_prep.transform_matrix_offset_center(shear_matrix, h, w)
    x = image_prep.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

def zoom(x, zx, zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.):
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = image_prep.transform_matrix_offset_center(zoom_matrix, h, w)
    x = image_prep.apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x

### Image Flipping
def random_flip(img, mask, wmap):
    img = image_prep.flip_axis(img, 1)
    mask = image_prep.flip_axis(mask, 1)
    if wmap is not None:
        wmap = image_prep.flip_axis(wmap, 1)
        return img, mask, wmap
    else:
        return img, mask, None

def random_rotate(img, mask, wmap, rotate_limit=(-20, 20)):
    theta = np.pi / 180 * np.clip(truncated_normal(1)*rotate_limit[1], rotate_limit[0],rotate_limit[1])[0]
    img = rotate(img, theta)
    mask = rotate(mask, theta)
    wmap = rotate(wmap, theta)
    if wmap is not None:
        wmap = rotate(wmap, theta)
        return img, mask, wmap
    else:
        return img, mask, None

def random_zoom(img, mask, wmap, zoom_range=(0.8, 1)):
    zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    img = zoom(img, zx, zy)
    mask = zoom(mask, zx, zy)
    if wmap is not None:
        wmap = zoom(wmap, zx, zy)
        return img, mask, wmap
    else:
        return img, mask, None

def random_shift(img, mask, wmap, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1)):
    wshift = np.random.uniform(w_limit[0], w_limit[1])
    hshift = np.random.uniform(h_limit[0], h_limit[1])
    img = shift(img, wshift, hshift)
    mask = shift(mask, wshift, hshift)
    if wmap is not None:
        wmap = shift(wmap, wshift, hshift)
        return img, mask, wmap
    else:
        return img, mask, None

def random_shear(img, mask, wmap, intensity_range=(-0.5, 0.5)):
    sh = np.random.uniform(-intensity_range[0], intensity_range[1])
    img = shear(img, sh)
    mask = shear(mask, sh)
    if wmap is not None:
        wmap = shear(wmap, sh)
        return img, mask, wmap
    else:
        return img, mask, None

def random_channel_shift(x, limit, channel_axis=2):
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_ch + np.random.uniform(-limit, limit), min_x, max_x) for x_ch in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x

def random_contrast(img, limit=(-0.2, 0.2)):
    alpha = 1.0 + np.random.uniform(limit[0], limit[1])
    coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
    gray = img * coef
    gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
    img = alpha * img + gray
    img = np.clip(img, 0., 1.)
    return img

def random_brightness(img, limit=(-0.2, 0.2)):
    alpha = 1.0 + np.random.uniform(limit[0], limit[1])
    img = alpha * img
    img = np.clip(img, 0., 1.)
    return img

def random_saturation(img, limit=(-0.2, 0.2)):
    alpha = 1.0 + np.random.uniform(limit[0], limit[1])
    coef = np.array([[[0.114, 0.587, 0.299]]])
    gray = img * coef
    gray = np.sum(gray, axis=2, keepdims=True)
    img = alpha * img + (1. - alpha) * gray
    img = np.clip(img, 0., 1.)
    return img
