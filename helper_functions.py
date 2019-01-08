"""====================== LIBRARIES ================="""
import numpy as np, scipy.misc as sm, cv2, os
import matplotlib.pyplot as plt
import network_library as network

### Mean and standard deviation values (precomputed) over the full dataset
#   Mean for each channel: (174.40739528397251, 176.17585317342369, 178.04820052575519)
mu = np.array([174.4074, 176.1759, 178.0482])
#   Std  for each channel: (62.522683695177058, 63.31837320659745, 62.266268654844517)
sd = np.array([62.5227, 63.3184, 62.2663])



"""================ IMAGE AUGMENTATION ========================"""
def transform_image(img, mask, weight_map=None, seed=1, trans_types = ["flip", "shear", "rotate", "shift", "zoom", "contrast", "brightness", "saturation", "color_shift"]):
    rng = np.random.RandomState(seed)
    random_selection = seed>0

    if "color_shift" in trans_types and (not random_selection or rng.randint(2)):
        img = network.random_channel_shift(img, limit=0.02)
    if "brightness" in trans_types and (not random_selection or rng.randint(2)):
        img = network.random_brightness(img, limit=(-0.2, 0.2))
    if "contrast" in trans_types and (not random_selection or rng.randint(2)):
        img = network.random_contrast(img, limit=(-0.2, 0.2))
    if "saturation" in trans_types and (not random_selection or rng.randint(2)):
        img = network.random_saturation(img, limit=(-0.2, 0.2))
    if "rotate" in trans_types and (not random_selection or rng.randint(2)):
        img, mask, weight_map = network.random_rotate(img, mask, weight_map, rotate_limit=(-20, 20))
    if "shear" in trans_types and (not random_selection or rng.randint(2)):
        img, mask, weight_map = network.random_shear(img, mask, weight_map, intensity_range=(-0.3, 0.3))
    if "flip" in trans_types and (not random_selection or rng.randint(2)):
        img, mask, weight_map = network.random_flip(img, mask, weight_map)
    if "shift" in trans_types and (not random_selection or rng.randint(2)):
        img, mask, weight_map = network.random_shift(img, mask, weight_map, w_limit=(-0.1, 0.1), h_limit=(-0.1, 0.1))
    if "zoom" in trans_types and (not random_selection or rng.randint(2)):
        img, mask, weight_map = network.random_zoom(img, mask, weight_map, zoom_range=(0.8, 1))

    return img, mask, weight_map



"""=================== DATA GENERATOR =================="""
def data_provision(data_list, gt_list, weight_list=[], slice_size=[], seed=-1, data_stand = True, slicewise_reps=1, augment=0, weighting=0):
    ### Shuffle Data
    if seed!=-1:
        np.random.seed(seed)
        np.random.shuffle(data_list)
        np.random.seed(seed)
        np.random.shuffle(gt_list)
        np.random.seed(seed)
        np.random.shuffle(weight_list)

    ### Add weight map info if required
    if len(weight_list)==0:
        use_weights=False
        weight_list = np.arange(len(gt_list))
    else:
        use_weights=True


    ### Yield training mask, slices and optionally weightmap.
    ### If the number of slices to use is not 1, the generator will make use of multiple
    ### images of the same car to try and improve the segmentation.
    for j,k,l in zip(data_list, gt_list, weight_list):
        dm = cv2.imread(j)
        dgt= (sm.imread(k)[:,:,0:1]/255).astype(np.uint8)
        if use_weights:
            bm = np.expand_dims(np.exp(weighting*np.load(l)),2)

        if len(slice_size)==0:
            dm = np.pad(dm,((0,0),(0,2),(0,0)),mode="reflect")
            dgt= np.pad(dgt,((0,0),(0,2),(0,0)),mode="reflect")
            if use_weights:
                bm= np.pad(bm,((0,0),(0,2),(0,0)),mode="reflect")

            if data_stand:
                if use_weights:
                    yield np.expand_dims((dm-mu)/sd,0), np.expand_dims(dgt,0), np.expand_dims(bm,0)
                else:
                    yield np.expand_dims((dm-mu)/sd,0), np.expand_dims(dgt,0), None
            else:
                if use_weights:
                    yield np.expand_dims(dm,0), np.expand_dims(dgt,0), np.expand_dims(bm,0)
                else:
                    yield np.expand_dims(dm,0), np.expand_dims(dgt,0), None
        else:
            for _ in xrange(slicewise_reps):
                coords =np.array([0,np.random.randint(0,dm.shape[0],1),np.random.randint(0,dm.shape[1],1),0])
                diff_add_x_up   = coords[1]+slice_size[0]/2-dm.shape[0]
                diff_add_x_down = coords[1]-slice_size[0]/2
                diff_add_y_up   = coords[2]+slice_size[1]/2-dm.shape[1]
                diff_add_y_down = coords[2]-slice_size[1]/2
                x_down    = np.clip(diff_add_x_down,0,None)-np.clip(diff_add_x_up,0,None)
                x_up      = np.clip(coords[1]+slice_size[0]/2,0,dm.shape[0])-np.clip(diff_add_x_down,None,0)
                y_down    = np.clip(diff_add_y_down,0,None)-np.clip(diff_add_y_up,0,None)
                y_up      = np.clip(coords[2]+slice_size[1]/2,0,dm.shape[1])-np.clip(diff_add_y_down,None,0)

                if use_weights:
                    yield np.expand_dims(((dm-mu)/sd)[x_down:x_up,y_down:y_up,:],0),np.expand_dims(dgt[x_down:x_up,y_down:y_up,0:1],0),np.expand_dims(bm[x_down:x_up,y_down:y_up],0)
                else:
                    yield np.expand_dims(((dm-mu)/sd)[x_down:x_up,y_down:y_up,:],0),np.expand_dims(dgt[x_down:x_up,y_down:y_up,0:1],0), None

                if augment:
                    for i in xrange(augment):
                        if use_weights:
                            image, mask, wmap = transform_image(dm/255., dgt, weight_map = bm, seed=seed+i, trans_types = ["flip", "zoom", "contrast", "brightness", "saturation", "color_shift"])
                            yield np.expand_dims(((image*255-mu)/sd)[x_down:x_up,y_down:y_up,:],0), np.expand_dims(mask[x_down:x_up,y_down:y_up,0:1],0), np.expand_dims(wmap[x_down:x_up,y_down:y_up,0:1],0)
                        else:
                            image, mask, _ = transform_image(dm/255., dgt, weight_map = None, seed=seed+i, trans_types = ["flip", "zoom", "contrast", "brightness", "saturation", "color_shift"])
                            yield np.expand_dims(((image*255-mu)/sd)[x_down:x_up,y_down:y_up,:],0), np.expand_dims(mask[x_down:x_up,y_down:y_up,0:1],0), None



### Mini-Generator to yield test images. Obsolete.
def test_image_provision(test_data):
    for filename in test_data:
        yield np.expand_dims((cv2.imread(filename)-mu)/sd,0)



### Convert Image (wxh) to run-length encoding
def convert_to_run_length_encoding(img):
    pxl_list = img.flatten()
    pxl_list[0] = 0
    pxl_list[-1]= 0

    rle_list = np.where(pxl_list[1:] != pxl_list[:-1])[0]+2
    rle_list[1::2] = rle_list[1::2]-rle_list[:-1:2]

    return " ".join(str(rle_val) for rle_val in rle_list)
