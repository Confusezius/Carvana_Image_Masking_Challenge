## Carvana Image Masking Challenge

This repository contains the code necessary to reproduce/generate a top-5% segmentation pipeline for the Carvance Image Masking Challenge. 
The data required is available through [Kaggle](https://www.kaggle.com/c/carvana-image-masking-challenge).

---

The code in this repository is build around tensorflow 1.3 and Keras 2.1. For usage of recent releases, the code needs to be updated respectively.
All other packages used are available within standard Anaconda.

---
This repository contains the following files:

1. `network_training.py`: on the left side.
2. `network_library.py`: Click the README.md link from the list of files.
3. `helper_functions.py`: Click the **Edit** button.
4. `create_bordermaps.py`: Delete the following text: *Delete this line to make a change to the README from Bitbucket.*
5. `validate_test.py`: After making your change, click **Commit** and then **Commit** again in the dialog. The commit page will open and you’ll see the change you just made.

The actual pipeline has the following structure:   
`create_bordermaps.py` to generate weightmaps to place more weights on accurate boundary segmentation `>>`   
`network_training.py` to train the network of choice with boundary masks `>>`   
`validate_test.py` to run the trained network on the test dataset.

---

Each of the elementary pipeline files can be exemplary run with the following input arguments:

_Create Boundary Mask_: 

```
python create_bordermaps.py --path_to_training_masks <path-to-ground-truth-target-mask> --save_path <location-of-choice-to-save-computed-masks>
```

_Train Network_: 

```
python network_training.py --path_to_input_images <path-to-training-images> --path_to_ground_truth_masks <path-to-target-segmentation-masks> --path_to_boundary_masks <path-to-precomputed-boundary-masks>
```

_Generate Test Results_: 
Please edit the paths to the test data and the choice of network weights withing the file, then run ```python validate_test.py```.