### Annotation of Polyps using images and ground truth masks
### Dataset from https://osf.io/mh9sj/
### Using Images and masks from the segmented_images folder

# pip install git+https://github.com/divamgupta/image-segmentation-keras
# pip install keras-segmentation

import os
from PIL import Image 
import matplotlib.pyplot as plt
import keras_segmentation
import numpy as np
from keras_segmentation.models.unet import vgg_unet
from keras_segmentation.models.unet import resnet50_unet
from imgaug import augmenters as iaa
import pickle

train_image_path = "E:\Data\segmented-images\png_images"
train_mask_path =  "E:\Data\segmented-images\png_masks"
checkpoint_path =  "E:\Data\Annotated Images\Polyps\Checkpoints"

model = resnet50_unet(n_classes=256)

print("[INFO] training segmentation mask regressor...")
model.train(
    train_images = train_image_path,
    train_annotations = train_mask_path,
    checkpoints_path = checkpoint_path,
    epochs=5
    )


print("[INFO] saving segmentation mask model...")
model.save('E:\Code\Final_SNN\segmentation_mask_augmentations\output\segmentation_model.h5', save_format="h5")

# path = "E:\Data\segmented-images\png_images"
# output = "E:\Data\Annotated Images\Polyps\Images"

# for i in os.listdir(path):
#   out = model.predict_segmentation(
#     inp= path + "/" + i,
#     out_fname= output + "/" + i,
#     overlay_img=True
#   )