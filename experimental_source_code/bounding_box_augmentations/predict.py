import os
from tensorflow.keras.models import load_model
from keras_segmentation.predict import model_from_checkpoint_path
import numpy as np
import matplotlib.pyplot as plt
from keras_segmentation.models.unet import resnet50_unet
from matplotlib import image
from PIL import Image
from segmentation_mask_overlay import overlay_masks

path = "E:/Data/Base Network Training/Unaltered/Polyps/131368cc17e44240_28956.jpg"
output = "E:/Data/Base Network Training/Segmentation Masks/Polyps"
checkpoint_path = "E:/Data/Annotated Images/Polyps/Checkpoints"
# print(os.path.isfile('E:/Data/Annotated Images/Polyps/Checkpoints_config.json'))
temp_path = "E:/Code/Final_SNN/temp.jpg"


print("[INFO] loading object detector...")
# model = load_model('E:\Code\Final_SNN\segmentation_mask_augmentations\output\segmentation_model.h5')
model = model_from_checkpoint_path(checkpoint_path)
# model = resnet50_unet(n_classes=256)

ann_img = np.zeros((30,30,3)).astype('uint8')
ann_img[ 3 , 4 ] = 1

img = image.imread(path)
print(np.shape(img))

# for i in os.listdir(path):
out = model.predict_segmentation(
    # inp= path + "/" + i,
  # checkpoints_path =checkpoint_path,
  inp=img,
  out_fname= temp_path,
  overlay_img=True
)
# print(np.shape(out))
# out = out.resize((224, 224))
# fig = overlay_masks(img, out)

# img = Image.fromarray(img)
# out = Image.fromarray(out)

# new_img = Image.blend(img, out, 0.5)
  
plt.imshow(image.imread(temp_path)) 
plt.show()