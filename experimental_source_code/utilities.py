import tensorflow as tf
import numpy as np
import PIL
import matplotlib.pyplot as plt
import config
from imgaug import augmenters as iaa 
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from keras_segmentation.predict import model_from_checkpoint_path
import random
from matplotlib import image as im
from PIL import Image, ImageDraw, ImageEnhance
# from base_model import preprocessing

bounding_box_augmentor_path = config.bounding_box_augmentor_path
seg_mask_augmentor_path = config.seg_mask_augmentor_path

bb_temp_path = "E:/Code/Final_SNN/bb_temp.jpg"
seg_mask_temp_path = "E:/Code/Final_SNN/temp2.jpg"

bb_model = load_model(bounding_box_augmentor_path)
seg_model = model_from_checkpoint_path(seg_mask_augmentor_path)

train_with_meta_filters = config.train_with_meta_filters
train_with_bounding_boxes = config.train_with_bounding_boxes
train_with_segmentation_masks = config.train_with_segmentation_masks

def preprocess_image(filename):
    """
    Load the specified file as a JPEG image, preprocess it and
    resize it to the target shape.
    """
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (224,224))
    image = image.numpy()
    # print(image.numpy().shape)
    # image = tf.expand_dims(image, 0)

    if (train_with_bounding_boxes):

        # image = image / 255.0
        image = np.expand_dims(image, axis=0)

        # print("Image before bb")
        # plt.imshow(image[0]) 
        # plt.show()

        preds = bb_model.predict(image)[0]
        (startX, startY, endX, endY) = preds
        # scale the predicted bounding box coordinates based on the image
        # dimensions
        startX = int(startX * 224)
        startY = int(startY * 224)
        endX = int(endX * 224)
        endY = int(endY * 224)
       
        # image = tensor_to_image(image)

        # draw = ImageDraw.Draw(image) 
        # draw.rectangle([startX, startY, endX, endY], outline ="green")
        # print("Image after bb")
        # plt.imshow(image[0]) 
        # plt.show()

        image = img_to_array(image[0])

        # return image

    if (train_with_segmentation_masks):

        image = np.expand_dims(image, axis=0)
        image = array_to_img(image[0])
        # print("Image before seg mask")
        # plt.imshow(image) 
        # plt.show()
        image = img_to_array(image)

        out = seg_model.predict_segmentation(
        # inp= path + "/" + i,
        # checkpoints_path =checkpoint_path,
        inp=image,
        out_fname= seg_mask_temp_path,
        overlay_img=True
        )

        image = im.imread(seg_mask_temp_path)

        # print("Image after seg mask")
        # plt.imshow(image) 
        # plt.show()

    if (train_with_bounding_boxes):
        image = array_to_img(image)

        draw = ImageDraw.Draw(image)
        draw.rectangle([startX, startY, endX, endY], outline ="green")
        # plt.imshow(image) 
        # plt.show()

        image = img_to_array(image)
    
    if(train_with_meta_filters):
        
        # print("Meta filter image before")
        image = array_to_img(image)
        # plt.imshow(image) 
        # plt.show()

        enhancer = ImageEnhance.Contrast(image)
        # rand = random.randint(0,9)
        # if (rand % 2 == 0):
        image = enhancer.enhance(3)
        # rand = random.randint(0,9)
        # if (rand % 2 == 0):
        #     image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        # rand = random.randint(0,9)
        # if (rand % 2 == 0):
        frac = 0.9
        left = image.size[0]*((1-frac)/2)
        upper = image.size[1]*((1-frac)/2)
        right = image.size[0]-((1-frac)/2)*image.size[0]
        bottom = image.size[1]-((1-frac)/2)*image.size[1]
        image = image.crop((left, upper, right, bottom))
        image = image.resize((224,224))
    
        # image = img_to_array(image)

        print("Meta filter image after:")
        # plt.imshow(image) 
        # plt.show()
        # print(image)
        image = img_to_array(image)
        # image = tf.io.decode_image(image, channels=3, dtype=tf.dtypes.float32)
        # image = tf.convert_to_tensor(image)

    
    image = tf.expand_dims(image, 0)
    return image#.numpy()

def preprocess_triplets(anchor, positive, negative):
    """
    Given the filenames corresponding to the three images, load and
    preprocess them.
    """

    return (
        preprocess_image(anchor),
        preprocess_image(positive),
        preprocess_image(negative)
    )

def tensor_to_image(tensor):
    # tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def visualize(triplet):
    anchor = triplet[0]
    positive = triplet[1]
    negative = triplet[2]
    plt.imshow(tensor_to_image(anchor))
    plt.show()
    plt.imshow(tensor_to_image(positive))
    plt.show()
    plt.imshow(tensor_to_image(negative))
    plt.show()