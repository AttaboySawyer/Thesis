import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.preprocessing.image import load_img
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from keras_segmentation.predict import model_from_checkpoint_path
from utilities import tensor_to_image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as im
from PIL import Image, ImageDraw, ImageEnhance
import os
from torchvision import transforms
import torch
import random
import pickle

# from snn import ResidualBlock

bounding_box_augmentor_path = "E:/Code/Final_SNN/bounding_box_augmentations/output/erosions_erythemas_polyps_ulcers_detector.h5"
bb_temp_path = "E:/Code/Final_SNN/bb_temp.jpg"
seg_mask_augmentor_path = "E:/Data/Annotated Images/Polyps/Checkpoints"

seg_mask_temp_path = "E:/Code/Final_SNN/temp.jpg"

# Controls
train_with_meta_filters = False
train_with_bounding_boxes = False
train_with_segmentation_masks = False

bb_model = load_model(bounding_box_augmentor_path)
        # model = model_from_checkpoint_path(seg_mask_augmentor_path)

def preprocessing(image):
    # print("Before:")
    # print(np.shape(image))
    # plt.imshow(array_to_img(image)) 
    # plt.show()

    if (train_with_bounding_boxes):



        # bb_model = load_model(bounding_box_augmentor_path)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

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

        # plt.imshow(image[0]) 
        # plt.show()

        image = img_to_array(image[0])
        # return image

    if (train_with_segmentation_masks):


        out = model.predict_segmentation(
        # inp= path + "/" + i,
        # checkpoints_path =checkpoint_path,
        inp=image,
        out_fname= seg_mask_temp_path,
        overlay_img=True
        )

        image = im.imread(seg_mask_temp_path)

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
    
        image = img_to_array(image)

    # print("After:")
    # plt.imshow(array_to_img(image)) 
    # plt.show()
    return image



batch_size = 10

def create_base_network(in_dims):
    """
    Base network that takes the inital input and begins learning.
    """
    model = tf.keras.Sequential(name="base_model")
    model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=in_dims))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2,2))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(2,2))

    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(1,1))
    model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
    # model.add(tf.keras.layers.MaxPooling2D(2,2))
    # model.add(ResidualBlock(64).forward(input=()))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(5, activation='softmax'))
    

    return model

model = create_base_network((224,224,3))
print(model.summary())

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocessing)
train_generator = train_datagen.flow_from_directory(
    'E:/Data/Base Network Training/Unaltered',
    target_size=(224,224),
    batch_size=batch_size,
    classes=['Erosions','Erythemas','Normal','Polyps','Ulcers'],
    class_mode='categorical',
    )

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
    metrics=['accuracy'])

total_sample = train_generator.n
# print(train_generator[0])
n_epochs = 10
history = model.fit(
    train_generator,
    steps_per_epoch=int(total_sample/batch_size),
    epochs=n_epochs,
    verbose=1)

model.save('E:/Data/Base Network Saves/small_training_model_size_224_224_batch_10_test_2_no_filters.h5')
with open('E:/Data/Base Network Saves/history saves/small_training_model_size_224_224_batch_10_test_2_BN_no_filters', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)