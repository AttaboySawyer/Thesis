import cv2
import os
import PIL

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import random as rand

from tensorflow.keras.applications.resnet50 import ResNet50

import sklearn
import torch.nn as nn
import torch
from torch import optim
from tensorflow.keras import metrics
import tensorflow as tf
import torchvision
import torch.nn.functional as F
from scipy.spatial.distance import cityblock

from tensorflow.keras.preprocessing import image
import torchvision.models as models
from random import randint

from torchvision import transforms
import imgaug.augmenters as iaa

from PIL import Image
from PIL import ImageFilter

from tqdm.notebook import tqdm, trange



np.random.seed(42)
tf.random.set_seed(42)
seed = 42
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

from flask import Flask
from flask_cors import CORS
from flask import request
from flask import jsonify
from base64 import encodebytes
from io import BytesIO
import base64
import io

def lossless_triplet_loss(y_true, y_pred, N = 3, beta=3, epsilon=1e-8, margin=0.2):

    anchor = tf.convert_to_tensor(y_pred[:,0])
    positive = tf.convert_to_tensor(y_pred[:,1]) 
    negative = tf.convert_to_tensor(y_pred[:,2])
    
    # distance between the anchor and the positive
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)),-1)
    # distance between the anchor and the negative
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)),-1)
    
    #Non Linear Values  
    
    # -ln(-x/N+1)
    # pos_dist = -tf.math.log(-tf.divide((pos_dist),beta)+1+epsilon)
    # neg_dist = -tf.math.log(-tf.divide((N-neg_dist),beta)+1+epsilon)
    
    # compute loss

    loss = pos_dist - neg_dist + margin
    loss = tf.maximum(loss,0.0)
    
    return loss



app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route("/extractframes", methods=['GET','POST'])
def hello_world():
    if request.method == 'GET':
        return "THIS IS WORKING!"
    if request.method == 'POST':
        all_frames_imgs = []
        all_frames_ids = []
        shape = (224, 224)

        mp4 = request.files['myFile']
        mp4.save("./temp.mp4")
        vidcap = cv2.VideoCapture("./temp.mp4")
        success,img = vidcap.read()
        
        count = 1
        while success:
            img = image.smart_resize(img, shape)
            img = np.asarray(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            all_frames_imgs.append(img)
            all_frames_ids.append("frame " + str(count))     
            success,img = vidcap.read()
            count += 1

        
        # Load the SNN
        model = tf.keras.models.load_model("E:/Data/TrainingAndTesting/Ulcers/OutputNetworks/Ulcers_unaltered_Model_test_20.h5",
                custom_objects={'lossless_triplet_loss':lossless_triplet_loss})
        # print(model.summary())

        cluster_collection = []
        cluster_array = []

        counter = 0

        print("filtering images...")

        before = all_frames_imgs[0] 
        before = np.expand_dims(before, 0)
                
        after = all_frames_imgs[1]
        after = np.expand_dims(after, 0)

        # Add first frame to first cluster
        cluster_array.append([before, all_frames_ids[counter], []])
        for compare_image in tqdm(all_frames_imgs):
            if(counter>0 and counter < len(all_frames_imgs)-4): # Skip the first frame
                cosine_similarity = metrics.CosineSimilarity()
                
                compare_image = np.expand_dims(compare_image, 0)
                
                output1, output2, output3 = model.predict((compare_image, before, after))
                
                similarity_before = cosine_similarity(output1, output2)
                similarity_after = cosine_similarity(output1, output3)
                
                # Uncomment this line to print the similarity between the two images
                # print('sim before: ', similarity_before.numpy(), 'sim after: ', similarity_after.numpy())
                # print(similarity_after.numpy() >= similarity_before.numpy())

                if (similarity_after.numpy() >= similarity_before.numpy()):
                    # [imageStr, idStr, anomolyDetectedList]
                    cluster_array.append([compare_image, all_frames_ids[counter], []])
                    cluster_collection.append([0,cluster_array])
                    cluster_array=[]
                    before = compare_image
                    after = all_frames_imgs[counter + 2]
                    after = np.expand_dims(after, 0)
                else:
                    cluster_array.append([compare_image, all_frames_ids[counter], []])
                    before = compare_image
                    after = all_frames_imgs[counter + 2]
                    after = np.expand_dims(after, 0)
            counter = counter + 1

        count = 0
        for cluster in cluster_collection:
            cluster_shape = np.shape(cluster[1])
            print('Cluster shape: ', cluster_shape)
            print(cluster_shape[0] < 2)
            if (cluster_shape[0] < 2):
                cluster_collection.pop(count)
            count = count + 1

        print("All images clustered")
        print("cluster collection: " + str(np.shape(cluster_collection)))


        ### Looking for signs of Crohn's ###
        normalImg = image.load_img("E:/Data/Crohn's Images/Normal.jpg", color_mode="rgb", target_size=shape)
        normalImg = image.img_to_array(normalImg)
        normalImg = np.expand_dims(normalImg, 0)

        ulcerImg = image.load_img("E:/Data/Crohn's Images/Ulcer.jpg", color_mode="rgb", target_size=shape)
        ulcerImg = image.img_to_array(ulcerImg)
        ulcerImg = np.expand_dims(ulcerImg, 0)

        polypImg = image.load_img("E:/Data/Crohn's Images/Polyp.jpg", color_mode="rgb", target_size=shape)
        polypImg = image.img_to_array(polypImg)
        polypImg = np.expand_dims(polypImg, 0)

        erosionImg = image.load_img("E:/Data/Crohn's Images/Erosion.jpg", color_mode="rgb", target_size=shape)
        erosionImg = image.img_to_array(erosionImg)
        erosionImg = np.expand_dims(erosionImg, 0)

        erythemaImg = image.load_img("E:/Data/Crohn's Images/Erythema.jpg", color_mode="rgb", target_size=shape)
        erythemaImg = image.img_to_array(erythemaImg)
        erythemaImg = np.expand_dims(erythemaImg, 0)

        anomolies_detected = 0

        image_counter = 1
        cluster_counter = 1
        anomoly_counter = 1
        found_anomoly = False
        anomoly_clusters = []
        for image_cluster in cluster_collection:
            images = image_cluster[1]
            for img in images:
                cosine_similarity = metrics.CosineSimilarity(axis=1)

                compare_image = img[0]

                #Looking for ulcers
                output1, output2, output3 = model.predict((compare_image, normalImg, ulcerImg))
                similarity_normal = cosine_similarity(output1, output2)
                similarity_ulcer = cosine_similarity(output1, output3)
                if (similarity_normal.numpy() < similarity_ulcer.numpy()):
                    found_anomoly = True
                    img[2].append('ulcer')
                    anomolies_detected = anomolies_detected + 1
                    # print("Ulcer detected at cluster " + str(cluster_counter) + ", frame " + str(image_counter))

                #Looking for polyps
                output1, output2, output3 = model.predict((compare_image, normalImg, polypImg))
                similarity_normal = cosine_similarity(output1, output2)
                similarity_polyp = cosine_similarity(output1, output3)
                if (similarity_normal.numpy() < similarity_polyp.numpy()):
                    found_anomoly = True 
                    img[2].append('polyp')
                    anomolies_detected = anomolies_detected + 1
                    # print("Polyp detected at cluster " + str(cluster_counter) + ", frame " + str(image_counter))
                
                #Looking for erosions
                output1, output2, output3 = model.predict((compare_image, normalImg, erosionImg))
                similarity_normal = cosine_similarity(output1, output2)
                similarity_erosion = cosine_similarity(output1, output3)
                if (similarity_normal.numpy() < similarity_erosion.numpy()):
                    found_anomoly = True
                    img[2].append('erosion')
                    anomolies_detected = anomolies_detected + 1
                    # print("Erosion detected at cluster " + str(cluster_counter) + ", frame " + str(image_counter))
                
                #Looking for erythema
                output1, output2, output3 = model.predict((compare_image, normalImg, erythemaImg))
                similarity_normal = cosine_similarity(output1, output2)
                similarity_erythema = cosine_similarity(output1, output3)
                if (similarity_normal.numpy() < similarity_erythema.numpy()):
                    found_anomoly = True
                    img[2].append('erythema')
                    anomolies_detected = anomolies_detected + 1
                    # print("Erythema detected at cluster " + str(cluster_counter) + ", frame " + str(image_counter))

                
                image_counter = image_counter + 1

            if(found_anomoly == True):
                anomoly_clusters.append(image_cluster)
            found_anomoly = False
            image_counter = 1                                                                         
            cluster_counter = cluster_counter + 1
        cluster_counter = 1
        anomoly_counter = anomoly_counter + 1
                
        # print("anomolies detected: " + str(anomolies_detected))
        print("Image clusters with anomolies: " + str((np.shape(anomoly_clusters))))

        backToImg = []
        for cluster in anomoly_clusters:
            tempArray = []
            images = cluster[1]
            for img in images: 
                i = image.array_to_img(img[0][0])
                buffered = BytesIO()
                i.save(buffered, format='PNG')
                img_str = encodebytes(buffered.getvalue()).decode('utf-8')
                tempArray.append([img_str, img[1], img[2]])
            backToImg.append(tempArray)
        print(np.shape(backToImg))
        return jsonify({'result': backToImg})