import tensorflow as tf
import numpy as np
import os
import random
import config

from snn import createSNN
from utilities import preprocess_triplets, preprocess_image
from training_and_testing import trainSnn

### HYPERPARAMETERS
epoch_num = 10
batch_sizes = config.batch_sizes
training_testing_split = 0.8
num_of_tests = 10

### Setting it up
unaltered_accuracy_totals = []
meta_accuracy_totals = []
bb_accuracy_totals = []
seg_mask_accuracy_totals = []
all_filters_accuracy_totals = []

def getPrediction(model, unknown, positive, negative):

    encodings = model.predict((unknown, positive, negative))
    unknown = tf.convert_to_tensor(encodings[0])
    positive = tf.convert_to_tensor(encodings[1]) 
    negative = tf.convert_to_tensor(encodings[2])
        
    cosine_similarity = tf.metrics.CosineSimilarity()

    positive_similarity = cosine_similarity(unknown, positive)
    print("Positive similarity:", positive_similarity.numpy())

    cosine_similarity = tf.metrics.CosineSimilarity()

    negative_similarity = cosine_similarity(unknown, negative)
    print("Negative similarity", negative_similarity.numpy())

    if (positive_similarity.numpy() > negative_similarity.numpy()):
        print("Correct prediction")
        return True
    else:
        print("Wrong prediction")
        return False

def OLDgetTrainingAndTestingDatasets(batch_size, training_testing_split, findings_folder_path, normal_folder_path):
    ulcer_filenames = []
    polyp_filenames = []
    erosion_filenames = []
    erythema_filenames = []
    normal_filenames = []
    all_findings_filenames = []

    #polyps
    for filename in  os.listdir(config.POLYPS_UNALTERED_PATH):
        f = os.path.join(config.POLYPS_UNALTERED_PATH, filename)
        polyp_filenames.append(f)
        all_findings_filenames.append(f)
    #ulcers
    for filename in  os.listdir(config.ULCER_UNALTERED_PATH):
        f = os.path.join(config.ULCER_UNALTERED_PATH, filename)
        ulcer_filenames.append(f)
        all_findings_filenames.append(f)
    #erythemas
    for filename in  os.listdir(config.ERYTHEMAS_UNALTERED_PATH):
        f = os.path.join(config.ERYTHEMAS_UNALTERED_PATH, filename)
        erythema_filenames.append(f)
        all_findings_filenames.append(f)
    #erosions
    for filename in  os.listdir(config.EROSIONS_UNALTERED_PATH):
        f = os.path.join(config.EROSIONS_UNALTERED_PATH, filename)
        erosion_filenames.append(f)
        all_findings_filenames.append(f)
    #normal
    for filename in  os.listdir(normal_folder_path):
        f = os.path.join(normal_folder_path, filename)
        normal_filenames.append(f)

    findings_size = np.shape(all_findings_filenames)
    ulcer_size = np.shape(ulcer_filenames)
    polyp_size = np.shape(polyp_filenames)
    erosion_size = np.shape(erosion_filenames)
    erythema_size = np.shape(erythema_filenames)
    normal_size = np.shape(normal_filenames)

    print("\nFindings image size: " + str(np.shape(all_findings_filenames)))
    print("Normal image size: " + str(np.shape(normal_filenames)) + "\n")

    train_data = []
    test_data = []
    print("Compiling training data...")
    for x in range(batch_size):
        if (x%2 == 0):
            anchor_file = all_findings_filenames[random.randrange(int(findings_size[0]))]
            positive_file = all_findings_filenames[random.randrange(int(findings_size[0]))]
            negative_file = normal_filenames[random.randrange(int(normal_size[0]))]
        else:
            anchor_file = normal_filenames[random.randrange(int(normal_size[0]))]
            positive_file = normal_filenames[random.randrange(int(normal_size[0]))]
            negative_file = all_findings_filenames[random.randrange(int(findings_size[0]))]

        triplet = preprocess_triplets(anchor_file, positive_file, negative_file)
        train_data.append(triplet)


    print("Compiling normal test image data...")
    normal_images = []
    for i in range(200):
        normal_images.append(preprocess_image(normal_filenames[random.randrange(int(normal_size[0]))]))
    test_data.append(normal_images)

    print("Compiling ulcer test image data...")
    ulcer_images  = []
    for file in ulcer_filenames:
        ulcer_images.append(preprocess_image(file))
    test_data.append(ulcer_images)

    print("Compiling polyp test image data...")
    polyp_images  = []
    for file in polyp_filenames:
        polyp_images.append(preprocess_image(file))
    test_data.append(polyp_images)

    print("Compiling erosion test image data...")
    erosion_images  = []
    for file in erosion_filenames:
        erosion_images.append(preprocess_image(file))
    test_data.append(erosion_images)

    print("Compiling erythema test image data...")
    erythema_images  = []
    for file in erythema_filenames:
        erythema_images.append(preprocess_image(file))
    test_data.append(erythema_images)


    print("Training data size: ", len(train_data))
    print("Testing data size: ", len(test_data))

    # visualize(train_data[0])

    return train_data, test_data

def OLDtestSnn(model, test_data):
    correct_preds = 0
    normal_preds =0 
    ulcer_preds = 0
    polyp_preds = 0
    erosion_preds = 0
    erythema_preds = 0

    normal_images = test_data[0]
    ulcer_images = test_data[1]
    polyp_images =test_data[2]
    erosion_images = test_data[3]
    erythema_images = test_data[4]

    # One shot verification 50 times
    normal_test_image = normal_images[random.randrange(np.shape(normal_images)[0])]
    ulcer_test_image = ulcer_images[random.randrange(np.shape(ulcer_images)[0])]
    polyp_test_image = polyp_images[random.randrange(np.shape(polyp_images)[0])]
    erosion_test_image = erosion_images[random.randrange(np.shape(erosion_images)[0])]
    erythema_test_image = erythema_images[random.randrange(np.shape(erythema_images)[0])]

    # Test for normal detection
    print("Testing normal image detecction...")
    for i in range(80):    
        unknown_image = normal_images[random.randrange(np.shape(normal_images)[0])]
        if(getPrediction(model, unknown_image, normal_test_image, ulcer_test_image)):
            if(getPrediction(model, unknown_image, normal_test_image, polyp_test_image)):
                if(getPrediction(model, unknown_image, normal_test_image, erosion_test_image)):
                    if(getPrediction(model, unknown_image, normal_test_image, erythema_test_image)):
                        correct_preds = correct_preds + 1
                        normal_preds = normal_preds + 1

    # Test for ulcer detection
    print("Testing ulcer image detecction...")
    for i in range(80):    
        unknown_image = ulcer_images[random.randrange(np.shape(ulcer_images)[0])]
        if(getPrediction(model, unknown_image, ulcer_test_image, normal_test_image)):
            if(getPrediction(model, unknown_image, ulcer_test_image, polyp_test_image)):
                if(getPrediction(model, unknown_image, ulcer_test_image, erosion_test_image)):
                    if(getPrediction(model, unknown_image, ulcer_test_image, erythema_test_image)):
                        correct_preds = correct_preds + 1
                        ulcer_preds = ulcer_preds + 1

    # Test for Polyp detection
    print("Testing polyp image detecction...")
    for i in range(80):    
        unknown_image = polyp_images[random.randrange(np.shape(polyp_images)[0])]
        if(getPrediction(model, unknown_image, polyp_test_image, normal_test_image)):
            if(getPrediction(model, unknown_image, polyp_test_image, ulcer_test_image)):
                if(getPrediction(model, unknown_image, polyp_test_image, erosion_test_image)):
                    if(getPrediction(model, unknown_image, polyp_test_image, erythema_test_image)):
                        correct_preds = correct_preds + 1
                        polyp_preds = polyp_preds + 1

    # Test for erosion detection
    print("Testing erosion image detecction...")
    for i in range(80):    
        unknown_image = erosion_images[random.randrange(np.shape(erosion_images)[0])]
        if(getPrediction(model, unknown_image, erosion_test_image, normal_test_image)):
            if(getPrediction(model, unknown_image, erosion_test_image, polyp_test_image)):
                if(getPrediction(model, unknown_image, erosion_test_image, ulcer_test_image)):
                    if(getPrediction(model, unknown_image, erosion_test_image, erythema_test_image)):
                        correct_preds = correct_preds + 1
                        erosion_preds = erosion_preds + 1

    # Test for erythema detection
    print("Testing erythema image detecction...")
    for i in range(80):    
        unknown_image = erythema_images[random.randrange(np.shape(erythema_images)[0])]
        if(getPrediction(model, unknown_image, erythema_test_image, normal_test_image)):
            if(getPrediction(model, unknown_image, erythema_test_image, polyp_test_image)):
                if(getPrediction(model, unknown_image, erythema_test_image, ulcer_test_image)):
                    if(getPrediction(model, unknown_image, erythema_test_image, erosion_test_image)):
                        correct_preds = correct_preds + 1
                        erythema_preds = erythema_preds + 1

    return([correct_preds, normal_preds, ulcer_preds, polyp_preds, erosion_preds, erythema_preds])


### Training adn Testing
train_data, test_data = OLDgetTrainingAndTestingDatasets(1000, training_testing_split, '', config.NOFINDINGS_UNALTERED_PATH)
model = createSNN('')

for i in range(1):
    model = trainSnn(model, 1, train_data)
    results = OLDtestSnn(model, test_data)
    model.save(os.path.sep.join([config.ULCER_TRAINED_NETWORK_PATH, "Ulcers_unaltered_Model_test_"+str(i+1)+".h5"]), save_format="h5")
print(results)
print("Results:")
print("Overall accuracy:", (results[0]/400)*100)
print("Normal accuracy:", (results[1]/80)*100)
print("Ulcer accuracy:", (results[2]/80)*100)
print("Polyp accuracy:", (results[3]/80)*100)
print("Erosion accuracy:", (results[4]/80)*100)
print("Erythema accuracy:", (results[5]/80)*100)



