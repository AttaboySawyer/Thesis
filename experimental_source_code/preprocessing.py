import os
import random
import numpy as np
from utilities import preprocess_triplets, visualize, preprocess_image
import config

random.seed(42)

def getTrainingAndTestingDatasets(batch_size, training_testing_split, findings_folder_path, normal_folder_path):
    findings_filenames = []
    normal_filenames = []
    
    for filename in  os.listdir(findings_folder_path):
        f = os.path.join(findings_folder_path, filename)
        findings_filenames.append(f)

    for filename in  os.listdir(normal_folder_path):
        f = os.path.join(normal_folder_path, filename)
        normal_filenames.append(f)

    findings_size = np.shape(findings_filenames)
    normal_size = np.shape(normal_filenames)

    print("\nFindings image size: " + str(np.shape(findings_filenames)))
    print("Normal image size: " + str(np.shape(normal_filenames)) + "\n")

    train_data = []
    test_data = []

    for x in range(batch_size):
        # Every second triplet, the anchor is normal
        # if(x < batch_size*training_testing_split):
        if (x%2 == 0):
            anchor_file = findings_filenames[random.randrange(int(findings_size[0]))]
            positive_file = findings_filenames[random.randrange(int(findings_size[0]))]
            negative_file = normal_filenames[random.randrange(int(normal_size[0]))]
        else:
            anchor_file = normal_filenames[random.randrange(int(normal_size[0]))]
            positive_file = normal_filenames[random.randrange(int(normal_size[0]))]
            negative_file = findings_filenames[random.randrange(int(findings_size[0]))]

        triplet = preprocess_triplets(anchor_file, positive_file, negative_file)
        train_data.append(triplet)
        # else:
    for x in range(200):
        if (x%2 == 0):
            anchor_file = findings_filenames[random.randrange(int(findings_size[0]))]
            positive_file = findings_filenames[random.randrange(int(findings_size[0]))]
            negative_file = normal_filenames[random.randrange(int(normal_size[0]))]
        else:
            anchor_file = normal_filenames[random.randrange(int(normal_size[0]))]
            positive_file = normal_filenames[random.randrange(int(normal_size[0]))]
            negative_file = findings_filenames[random.randrange(int(findings_size[0]))]

        triplet = preprocess_triplets(anchor_file, positive_file, negative_file)
        test_data.append(triplet)


    print("Training data size: ", len(train_data))
    print("Testing data size: ", len(test_data))

    # visualize(train_data[0])

    return train_data, test_data

