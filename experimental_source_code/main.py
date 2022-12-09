import tensorflow as tf
import numpy as np
import os
import random
import config

from snn import createSNN
from utilities import preprocess_triplets, visualize
from preprocessing import getTrainingAndTestingDatasets
from training_and_testing import trainSnn, testSnn

### HYPERPARAMETERS
epoch_num = 10
batch_sizes = config.batch_sizes
batch_size = 36
training_testing_split = 0.8
num_of_tests = 20

normal_folder_path = config.NOFINDINGS_UNALTERED_PATH
ulcer_folder_path = config.ULCER_UNALTERED_PATH
polyps_folder_path = config.POLYPS_UNALTERED_PATH
erosions_folder_path = config.EROSIONS_UNALTERED_PATH
erythemas_folder_path = config.ERYTHEMAS_UNALTERED_PATH

# output_fodler = config.ULCER_TRAINEDNETWORK_PATH

base_model_path = config.base_model_path

### Setting it up
unaltered_accuracy_totals = []
meta_accuracy_totals = []
bb_accuracy_totals = []
seg_mask_accuracy_totals = []
all_filters_accuracy_totals = []

f = open(config.RESULTS_PATH, "w")
f.write("RESULTS:\n")

### Unaltered testing
f.write("No Filters:\n")
history = []
### Ulcer detection
train_data, test_data = getTrainingAndTestingDatasets(batch_size, training_testing_split, ulcer_folder_path, normal_folder_path)
model = createSNN(base_model_path)
for i in range(num_of_tests):
    model = trainSnn(model, 1, train_data)
    accuracy = testSnn(model, test_data)
    model.save(os.path.sep.join([config.ULCER_TRAINED_NETWORK_PATH, "Ulcers_unaltered_Model_test_"+str(i+1)+".h5"]), save_format="h5")
    history.append(accuracy)
f.write("Ulcers detection test accuracies: " + ", ".join(str(a) for a in history) + "\n" )
f.write("Average accuracy: "+ str(sum(history)/len(history)) + "\n" )
### Polyps detection
history = []
train_data, test_data = getTrainingAndTestingDatasets(batch_size, training_testing_split, polyps_folder_path, normal_folder_path)
model = createSNN(base_model_path)
for i in range(num_of_tests):
    model = trainSnn(model, 1, train_data)
    accuracy = testSnn(model, test_data)
    model.save(os.path.sep.join([config.POLYPS_TRAINED_NETWORK_PATH, "Polyps_unaltered_Model_test_"+str(i+1)+".h5"]), save_format="h5")
    history.append(accuracy)
f.write("Polyps detection test accuracies: " + ", ".join(str(a) for a in history) + "\n" )
f.write("Average accuracy: "+ str(sum(history)/len(history)) + "\n" )
### Erosions detection
history = []
train_data, test_data = getTrainingAndTestingDatasets(batch_size, training_testing_split, erosions_folder_path, normal_folder_path)
model = createSNN(base_model_path)
for i in range(num_of_tests):
    model = trainSnn(model, 1, train_data)
    accuracy = testSnn(model, test_data)
    model.save(os.path.sep.join([config.EROSIONS_TRAINED_NETWORK_PATH, "Erosions_unaltered_Model_test_"+str(i+1)+".h5"]), save_format="h5")
    history.append(accuracy)
f.write("Erosions detection test accuracies: " + ", ".join(str(a) for a in history) + "\n" )
f.write("Average accuracy: "+ str(sum(history)/len(history)) + "\n" )
### Erythemas detection
history = []
train_data, test_data = getTrainingAndTestingDatasets(batch_size, training_testing_split, erythemas_folder_path, normal_folder_path)
model = createSNN(base_model_path)
for i in range(num_of_tests):
    model = trainSnn(model, 1, train_data)
    accuracy = testSnn(model, test_data)
    model.save(os.path.sep.join([config.ERYTHEMAS_TRAINED_NETWORK_PATH, "Erythemas_unaltered_Model_test_"+str(i+1)+".h5"]), save_format="h5")
    history.append(accuracy)
f.write("Erythemas detection test accuracies: " + ", ".join(str(a) for a in history) + "\n" )
f.write("Average accuracy: "+ str(sum(history)/len(history)) + "\n" )