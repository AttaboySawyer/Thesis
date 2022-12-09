## Source code for all networks, and experimentations

### Old network, training and testing - OLDTrainingAndTesting.py
During initla experimentation, I was working with a SNN with no base netowrk training. I found this cause the network to wokkr in extremes during testing, acheiving either 100% or 0% accuracies. As such, I moved to the base netowrk training method and the results were more consistant. I have included this intial code as it is mentioned in the written thesis.  

### Used network code
All other files, except for the bounding_box_augmentations and segmentation_mask_augmentations folders, pertain to the base network and SNN used in the written thesis, example, and prototype. I have tried to bring all pathing to datasets to the config file so that it can be more easily ran on all machines. Since the datassets are too large to be stored on Github, you must download them from the provided links and change the paths in the config file to the locations on you machine. As well, the main file only runs one set of tests with given preprocessing augmentations. All tests are saved to the results.txt file, and if you want other results using other augmentations, you must save the results in a seperate file as it will be overwritten, Change the appropriate preprocessing flags in the config file to True, and run main again,

### Bounding Box Augmentations
This is the code used to train the network that applies bounding boxes on images during preprocessing. You don;t need to run this with the main file as the resulting network of this code is saved and loaded, Not re-generated everytime.

### Segmentation Mask Augmentations
Same as bounding boxes, this is the code used to train the segmetnation mask network. It does not need to be ran with the main file as the resulting netowrk is saved and loaded.
