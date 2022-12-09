import os

# Hyperparameters
train_with_meta_filters = False
train_with_bounding_boxes = False
train_with_segmentation_masks = False

# Paths
base_model_path = 'E:/Data/Base Network Saves/small_training_model_size_224_224_batch_10_BN_no_filters.h5'
bounding_box_augmentor_path = "E:/Code/Final_SNN/bounding_box_augmentations/output/erosions_erythemas_polyps_ulcers_detector.h5"
seg_mask_augmentor_path = "E:/Data/Annotated Images/Polyps/Checkpoints"


# Training and Testing parameters
batch_sizes = [36, 50, 100, 200]
res_layers = [0]

# Saving results
BASE_PATH = "E:/Code/Final_SNN"
RESULTS_PATH = os.path.sep.join([BASE_PATH, "results.txt"])

# We have 4 findings fodlers and 1 normal folder

BASE_ULCER_PATH = "E:/Data/TrainingAndTesting/Ulcers"
BASE_POLYPS_PATH = "E:/Data/TrainingAndTesting/Polyps"
BASE_EROSIONS_PATH = "E:/Data/TrainingAndTesting/Erosions"
BASE_ERYTHEMAS_PATH = "E:/Data/TrainingAndTesting/Erythemas"
BASE_NOFINDINGS_PATH = "E:/Data/TrainingAndTesting/NoFindings"

# Ulcer paths
ULCER_UNALTERED_PATH = os.path.sep.join([BASE_ULCER_PATH, "Unaltered"])
ULCER_METAFILTERS_PATH = os.path.sep.join([BASE_ULCER_PATH, "MetaFilters"])
ULCER_BOUNDINGBOXES_PATH = os.path.sep.join([BASE_ULCER_PATH, "BoundingBoxes"])
ULCER_SEGMENTATIONMASKS_PATH = os.path.sep.join([BASE_ULCER_PATH, "SegmentationMasks"])
ULCER_TRAINED_NETWORK_PATH = os.path.sep.join([BASE_ULCER_PATH, "OutputNetworks"])

# Polyps paths
POLYPS_UNALTERED_PATH = os.path.sep.join([BASE_POLYPS_PATH, "Unaltered"])
POLYPS_METAFILTERS_PATH = os.path.sep.join([BASE_POLYPS_PATH, "MetaFilters"])
POLYPS_BOUNDINGBOXES_PATH = os.path.sep.join([BASE_POLYPS_PATH, "BoundingBoxes"])
POLYPS_SEGMENTATIONMASKS_PATH = os.path.sep.join([BASE_POLYPS_PATH, "SegmentationMasks"])
POLYPS_TRAINED_NETWORK_PATH = os.path.sep.join([BASE_POLYPS_PATH, "OutputNetworks"])

# Erosions paths
EROSIONS_UNALTERED_PATH = os.path.sep.join([BASE_EROSIONS_PATH, "Unaltered"])
EROSIONS_METAFILTERS_PATH = os.path.sep.join([BASE_EROSIONS_PATH, "MetaFilters"])
EROSIONS_BOUNDINGBOXES_PATH = os.path.sep.join([BASE_EROSIONS_PATH, "BoundingBoxes"])
EROSIONS_SEGMENTATIONMASKS_PATH = os.path.sep.join([BASE_EROSIONS_PATH, "SegmentationMasks"])
EROSIONS_TRAINED_NETWORK_PATH = os.path.sep.join([BASE_EROSIONS_PATH, "OutputNetworks"])

# Erythemas paths
ERYTHEMAS_UNALTERED_PATH = os.path.sep.join([BASE_ERYTHEMAS_PATH, "Unaltered"])
ERYTHEMAS_METAFILTERS_PATH = os.path.sep.join([BASE_ERYTHEMAS_PATH, "MetaFilters"])
ERYTHEMAS_BOUNDINGBOXES_PATH = os.path.sep.join([BASE_ERYTHEMAS_PATH, "BoundingBoxes"])
ERYTHEMAS_SEGMENTATIONMASKS_PATH = os.path.sep.join([BASE_ERYTHEMAS_PATH, "SegmentationMasks"])
ERYTHEMAS_TRAINED_NETWORK_PATH = os.path.sep.join([BASE_ERYTHEMAS_PATH, "OutputNetworks"])

# No Findings paths
NOFINDINGS_UNALTERED_PATH = os.path.sep.join([BASE_NOFINDINGS_PATH, "Unaltered"])
NOFINDINGS_METAFILTERS_PATH = os.path.sep.join([BASE_NOFINDINGS_PATH, "MetaFilters"])
NOFINDINGS_BOUNDINGBOXES_PATH = os.path.sep.join([BASE_NOFINDINGS_PATH, "BoundingBoxes"])
NOFINDINGS_SEGMENTATIONMASKS_PATH = os.path.sep.join([BASE_NOFINDINGS_PATH, "SegmentationMasks"])
