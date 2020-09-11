#model path
DEFAULT_MODEL_PATH = 'model_data/yolo.h5'
#anchor path
DEFAULT_ANCHORS_PATH = 'model_data/yolo_anchors.txt' 
#class name 
DEFAULT_CLASSES_PATH = 'model_data/coco_classes.txt'
#Class Score
SCORE = 0.3
#IOU VALUE
IOU = 0.45
#Image size
MODEL_IMAGE_SIZE = (416,416)
#GPU
GPU_NUM = 1
#Train data path
TRAIN_DATA_PATH = 'train.txt'
#Validation Split
VALID_SPLIT = 0.1
#Image Argumentation
IMAGE_ARGMENTATION = True
#Log Directory
LOGS_DIR = 'logs/000/'
#Pretrained Model Weight
PRETRAINED_WEIGHT = 'model_data/yolo_weights.h5'
#Frozen Training
FROZEN_TRAIN = False
#Forzen layer learning rate
FROZEN_TRAIN_LR = 1e-3
#FROZEN BATCH SIZE
FROZEN_TRAIN_BATCH_SIZE = 32
#Frozen output weights
FROZEN_TRAIN_OUTPUT_WEIGHTS = 'trained_weights_stage_1.h5'

UNFREEZE_TRAIN = True
# Unfreze training learning rate
UNFREEZE_TRAIN_LR = 1e-4
# Unfreeze training batch size
UNFREEZE_TRAIN_BATCH_SIZE = 1

UNFREEZE_TRAIN_OUTPUT_WEIGHTS = 'trained_weights_stage_2.h5'
# Final output weights
FINAL_OUTPUT_WEIGHTS = 'trained_weights_final.h5'