# detect kangaroos in photos with mask rcnn model
from os import listdir
import os
# detect animals in photos with mask rcnn model
from os import listdir
from xml.etree import ElementTree
from numpy import zeros, asarray, expand_dims,  mean 
from matplotlib import pyplot
from matplotlib.patches import Rectangle

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.model import mold_image
from mrcnn.utils import Dataset
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt 
from mrcnn.utils import extract_bboxes
from mrcnn.visualize import display_instances

IMAGE_DIR = os.path.join(ROOT_DIR, "kangaroo")
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class KangarooConfig(Config):
    # define the name of the configuration
    NAME = "kangaroo_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # number of training steps per epoch
    STEPS_PER_EPOCH = 131
 

class PredictionConfig(Config):
    # define the name of the configuration
    NAME = "kangaroo_cfg"
    # number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    def evaluate_model(dataset, model, cfg):
        APs = list()
        for image_id in dataset.image_ids:
            # load image, bounding boxes, and masks for the image id
            image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id, use_mini_mask=False)
            
            # convert pixel values (e.g., center)
            scaled_image = mold_image(image, cfg)
            
            # convert image into one sample
            sample = expand_dims(scaled_image, 0)
            
            # make prediction
            yhat = model.detect(sample, verbose=0)
            
            # extract results for the first sample
            r = yhat[0]
            
            # calculate statistics, including AP
            AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
            
            # store
            APs.append(AP)
        
        # calculate the mean AP across all images
        mAP = mean(APs)
        
        return mAP

# load the test dataset
train_set = KangarooDataset()
# train_set.load_dataset('kangaroo', is_train=True)
train_set.load_dataset(IMAGE_DIR, is_train=True)
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))
print('Test: %d' % len(test_set.image_ids))
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# evaluate model on training dataset
train_mAP = evaluate_model(train_set, model, cfg)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
test_mAP = evaluate_model(test_set, model, cfg)
print("Test mAP: %.3f" % test_mAP)