import cv2
import os
import sys
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn.config import Config
import mrcnn.model as modellib
from mrcnn.model import MaskRCNN
import uuid
import argparse
import skimage
import colorsys
import tensorflow as tf
import numpy as np
import shutil
import random
import argparse

class TestConfig(Config):
    NAME = "Deepfashion2"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 13
config = TestConfig()



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
ap.add_argument("-o", "--output", required=True)
args = vars(ap.parse_args())

input_image = args["input"]
output = args["output"]


model = modellib.MaskRCNN(mode="inference", config=config, model_dir='/content/drive/My Drive/')
model.load_weights('/content/drive/My Drive/mask_rcnn_deepfashion2_0100.h5', by_name=True)

class_names = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear', 'vest', 'sling', 
               'shorts', 'trousers', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress',
               'vest_dress', 'sling_dress','']

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors

colors = random_colors(len(class_names))
class_dict = {
    name: color for name, color in zip(class_names, colors)
}

def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    n_instances = boxes.shape[0]
    print("no of potholes in frame :",n_instances)
    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i in range(n_instances):
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        color = class_dict[label]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        random_name = str(uuid.uuid4())
        mask = masks[:, :, i]  
        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
        
    return image

frame = cv2.imread(input_image)
results = model.detect([frame], verbose=0)
r = results[0]
masked_image = display_instances(frame, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
random_name = str(uuid.uuid4())
cv2.imwrite(output + "/" + random_name + ".jpg", masked_image) 
