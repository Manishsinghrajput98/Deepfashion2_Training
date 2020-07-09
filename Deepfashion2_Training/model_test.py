from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import matplotlib.pyplot as plt
from mrcnn.visualize import * 
from mrcnn.visualize import display_instances
from mrcnn import visualize
import skimage 
import cv2
import uuid
class_names = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear', 'vest', 'sling', 
               'shorts', 'trousers', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress',
               'vest_dress', 'sling_dress']

class TestConfig(Config):
     NAME = "test"
     GPU_COUNT = 1
     IMAGES_PER_GPU = 1
     NUM_CLASSES = 1 + 13

rcnn = MaskRCNN(mode='inference', model_dir='/home/link/Desktop/final_100', config=TestConfig())
rcnn.load_weights('mask_rcnn_deepfashion2_0100.h5', by_name=True)
img = skimage.io.imread('Test7.jpg')
results = rcnn.detect([img], verbose=1)
r = results[0]
visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
mask = r['masks']
mask1 = mask.astype(int)
img[:,:,2] = img[:,:,1] * mask1[:,:,2]
skimage.io.imsave("detecteded.jpg",img[:,:,2])