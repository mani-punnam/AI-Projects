from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os
import matplotlib.pyplot as plt

CLASS_NAMES = open(r'C:\Users\Manikanta Punnam\coursera\Notebooks\Convolutional Neural Networks\keras-mask-rcnn\keras-mask-rcnn'+'/coco_labels.txt').read().strip().split("\n")

class SimpleConfig(Config):
	# give the configuration a recognizable name
	NAME = "coco_inference"
	# set the number of GPUs to use along with the number of images
	# per GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
	# number of classes (we would normally add +1 for the background
	# but the background class is *already* included in the class
	# names)
	NUM_CLASSES = len(CLASS_NAMES)
    
# initialize the inference configuration
config = SimpleConfig()
# initialize the Mask R-CNN model for inference and then load the
# weights
print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config,
	model_dir=os.getcwd())
model.load_weights(r'C:\Users\Manikanta Punnam\coursera\Notebooks\Convolutional Neural Networks\keras-mask-rcnn\keras-mask-rcnn'+'/mask_rcnn_coco.h5', by_name=True)


def extract_foreground(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # converting from bgr rgb
    image = imutils.resize(image, width=512) # resizing to a particular shape , so this satisfies the Mask R-CNN requirements.
    # perform a forward pass of the network to obtain the results
    print("[INFO] making predictions with Mask R-CNN...")
    r = model.detect([image], verbose=1)[0]
    indices = np.argsort(r['scores'])[::-1] #sorting detected objects based on probability they get.
    for i in indices:
        classID = r["class_ids"][i] # getting the class id (1 means person)
        if(classID == 1):
            score = r['scores'][i]
            if(score > 0.5): #checking if the detected object satisfies the given threshold.Here we want only the closest person,whenever we get the closest person, we will quit from this loop.
                mask = r["masks"][:, :, i] # taking the mask of corresponding object.
                mask = mask.astype('uint8')
                image_bgr = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
                blur_image = cv2.GaussianBlur(image_bgr,(41,41),0) # creating a blur image of original image.
                mask_1 = np.expand_dims(mask,axis=2)
                outputMask = np.concatenate((mask_1,mask_1,mask_1),axis=-1) # creating a 3-dimensional mask to satisfies this condition below.
                output_image = np.where((outputMask==0),blur_image,image_bgr) #overlaying detected object onto the blurred image using mask.
                return output_image,True #returning resulting image.
            else:
                return None,False # returning False in case if the closest person object is not found.
        

cap = cv2.VideoCapture('sample_video.avi') #Opening a recorded video.
frame_width = int(cap.get(3)) # getting the frame size of this video
frame_height = int(cap.get(4))
size = (frame_width, frame_height)   
result = cv2.VideoWriter('sample_video_output.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size)
while cap.isOpened():
    ret,frame = cap.read()
    h,w = frame.shape[:2]
    if(ret == True):
        output,found = extract_foreground(frame)
        if(found == False): # if there is no closest person object detected, then store the original frame into video file.
            output = frame
        else:
            output = cv2.resize(output,(w,h)) # resizing the output image size to the size of original frame.
        result.write(output)
    else:
        break
    if(cv2.waitKey(1) & 0XFF == 27):
        break
cv2.destroyAllWindows()
cap.release()