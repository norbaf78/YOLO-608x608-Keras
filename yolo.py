# import the needed modules
import os
#from matplotlib.pyplot import imshow
import scipy.io
#import scipy.misc
import numpy as np
#from PIL import Image
import cv2

from keras import backend as K
from keras.models import load_model

# The below provided fucntions will be used from yolo_utils.py
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, preprocess_image_new, draw_boxes

# The below functions from the yad2k library will be used
from yad2k.models.keras_yolo import yolo_head, yolo_eval


cap=cv2.VideoCapture("http://root:progtrl01@192.168.208.55/mjpg/1/video.mjpg")
_,input_image=cap.read() # acquire a new image
height, width, _ = input_image.shape 
width = np.array(width, dtype=float)
height = np.array(height, dtype=float)
#Assign the shape of the input image to image_shape variable
image_shape = (height,width)


#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")

#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolov2.h5")

#Print the summery of the model
yolo_model.summary()

#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

#Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
#If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)


# Initiate a session
sess = K.get_session()


while True:
    _,input_image=cap.read() # acquire a new image
    #Preprocess the input image before feeding into the convolutional network
    image_data = preprocess_image_new(input_image, model_image_size = (608, 608))

    #Run the session
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})

    #Print the results
    print('Found {} boxes in image '.format(len(out_boxes)))
    #Produce the colors for the bounding boxs
    colors = generate_colors(class_names)
    #Draw the bounding boxes
    draw_boxes(input_image, out_scores, out_boxes, out_classes, class_names, colors)

    resized_image = cv2.resize(input_image,  (int(input_image.shape[1]/2), int(input_image.shape[0]/2)))
    cv2.imshow("CAMERA IMAGE", resized_image)   
    #cv2.imshow("CAMERA IMAGE", input_image) 
        
    ch = 0xFF & cv2.waitKey(1)
    if ch == 27:
        break
cv2.destroyAllWindows()

