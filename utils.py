import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os

def load_and_resize_image(path):
    image=cv2.imread(path)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image=cv2.resize(image,(1280,856))
    image=image/255.0
    return image

def display_image(image):
  fig = plt.figure(figsize=(20, 15))
  plt.axis("off")
  plt.imshow(image)

def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
  classes=list(set(class_names))
  colors=[(random.random(),random.random(),random.random()) for _ in range(len(classes))]
  class_color={key:val for key,val in zip(classes,colors)}

  for i in range(min(boxes.shape[0], max_boxes)):
    if scores[i] >= min_score:
      ymin, xmin, ymax, xmax = tuple(boxes[i])
      display_str = "{}: {}%".format(class_names[i].decode("ascii"),int(100 * scores[i]))
      h,w,_=image.shape
      xmin,ymin,xmax,ymax=map(int,[w*xmin,h*ymin,w*xmax,h*ymax])
      image=cv2.rectangle(image,(xmin,ymin),(xmax,ymax),class_color[class_names[i]],5)
      image=cv2.putText(image,display_str,(xmin,ymin-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,class_color[class_names[i]],2)
  return image

def run_detector(detector, image):
  converted_img  = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]
  result = detector(converted_img)
  result = {key:value.numpy() for key,value in result.items()}
  image_with_boxes = draw_boxes(image, boxes=result["detection_boxes"],class_names=result["detection_class_entities"], scores=result["detection_scores"])
  display_image(image_with_boxes)
  return image_with_boxes
