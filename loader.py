#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 10:44:40 2021
@author: harry
"""

import torch
import torchvision
import io
import numpy as np
import cv2 as cv
import cv2
from torchvision import transforms
from PIL import Image
import collections
import random
import PIL.ImageDraw as ImageDraw


import time
STANDARD_COLORS = [
    'Pink', 'Green', 'SandyBrown',
    'SeaGreen',  'Silver', 'SkyBlue', 'White',
    'WhiteSmoke', 'YellowGreen'
]

def draw_box(image, boxes, classes, line_thickness=20):
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    
    col = int(random.random() * len(STANDARD_COLORS))
    #filter_low_thresh(boxes, scores, classes, category_index, thresh, box_to_display_str_map, box_to_color_map, col)

    #draw = ImageDraw.Draw(image)
    #print(image)
    #im_width, im_height = image.size
 
    
    rects = []
    # loop over the detections
    for i in range(0, len(boxes)):
        box = boxes[i]
        rects.append(box)
    			# draw a bounding box surrounding the object so we can
    			# visualize it
        (startX, startY, endX, endY) = box
        print((startX, startY, endX, endY))
        cv2.rectangle(frame, (startX, startY), (endX, endY),
    				(0, 255, 0), 2)
    
    
    
    for box, color in box_to_color_map.items():
        xmin, ymin, xmax, ymax = box
        (left, right, top, bottom) = (xmin * 1, xmax * 1,
                                      ymin * 1, ymax * 1)
        image = cv2.rectangle(image, (left, top), (bottom, right), width=line_thickness)
    return image
        #draw.line([(left, top), (left, bottom), (right, bottom),
        #           (right, top), (left, top)], width=line_thickness, fill=color)
        #draw_text(draw, box_to_display_str_map, box, left, right, top, bottom, color)

def filter_low_thresh(boxes, scores, classes, category_index, thresh):
    for i in range(boxes.shape[0]):
        if scores[i] > thresh:
            box = tuple(boxes[i].tolist())  # numpy -> list -> tuple
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]
            #else:
            #    class_name = 'N/A'
            #display_str = str(class_name)
            #display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
            #box_to_display_str_map[box].append(display_str)
            #box_to_color_map[box] = STANDARD_COLORS[col]
        else:
            break  # Scores have been sorted


def visual_test(model, model_name, device, frame):
        # read class_indict
    category_index = {"Grape": 1}
    preds = []
    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(frame)
    img.to(device)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    img = list(img)
    
    model.eval()
    with torch.no_grad():
        since = time.time()
        predictions = model(img)
        #print('Time:{}s'.format(time.time() - since))
        predictions = list(predictions[1])
        predictions = predictions[0]
        predict_boxes = predictions["boxes"].to(device)
        predict_classes = predictions["labels"].to(device)
        predict_scores = predictions["scores"].to(device)
        
        #print(type(predict_boxes))
        #print(predict_classes)
        print(predict_scores)
        thresh=0.8
        for i in range(0, len(predict_boxes)):
            if predict_scores[i] > thresh:
                print("here")
                print(predict_boxes[i])
                preds.append(predict_boxes[i])
        #filter_low_thresh(predict_boxes, predict_scores, category_index, category_index, thresh)
        
        #draw_box(img, predict_boxes, category_index, line_thickness=20)
        
        
        
        rects = []
        for i in range(0, len(preds)):
            box = preds[i]
            rects.append(box)
        			# draw a bounding box surrounding the object so we can
        			# visualize it
            (startX, startY, endX, endY) = box
            print((startX, startY, endX, endY))
            startX = int(startX)
            startY = int(startY)
            endX = int(endX)
            endY = int(endY)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
        				(0, 255, 0), 2)
        
        
        
        
        #iamge = cv2.rectangle(img, predict, pt2, color)
        #plt.imshow(frame)
        #plt.savefig(model_name + ".jpg")
        #plt.show()
            
    
        predict = ""
        for box, score in zip(predict_boxes, predict_scores):
            str_box = ""
            box[2] = box[2] - box[0]
            box[3] = box[3] - box[1]
            for b in box:
                str_box += str(b) + ' '
            predict += str(score) + ' ' + str_box
        preds.append(predict)


model = torch.jit.load('./saved_models/tv-training-Res.pt', map_location="cpu:0")

# Load ScriptModule from io.BytesIO object
with open('./saved_models/tv-training-Res.pt', 'rb') as f:
    buffer = io.BytesIO(f.read())

# Load all tensors to the original device
torch.jit.load(buffer)

# Load all tensors onto CPU, using a device
buffer.seek(0)

loaded_model = torch.jit.load(buffer, map_location=torch.device('cpu'))

prev_frame_time = 0

cap = cv.VideoCapture(0)


loaded_model.eval()

model.eval()
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
     # resizing the frame size according qto our need
    gray = cv2.resize(frame, (500, 300))
 
    # time when we finish processing for this frame
    new_frame_time = time.time()
 
    # Calculating the fps
 
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
 
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
    print("FPS = " + fps)
    # Display the resulting frame
    
    image_np = np.array(frame)

    input_tensor = torch.tensor(np.expand_dims(image_np, 0), dtype=torch.float32)
    list_tensor = list(input_tensor)
    visual_test(model, model_name="norm", device="cpu", frame=frame)
    #cv.imshow('frame', image)
    cv.imshow('frame', frame)
   
    
    
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
