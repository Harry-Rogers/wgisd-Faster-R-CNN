
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

import time




def visual_test(model, model_name, device, frame):
        # read class_indict
    category_index = {"Grape": 1}
    preds = []
    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(frame)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    img = list(img)
    model.eval()
    with torch.no_grad():
        since = time.time()
        predictions = model(img)
        print('Time:{}s'.format(time.time() - since))
        predictions = list(predictions[1])
        predictions = predictions[0]
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        
        print(predict_boxes)
        print(predict_classes)
        print(predict_scores)
        
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


#torch.jit.load('./saved_models/tv-training-Mob.pt')

# Load ScriptModule from io.BytesIO object
with open('./saved_models/tv-training-QAT-Mob.pt', 'rb') as f:
    buffer = io.BytesIO(f.read())

# Load all tensors to the original device
torch.jit.load(buffer)

# Load all tensors onto CPU, using a device
buffer.seek(0)

loaded_model = torch.jit.load(buffer, map_location=torch.device('cpu'))

prev_frame_time = 0

cap = cv.VideoCapture(0)

loaded_model.eval()

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
 
    # font which we will be using to display FPS
    font = cv2.FONT_HERSHEY_SIMPLEX
    # time when we finish processing for this frame
    new_frame_time = time.time()
 
    # Calculating the fps
 
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
 
    # converting the fps into integer
    fps = int(fps)
 
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
    print("FPS = " + fps)
    # Display the resulting frame
    cv.imshow('frame', gray)
    image_np = np.array(frame)

    input_tensor = torch.tensor(np.expand_dims(image_np, 0), dtype=torch.float32)
    list_tensor = list(input_tensor)
    visual_test(loaded_model, model_name="norm", device="cpu", frame=frame)
    
    
   
    
    
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()