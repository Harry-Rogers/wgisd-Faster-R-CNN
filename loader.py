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


#torch.jit.load('./saved_models/tv-training-Mob.pt')

# Load ScriptModule from io.BytesIO object
with open('./saved_models/tv-training-Mob.pt', 'rb') as f:
    buffer = io.BytesIO(f.read())

# Load all tensors to the original device
torch.jit.load(buffer)

# Load all tensors onto CPU, using a device
buffer.seek(0)
torch.jit.load(buffer, map_location=torch.device('cpu'))

# Load all tensors onto CPU, using a string
buffer.seek(0)
loaded_model = torch.jit.load(buffer, map_location='cpu')

cap = cv.VideoCapture(0)
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
    gray = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    # Display the resulting frame
    cv.imshow('frame', gray)
    image_np = np.array(frame)

    input_tensor = torch.tensor(np.expand_dims(image_np, 0), dtype=torch.float32)
    list_tensor = list(input_tensor)
    
    loaded_model(list_tensor)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
