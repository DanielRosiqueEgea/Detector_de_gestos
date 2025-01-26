import cv2 as cv
import os
import numpy as np
from opencv_utils import *


clases_file = "openCv/weights/yolov3.txt"
weights_file = "openCv/weights/yolov3.weights"
cfg_file = "openCv/weights/yolov3.cfg"



classes = None
with open(clases_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv.dnn.readNet(weights_file, cfg_file)


cap = cv.VideoCapture(0)

if not (cap.isOpened()):
    print("Could not open video device")
    exit(-1)

frame_idx = 0
cv.namedWindow("preview")
while cv.getWindowProperty('preview', cv.WND_PROP_VISIBLE) == 1:
# Capture frame-by-frame
    ret, frame = cap.read()
    if ret is False:
        break
    frame = cv.flip(frame, 1)
    
    generate_blob(frame, net)
    frame_idx += 1
    # Display the resulting frame
    frame = detect_objects(net,frame,classes,COLORS)
    cv.imshow('preview',frame)
    # Waits for a user input to quit the application
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
