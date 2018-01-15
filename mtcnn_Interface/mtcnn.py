
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: liupeng

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from os.path import join as pjoin
import sys
import copy
import detect_face

video_capture = cv2.VideoCapture(0)
c=0

frame_interval=1
 
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    timeF = frame_interval
    
    if(c%timeF == 0): #frame_interval==3, face detection every 3 frames
        
        img = frame

        bounding_boxes, _ = detect_face.ft(img)

        nrof_faces = bounding_boxes.shape[0]#number of faces
        #print('找到人脸数目为：{}'.format(nrof_faces))
        for face_position in bounding_boxes:
            
            face_position=face_position.astype(int)
            
            #print((int(face_position[0]), int( face_position[1])))
            #word_position.append((int(face_position[0]), int( face_position[1])))
           
            cv2.rectangle(frame, (face_position[0], 
                            face_position[1]), 
                      (face_position[2], face_position[3]), 
                      (0, 255, 0), 2)
    c+=1

    # Display the resulting frame

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture

video_capture.release()
cv2.destroyAllWindows()