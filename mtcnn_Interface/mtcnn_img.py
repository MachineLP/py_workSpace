
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: liupeng

import cv2
import detect_face

img_path = 'lp.jpg'
img = cv2.imread(img_path)
bounding_boxes, _ = detect_face.ft(img)

for face_position in bounding_boxes:
    
    face_position=face_position.astype(int)
    
    #print((int(face_position[0]), int( face_position[1])))
    #word_position.append((int(face_position[0]), int( face_position[1])))
   
    cv2.rectangle(img, (face_position[0], 
                    face_position[1]), 
              (face_position[2], face_position[3]), 
              (0, 255, 0), 2)

# Display the resulting img
cv2.imshow('Video', img)
cv2.waitKey(0)
cv2.destroyAllWindows()