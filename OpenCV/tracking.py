#encoding:utf-8  

import cv2  
import numpy as np  
camera=cv2.VideoCapture(0)  
firstframe=None  
while True:  
    ret,frame = camera.read()  
    if not ret:  
        break  
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  
    gray=cv2.GaussianBlur(gray,(21,21),0)  
    if firstframe is None:  
        firstframe=gray  
        continue  
      
    frameDelta = cv2.absdiff(firstframe,gray)  
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]  
    thresh = cv2.dilate(thresh, None, iterations=2)  
    # cnts= cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  
      
    x,y,w,h=cv2.boundingRect(thresh)  
    frame=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)  
    cv2.imshow("frame", frame)  
    cv2.imshow("Thresh", thresh)  
    cv2.imshow("frame2", frameDelta)  
    key = cv2.waitKey(1)&0xFF  
      
    if key == ord("q"):  
        break  
  
camera.release()  
cv2.destroyAllWindows()