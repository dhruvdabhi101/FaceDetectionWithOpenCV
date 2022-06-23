#importing needed modules


import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib


#Creating a Detector
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if ret == False:
        continue
    all_faces = detector.detectMultiScale(frame,1.5,5)
    for faces in all_faces:
        #creating rectangles around the faces
        x,y,w,h = faces
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("face detection",frame)  
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break

# Releasing Camera and Destroying the window
cam.release()
cv2.destroyAllWindows()
