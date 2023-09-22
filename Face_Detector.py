import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# img = cv2.imread('peoples.png')
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # detects faces(coordinates of rectangles)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)


    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)


    cv2.imshow('cl',frame)
    key = cv2.waitKey(2)


    if key==81 or key==113:
        break



print("code complete")