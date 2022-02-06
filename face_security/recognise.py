import cv2
import os
import time

names = []
for users in os.listdir('dataset'):
    names.append(users)

recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read('training.yml')


cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('http://192.168.81.133:4747/video')
cap.set(cv2.CAP_PROP_BUFFERSIZE,15)
while True:
    success,frame = cap.read()
    face_classifier_obj = cv2.CascadeClassifier('library/haarcascade_frontalface_default.xml')
    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    face_found = face_classifier_obj.detectMultiScale(rgb_frame,1.3,5)
    c=0
    if face_found!=():
        for (x,y,w,h) in face_found:
            c+=1
        if c==1:
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            face_border = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
            ids, matching_perecentage = recogniser.predict(gray_frame[y:y+h,x:x+w])
            if ids and matching_perecentage>=85:
                # print(garbage)
                cv2.putText(frame,names[ids-1],(200,200),cv2.FONT_ITALIC,2,(0,0,255),3)
        else:
            print('Multiple face detected...!')
    else:
        print('Face Not found...',end='\r')
    cv2.imshow('Captured Photo',frame)
    if cv2.waitKey(1)==ord('q'):
        break