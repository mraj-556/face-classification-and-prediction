import cv2
import os
import time
import numpy as np

cap = cv2.VideoCapture(0)
progress = 1
collect_flag=1
user_id = 0
while collect_flag:
    os.system('clear')
    user_name = input('Enter your name : ')
    try:
    # if True:
        os.mkdir('/home/ashutosh/Desktop/project/face_security/dataset/{}/'.format(user_name))
        user_id = input('Enter your id : ')
        while True:
            bolean,frame = cap.read()
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            face_classifier_obj = cv2.CascadeClassifier('library/haarcascade_frontalface_default.xml')
            face_found = face_classifier_obj.detectMultiScale(gray_frame,1.3,5)
            if face_found!=():
                try:
                # if True:
                    c = 0
                    for (x,y,w,h) in face_found:
                        c+=1
                        face_only = gray_frame[y:y+h,x:x+w]
                        org_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)[y:y+h,x:x+w]
                        face_only = cv2.resize(org_frame,(400,400))
                        database_path = '/home/ashutosh/Desktop/project/face_security/dataset/{}/'.format(user_name)+'{}_'.format(user_id)+str(progress)+'.jpg'
                        cv2.imwrite(database_path,org_frame)
                        cv2.putText(frame,str(progress),(0,50),cv2.FONT_ITALIC,1,(0,255,0),2)
                        face_border = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
                        cv2.imshow('f',frame)
                        progress+=1
                except:
                    pass
                if progress==51:
                    print('Collection Completed...')
                    collect_flag = 0
                    break
            else:
                print('sorry',end='\r')
            if cv2.waitKey(1)==ord('q'):
                break
    except:
        print('User exists...',end='\r')
        time.sleep(3)

cv2.destroyAllWindows
