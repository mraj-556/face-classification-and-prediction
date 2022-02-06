import os
import cv2
import numpy as np
from PIL import Image


names = []
paths = []

for users in os.listdir('dataset'):
    names.append(users)

for name in names:
    images_for_training = os.listdir('dataset/{}'.format(name))
    for img in images_for_training:
        path_str = os.path.join('dataset/{}'.format(name),img)
        paths.append(path_str)

faces = []
face_id = []

for img_path in paths:
    image = Image.open(img_path)
    img_np_array = np.array(image,'uint8')
    faces.append(img_np_array)

    face_id_of_user = int(img_path.split('/')[2].split('_')[0])
    face_id.append(face_id_of_user)

face_id = np.array(face_id)

trainer = cv2.face.LBPHFaceRecognizer_create()
trainer.train(faces,face_id)
write('training.ymtrainer.l')

# print(face_id)
# print(names)