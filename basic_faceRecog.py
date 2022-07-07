import cv2
import numpy as np
import face_recognition

img_main = face_recognition.load_image_file('agra_main.jpeg')
img_main = cv2.cvtColor(img_main, cv2.COLOR_BGR2RGB)

img_test1 = face_recognition.load_image_file('agra_test.jpeg')
img_test1 = cv2.cvtColor(img_test1, cv2.COLOR_BGR2RGB)

face_main = face_recognition.face_locations(img_main)[0]
encode_main = face_recognition.face_encodings(img_main)[0]
cv2.rectangle(img_main, (face_main[3], face_main[0]), (face_main[1], face_main[2]), (250, 0, 250), 2)

face_test1 = face_recognition.face_locations(img_test1)[0]
encode_test1 = face_recognition.face_encodings(img_test1)[0]
cv2.rectangle(img_test1, (face_test1[3], face_test1[0]), (face_test1[1], face_test1[2]), (50, 50, 50), 2)

result1 = face_recognition.compare_faces([encode_main], encode_test1)
face_dis = face_recognition.face_distance([encode_main], encode_test1)
print(result1, face_dis)
cv2.putText(img_test1, f'{result1} {round(face_dis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('AGRA Main', img_main)
cv2.imshow('AGRA Test', img_test1)
cv2.waitKey(0)
