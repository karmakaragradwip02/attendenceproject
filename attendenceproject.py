import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Images'
images = []
PersonNames = []
myList = os.listdir(path)
print(myList)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    PersonNames.append(os.path.splitext(cls)[0])
print(PersonNames)


def findEncodings(imgs):
    encode_list = []
    for Img in imgs:
        Img = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(Img)[0]
        encode_list.append(encode)
    return encode_list


encodeList_known = findEncodings(images)
print("Encoding Complete")


def attendanceMark(names):
    with open('attendence_Mark.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if names not in nameList:
            time_now = datetime.now()
            timeStr = time_now.strftime('%H:%M:%S')
            dateStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'{names}, {dateStr}, {timeStr}')


cap = cv2.VideoCapture(1)
while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
#    cv2.imshow('frame', imgS)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    faceCurrFrame = face_recognition.face_locations(faces)
    encodeCurFrame = face_recognition.face_encodings(faces, faceCurrFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurrFrame):
        matches = face_recognition.compare_faces(encodeList_known, encodeFace)
        faceDist = face_recognition.face_distance(encodeList_known, encodeFace)

        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = PersonNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            attendanceMark(name)

    cv2.imshow("camera", frame)
    if cv2.waitKey(10) == 13:
        break
cap.release()
cv2.destroyAllWindows()
