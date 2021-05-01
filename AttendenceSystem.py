import cv2
import numpy as np
import face_recognition
import os
import pyttsx3
from datetime import date
from datetime import datetime
# from PIL import ImageGrab


path = 'E:\Course\Python Programs\Python\FaceRecognition_Attendance_System\ImagesAttendance'
images = []
classNames = []
myList = os.listdir(path)
#print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
#print(classNames)

engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.runAndWait()


print(" * * * Encoding Start * * * ")
def talk(audio):
    engine.say(audio)
    engine.runAndWait()

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            today = date.today()
            now = datetime.now()
            dateString = today.strftime("%b-%d-%Y")
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dateString},{dtString}')
            talk("Thank you, Your attendance has been saved")
        if name in nameList:
            talk("oops!!! Attendance already saved")

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

encodeListKnown = findEncodings(images)
print(" * * * Encoding Complete * * * ")
print("\n")

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    # img = captureScreen()
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceloc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if faceDis[matchIndex] < 0.50:
            name = classNames[matchIndex].upper()
        else:
            name = 'Unknown'
        # print(name)
        y1, x2, y2, x1 = faceloc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        print(" * * * Attendance Counting * * * ")
        markAttendance(name)

    cv2.imshow('WebCam', img)
    if cv2.waitKey(10) == ord('q'):
        break