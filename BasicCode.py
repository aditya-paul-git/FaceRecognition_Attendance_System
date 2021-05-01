import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('Image/elon mask.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('Image/bill gate.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

resize_img1 = cv2.resize(imgElon, (800,500))
resize_img2 = cv2.resize(imgTest, (600,400))

faceLoc = face_recognition.face_locations(resize_img1)[0]
encodeElon = face_recognition.face_encodings(resize_img1)[0]
cv2.rectangle(resize_img1, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(resize_img2)[0]
encodeTest = face_recognition.face_encodings(resize_img2)[0]
cv2.rectangle(resize_img2, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results, faceDis)
cv2.putText(resize_img2, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Elon Musk', resize_img1)
cv2.imshow('Elon Test', resize_img2)
cv2.waitKey(0)
