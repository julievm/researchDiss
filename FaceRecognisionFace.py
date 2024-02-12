import cv2
import dlib
import matplotlib
import face_recognition
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

imagePath = 'Politician_photos/Shri_Rajnath_Singh.jpg'
Outpath = 'Politician_photos_face_rec/Shri_Rajnath_Singh.jpg'

img = cv2.imread(imagePath)

color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


face_locations = face_recognition.face_locations(color_img)


for (top, right, bottom, left) in face_locations:
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 4)
    print(left, top, (right-left), (bottom-top))

cv2.imwrite(Outpath, img)