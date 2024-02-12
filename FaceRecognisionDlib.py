import cv2
import dlib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

imagePath = 'Politician_photos/Pete_Buttigieg.jpg'
Outpath = 'Politician_photos_face_detection_dlib/Pete_Buttigieg.jpg'

img = cv2.imread(imagePath)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


faceDetector = dlib.get_frontal_face_detector()


faces = faceDetector(gray_image)

for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    print(x, y, w, h)

cv2.imwrite(Outpath, img)