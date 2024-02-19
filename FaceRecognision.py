import cv2
import dlib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

imagePath = 'Politician_photos/arvind-kejriwal.jpg'
Outpath = 'Features/more_neighbors10.jpg'

img = cv2.imread(imagePath)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Using cv2 haarcascade
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# The scale impacts how much the image is scaled each iteration, the lower the value,
# the more accurate but more computational expensive

# The minNeightbors, the higher the higher the confidence, but also harder to detect faces.
face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=10, minSize=(40, 40)
)

#
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    print(x, y, w, h)

cv2.imwrite(Outpath, img)
