import cv2
import dlib
import matplotlib
import face_recognition

imagePath = "Politician_photos/Shri_Rajnath_Singh.jpg"
outPath = "Combined_photos/Shri_Rajnath_Singh.jpg"

img = cv2.imread(imagePath)

gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
color_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Using cv2 haarcascade
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# face_recognition
face_locations = face_recognition.face_locations(color_img)

face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

# dlib
faceDetector = dlib.get_frontal_face_detector()
faces = faceDetector(gray_image)

# haarcascade (green)
for (x, y, w, h) in face:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

# face_recognition (red)
for (top, right, bottom, left) in face_locations:
    cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 4)

# dlib (blue)
for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)

cv2.imwrite(outPath, img)