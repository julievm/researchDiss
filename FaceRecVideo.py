import cv2
import matplotlib
import dlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

videoPath = 'Politician_videos/Trump.mp4'

video = cv2.VideoCapture(videoPath)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Get the properties of the video
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

output = cv2.VideoWriter(
    "Politician_videos_face_detected/Trump.mp4", cv2.VideoWriter_fourcc(*'MPEG'), fps, (frame_width, frame_height))

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

while True:
    result, frame = video.read()
    if result is False:
        break

    faces = detect_bounding_box(frame)
    output.write(frame)
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
output.release()
cv2.destroyAllWindows()