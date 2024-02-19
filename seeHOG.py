# https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f
import cv2
import matplotlib
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

img = imread("Politician_photos/Modi.png")
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# resized_img = resize(gray_image, (128*4, 64*4))
fd, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True)
plt.axis("off")
plt.imshow(hog_image, cmap="gray")
plt.show()
