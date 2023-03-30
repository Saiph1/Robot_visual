from utils import display, Detect_marker
import cv2 
import numpy as np

# Read the image
img = cv2.imread('test3.jpg')



# Draw markers axes and contours
img = Detect_marker(img)
# Display
display(img)

