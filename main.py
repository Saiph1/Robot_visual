from utils import display, Detect_marker, Detect_edge, get_sharp_points, Detect_object, box
import cv2 
import numpy as np

if __name__ == "__main__" :
    # Read the image
    img = cv2.imread('./Test/test6.jpg')
    # Draw markers axes and contours
    # get_sharp_points(img)
    # img = Detect_edge(img)
    # img = Detect_marker(img)
    box(img)
    # Display
    # img2 = Edge_detect(img)
    display(img)
    cv2.imwrite('./result1.png', img)

