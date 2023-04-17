from utils import display, Detect_marker, Detect_edge
from utils import get_sharp_points, Detect_object, box, demo, video_demo, video, videodraw
import cv2 
import numpy as np

if __name__ == "__main__" :
    # Read the image
    # img = cv2.imread('./Test/test6.jpg')
    # Draw markers axes and contours
    # get_sharp_points(img)
    # img = Detect_edge(img)
    # img = Detect_marker(img)

    # Stage 1:
    # box(img)
    # display(img)
    # cv2.imwrite('./result1.png', img)

    # Stage 2 for final:
    # img = cv2.imread('./Demo/20230417195418.jpg')
    # demo(img)
    # display(img)

    # Stage 3: 
    cap = cv2.VideoCapture('./Demo/PXL_20230417_134842457.TS.mp4')

    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        video(frame)
        cv2.imshow('with circle', frame)
        # define q as the exit button
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # release the video capture object
    cap.release()
    # Closes all the windows currently opened.
    cv2.destroyAllWindows()
    

    # # COLOR TOOL
    # # get HSV value
    # # Load the image
    # img = cv2.imread('orange.png')
    # # Create a function to handle mouse events
    # def mouse_callback(event, x, y, flags, params):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         # Get the HSV value of the clicked pixel
    #         hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #         pixel_value = hsv[y, x]
    #         print('HSV value:', pixel_value)

    # # Create a window and set the mouse callback function
    # cv2.namedWindow('image')
    # cv2.setMouseCallback('image', mouse_callback)

    # # Display the image
    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
