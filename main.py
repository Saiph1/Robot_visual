# from utils import display, Detect_marker, Detect_edge
# from utils import get_sharp_points, Detect_object, box, demo, video_demo, video, videodraw
from mac_utils import video, test, orange_box
import cv2 
import numpy as np
import math

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
    # centers = []
    x=[]
    y=[]
    tmp_x = [0]
    tmp_y = [0]
    height = [0]
    # ratio = 35.6/159
    ratio = 35.6/171.38
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        #crop
        frame = frame[0:280, 0:500]
        x1, y1 = orange_box(frame)
        x.append(x1)
        y.append(y1)
        # video(frame)
        # compute moving average
        window_size = 3
        weights = np.ones(window_size) / window_size
        x_ma = np.convolve(x, weights, mode='valid')
        y_ma = np.convolve(y, weights, mode='valid')
        # define q as the exit button
        for i in range(len(x_ma)):
            if(i>7 and len(x_ma)>7): 
                break
            # print(i)
            cv2.circle(frame, [int(x_ma[len(x_ma)-1-i]), int(y_ma[len(y_ma)-1-i])], 2, (255, 255, 0), -1)
        
        # update once every 10 frames
        if (len(x)>1):
            dx = np.gradient(x)
            dy = np.gradient(y)
            if (not len(x)%10):
                tmp_x[0] = dx[len(dx)-1]
                tmp_y[0] = dy[len(dy)-1]
            cv2.putText(frame, "dx :{} cm".format(round(tmp_x[0]*ratio, 3)), (200,230), 
                cv2.FONT_HERSHEY_PLAIN, 1, 
                (0, math.floor(200-abs(tmp_x[0])*20), math.floor(abs(tmp_x[0])*20)), 2)
            cv2.putText(frame, "dy :{} cm".format(round(tmp_y[0]*ratio, 3)), (200,250), 
                cv2.FONT_HERSHEY_PLAIN, 1,
                (0, math.floor(200-abs(tmp_y[0])*20), math.floor(abs(tmp_y[0])*20)), 2)
        
        cv2.putText(frame, "x_dis :{} cm".format(round((x[len(x)-1]-x[0])*ratio, 3)), (20,230), 
                cv2.FONT_HERSHEY_PLAIN, 1, 
                (0, 200, 0, 2),2)
        cv2.putText(frame, "y_dis :{} cm".format(-round((y[len(y)-1]-x[0])*ratio, 3)), (20,250), 
                cv2.FONT_HERSHEY_PLAIN, 1,
                (0, 200, 0, 2),2)
        
        cv2.imshow('Processed', frame)

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
