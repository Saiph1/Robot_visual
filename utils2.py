import cv2
import numpy as np

def video(img):
    # Load the image
     
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the orange color in HSV
    lower_orange = (5, 150, 120)
    upper_orange = (20, 255, 255)

    # Create a mask for the orange color
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Apply a morphological opening to the mask to remove small objects
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)
    
    im, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Apply a Gaussian blur to the mask to reduce noise
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Apply the mask to the original image to extract the orange object
    # orange = cv2.bitwise_and(img, img, mask=mask)
    
    max_cont = []
    for contour in contours:
    # Compute the area of the contour
        area = cv2.contourArea(contour)

        # If the area is large enough, it's likely the orange object
        if area > 2000:
            # cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
            max_cont.append(contour)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Threshold the image to obtain a binary image
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Apply a morphological opening to the binary image to remove noise
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # opened = cv2.dilate(opened, kernel, iterations=2)

    # Find contours in the opened image
    im, contours, _ = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    circle_centers = []
    # Iterate through each contour and check if it is closed
    for contour in contours:
        # Compute the area of the contour
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # If the area is small, it's likely a black mark
        if (10<area < 100):
            for j in (contour[0]):
                if cv2.pointPolygonTest(max_cont[0], tuple([int(j[0]), int(j[1])]), False) >= 0:
                    # if so, draw the mark
                    # cv2.circle(img, (cX, cY), 7, (255, 255, 255), -1)
                    cv2.drawContours(img, contour, -1, (0, 255, 0), 2)
                    break            

def orange_box(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the orange color in HSV
    lower_orange = (10, 220, 120)
    upper_orange = (20, 255, 235)

    # Create a mask for the orange color
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Apply a morphological opening to the mask to remove small objects
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)
    
    im, contours, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    # Find largest contour with largest area
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]

    # Get rect
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    # print("w+h", h)
    # w=100

    # Display rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(img, [box], True, (255, 0, 0), 2)
    return [int(x), int(y)]

    # Detect markers
    # parameters = cv2.aruco.DetectorParameters_create()
    # aruco_dict = cv2.aruco.Dictionary_get(aruco.DICT_6X6_250)
    # corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    # int_corners = np.int0(corners)
    # cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

    # # Aruco Perimeter
    # aruco_perimeter = cv2.arcLength(corners[0], True)

    # # Pixel to cm ratio
    # pixel_cm_ratio = aruco_perimeter / 2.5
    
    # # Get Width and Height of the Objects by applying the Ratio pixel to cm
    # object_width = w / pixel_cm_ratio
    # object_height = h / pixel_cm_ratio
    # cv2.putText(img, "Width :{} cm".format(round(object_width, 3)), (int(x-600), int(y+800)), cv2.FONT_HERSHEY_PLAIN, 10, (0, 200, 0), 10)
    # cv2.putText(img, "Height :{} cm".format(round(object_height, 3)), (int(x-600), int(y+650)), cv2.FONT_HERSHEY_PLAIN, 10, (0, 200, 0), 10)

def test(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the orange color in HSV
    lower_orange = (10, 220, 120)
    upper_orange = (20, 255, 255)

    # Create a mask for the orange color
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Apply a morphological opening to the mask to remove small objects
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=2)
    
    im, contours, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    # Find largest contour with largest area
    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt=contours[max_index]

    # Get rect
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect
    # Display rectangle
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    # cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(img, [box], True, (255, 0, 0), 2)
    # return max(w)
    return w