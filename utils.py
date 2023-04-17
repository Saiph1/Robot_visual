import cv2
import numpy as np
import cv2.aruco as aruco

camera_matrix = np.array([[3930.0, 0.0, 2040.0], [0.0, 3930.0, 1536.0], [0.0, 0.0, 1.0]]) # Specify the camera matrix
dist_coeffs = np.zeros((4,1)) # Specify the distortion coefficients

# This function is used to draw axes and the contours for the detected aruco markers. 
def Detect_marker(img):

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Define the ArUco dictionary
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

    # Define the ArUco parameters
    aruco_params = aruco.DetectorParameters_create()

    # Detect the ArUco markers in the image
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    # Iterate through the markers
    for i in range(len(ids)):
    # Get the corner coordinates of the marker
    # marker_corners = corners[i][0]

        # Calculate the orientation of the marker using the solvePnP function
        marker_length = 50 # Specify the length of the marker in meters
        # ret, rvec, tvec = cv2.solvePnP(np.array([[0, 0, 0], [marker_length, 0, 0], [0, marker_length, 0], [0, 0, marker_length]]), corners[i][0], camera_matrix, dist_coeffs)
        print("corners = ", corners[i])
        rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, camera_matrix, dist_coeffs)
        (rvec - tvec).any()  # get rid of that nasty numpy value array error

        aruco.drawDetectedMarkers(img, corners)
        aruco.drawAxis(img, camera_matrix, dist_coeffs, rvec, tvec, 0.01)  # Draw Axis
        
    return img 

# Display the img that is passed into this function and press any key to end. 
def display(img):
    # Display the image
    img = cv2.resize(img, (960, 540))
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Detect_edge(img):
    # Apply Canny edge detection
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 100, 200)

    # Find contours of edges
    image,contours,_ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Extract points of edges
    edge_points = []
    for contour in contours:
        for point in contour:
            edge_points.append(point.squeeze())

    # Display result
    img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    kernel = np.ones((3,3), np.uint8)
    img = cv2.erode(img, kernel, iterations = 1)
    img = cv2.dilate(img, kernel, iterations = 1)
    return img 

def Detect_object(img):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Apply threshold to convert the image to binary
    ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

    # Find contours of objects in the image
    image, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours of objects in the image
    img = cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    # ------------------------------------------------------------------------------------------
    # Convert image to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # Apply adaptive thresholding to the image
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # # Apply morphological operations to remove noise
    # kernel = np.ones((5,5), np.uint8)
    # thresh = cv2.erode(thresh, kernel, iterations=1)
    # thresh = cv2.dilate(thresh, kernel, iterations=1)

    # # Find contours of white objects in the image
    # image, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Draw bounding boxes around white objects
    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area > 500:
    #         x,y,w,h = cv2.boundingRect(contour)
    #         cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)

def get_sharp_points(img):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    aruco_params = aruco.DetectorParameters_create()
    # Detect ArUco marker
    corners, ids, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=aruco_params)
    # Estimate pose of camera relative to marker
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, camera_matrix, dist_coeffs)
    img_point = np.array([100, 200]).astype(np.float32)  # example point 

    ray = cv2.fisheye.distortPoints(img_point[None, None, :], camera_matrix, dist_coeffs)[0, 0, :]
    ray /= np.linalg.norm(ray)

    # Triangulate point in 3D space
    marker_point = np.array([0, 0, 0.05])  # use a point on the marker as a reference
    cam_point = np.dot(np.linalg.inv(camera_matrix), np.array([img_point[0], img_point[1], 1]))
    cam_point = cam_point / np.linalg.norm(cam_point)
    tvec = tvec.squeeze()
    point_3d_world = (marker_point - tvec) / np.dot(cam_point, np.transpose(rvec[0])).squeeze() * cam_point

    print("Point coordinate in 3D world space:", point_3d_world)

def box(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    image, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(img, [box], True, (255, 0, 0), 2)

    # Detect markers
    parameters = cv2.aruco.DetectorParameters_create()
    aruco_dict = cv2.aruco.Dictionary_get(aruco.DICT_6X6_250)
    corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    int_corners = np.int0(corners)
    cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

    # Aruco Perimeter
    aruco_perimeter = cv2.arcLength(corners[0], True)

    # Pixel to cm ratio
    pixel_cm_ratio = aruco_perimeter / 2.5
    
    # Get Width and Height of the Objects by applying the Ratio pixel to cm
    object_width = w / pixel_cm_ratio
    object_height = h / pixel_cm_ratio
    cv2.putText(img, "Width :{} cm".format(round(object_width, 3)), (int(x-600), int(y+800)), cv2.FONT_HERSHEY_PLAIN, 10, (0, 200, 0), 10)
    cv2.putText(img, "Height :{} cm".format(round(object_height, 3)), (int(x-600), int(y+650)), cv2.FONT_HERSHEY_PLAIN, 10, (0, 200, 0), 10)

def demo(img): 

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 20, 40)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    image, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(img, [box], True, (255, 0, 0), 2)

def video_demo(img):
    
    # Loop until the end of the video
        img = cv2.resize(img, (540, 380), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian blur to the grayscale image to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply adaptive thresholding to obtain a binary image
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

        # Invert the binary image to obtain black marks on white background
        thresh = cv2.bitwise_not(thresh)

        # Find contours in the binary image
        image, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Iterate through each contour and identify black marks on the orange object
        for contour in contours:
            # Compute the area of the contour
            area = cv2.contourArea(contour)

            # If the area is small and the contour is approximately circular, it's likely a black mark on the orange object
            if 20< area < 50 and len(contour) > 5:
                # Compute the convex hull of the contour
                hull = cv2.convexHull(contour)

                # Compute the solidity of the contour (ratio of contour area to convex hull area)
                solidity = area / cv2.contourArea(hull)

                # If the solidity is high, it's likely a black mark on the orange object
                if solidity > 0.8:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(img, center, radius, (0, 0, 255), 2)

def video_point(img):
    # Loop until the end of the video
    # Capture frame-by-frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(gray, 50, 100)
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    image, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(img, [box], True, (255, 0, 0), 2)

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
    
    
    image, contours, _  = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Apply a morphological opening to the binary image to remove noise
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours in the opened image
    image, contours, _ = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Iterate through each contour and check if it is closed
    for contour in contours:
        # Compute the area of the contour
        area = cv2.contourArea(contour)
        # If the area is small, it's likely a black mark
        if (50<area < 100):
            for j in (contour[0]):
                if cv2.pointPolygonTest(max_cont[0], tuple(j), False) >= 0:
                    # if so, draw the mark
                # print(max_cont)
                    cv2.drawContours(img, contour, -1, (0, 255, 0), 2)
                    break
        

def videodraw(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to obtain a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Apply a morphological opening to the binary image to remove noise
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Find contours in the opened image
    image, contours, _ = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Iterate through each contour and check if it is within another contour
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        # if (area<50):
        for j in range(i+1, len(contours)):
            # Check if the contour is within the other contour
            if cv2.pointPolygonTest(contours[j], tuple(contour[0][0]), False) >= 0:
                print("contour[0][0]", contour[0][0])

    # Display the image with the contours
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)