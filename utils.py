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
    pixel_cm_ratio = aruco_perimeter / 20
    
    # Get Width and Height of the Objects by applying the Ratio pixel to cm
    object_width = w / pixel_cm_ratio
    object_height = h / pixel_cm_ratio
    cv2.putText(img, "Width :{} cm".format(round(object_width, 1)), (int(x-600), int(y+600)), cv2.FONT_HERSHEY_PLAIN, 10, (0, 200, 0), 10)
    cv2.putText(img, "Height :{} cm".format(round(object_height, 1)), (int(x-600), int(y+450)), cv2.FONT_HERSHEY_PLAIN, 10, (0, 200, 0), 10)