import cv2
import numpy as np
import cv2.aruco as aruco

# Read the image
img = cv2.imread('test3.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Define the ArUco dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# Define the ArUco parameters
aruco_params = aruco.DetectorParameters_create()

# Detect the ArUco markers in the image
corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

print("numbers of marker = ", len(ids))
print("corners = ", corners[5][0])

# Iterate through the markers
for i in range(len(ids)):
    # Get the corner coordinates of the marker
    # marker_corners = corners[i][0]

    # Calculate the orientation of the marker using the solvePnP function
    marker_length = 0.5 # Specify the length of the marker in meters
    camera_matrix = np.array([[3930.0, 0.0, 2040.0], [0.0, 3930.0, 1536.0], [0.0, 0.0, 1.0]]) # Specify the camera matrix
    dist_coeffs = np.zeros((4,1)) # Specify the distortion coefficients
    # ret, rvec, tvec = cv2.solvePnP(np.array([[0, 0, 0], [marker_length, 0, 0], [0, marker_length, 0], [0, 0, marker_length]]), marker_corners, camera_matrix, dist_coeffs)
    
    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, camera_matrix, dist_coeffs)
    (rvec - tvec).any()  # get rid of that nasty numpy value array error
    
    aruco.drawDetectedMarkers(img, corners)
    aruco.drawAxis(img, camera_matrix, dist_coeffs, rvec, tvec, 0.01)  # Draw Axis

# Display the image
img = cv2.resize(img, (960, 540))
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()