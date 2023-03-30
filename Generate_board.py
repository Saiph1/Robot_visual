import cv2
import numpy as np
import cv2.aruco as aruco

# Define the ArUco dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# Define the size of the markers in pixels and the number of markers per row and column
marker_size = 50
markers_per_row = 2
markers_per_column = 2

# Create the marker board using the drawPlanarBoard() function
board = aruco.GridBoard_create(markers_per_row, markers_per_column, marker_size, 200, aruco_dict)

# Define the size of the board in pixels
# Ipad: 2388x1668
board_size = 2388, 1688


# Create the image of the board using the draw() function
# Margin size determines the edge padding for the board in nums of pixels
img = board.draw(board_size, marginSize = 150)

# Save the image of the board
cv2.imwrite('./marker/board3.png', img)