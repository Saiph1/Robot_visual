import cv2
import numpy as np
import cv2.aruco as aruco

# Define the ArUco dictionary
aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

# Define the size of the markers in pixels and the number of markers per row and column
marker_size = 200
markers_per_row = 4
markers_per_column = 3

# Create the marker board using the drawPlanarBoard() function
board = aruco.GridBoard_create(markers_per_row, markers_per_column, marker_size, 0.5 * marker_size, aruco_dict)

# Define the size of the board in pixels
board_size = (markers_per_row + 1) * marker_size, (markers_per_column + 1) * marker_size

# Create the image of the board using the draw() function
img = board.draw(board_size)

# Save the image of the board
cv2.imwrite('board.png', img)