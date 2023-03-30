import cv2
import os

### --- parameter --- ###

# Save location
dir_mark = './marker'

# Parameter
num_mark = 20 #Number of markers
size_mark = 500 #Size of markers

### --- marker images are generated and saved --- ###
# Call marker type
dict_aruco = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

for count in range(num_mark) :

    id_mark = count
    img_mark = cv2.aruco.drawMarker(dict_aruco, id_mark, size_mark)

    if count < 10 :
        img_name_mark = 'mark_id_0' + str(count) + '.jpg'
    else :
        img_name_mark = 'mark_id_' + str(count) + '.jpg'
    path_mark = os.path.join(dir_mark, img_name_mark)

    cv2.imwrite(path_mark, img_mark)