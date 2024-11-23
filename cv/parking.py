# Following tutorial from https://www.youtube.com/watch?v=F-884J2mnOY&list=PLb49csYFtO2GunSHX38EjYHoYYbJYwrB-

import cv2
import numpy as np
import pickle
from skimage.transform import resize

EMPTY = True
NOT_EMPTY = False

# MODEL = pickle.load(open("model.p", "rb"))

# # Utils
# def empty_or_not(spot_bgr):

#     flat_data = []

#     img_resized = resize(spot_bgr, (15, 15, 3))
#     flat_data.append(img_resized.flatten())
#     flat_data = np.array(flat_data)

#     y_output = MODEL.predict(flat_data)

#     if y_output == 0:
#         return EMPTY
#     else:
#         return NOT_EMPTY


def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components

    slots = []
    coef = 1
    for i in range(1, totalLabels):

        # Now extract the coordinate points
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)

        slots.append([x1, y1, w, h])

    return slots

### Utils end

video_path = "./data/parking_crop_loop.mp4"
mask_path = "./data/mask_crop.png"

cap = cv2.VideoCapture(video_path)
mask = cv2.imread(mask_path, 0)

connected = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S) # this is used to map pixels to individual parking spots

spots = get_parking_spots_bboxes(connected)

print(spots[0])

ret = True

while ret:
    ret, frame = cap.read()

    for spot in spots:
        x,y,w,h = spot

        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow('frame', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()