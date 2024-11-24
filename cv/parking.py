# Following tutorial from https://www.youtube.com/watch?v=F-884J2mnOY&list=PLb49csYFtO2GunSHX38EjYHoYYbJYwrB-

import cv2
import numpy as np
import pickle
from skimage.transform import resize
import torch 
import torch.nn as nn
import torch.nn.functional as F

EMPTY = True
NOT_EMPTY = False

# MODEL = pickle.load(open("model.p", "rb"))

class ParkingModel(nn.Module):
    def __init__(self):
        super(ParkingModel, self).__init__()
        self.cnn1 = nn.Conv2d(3,16,kernel_size=3,stride=1, padding = 1)
        self.maxpool1 = nn.MaxPool2d(3, stride=2)
        self.cnn2 = nn.Conv2d(16,64,kernel_size=2,stride=1, padding = 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.linear = nn.Linear(64 * 8 * 17, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.maxpool1(F.relu(self.cnn1(x)))
        x = self.maxpool2(F.relu(self.cnn2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear(x))
        return self.linear2(x)


# Loading the model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ParkingModel().to(device)
model.load_state_dict(torch.load("parking_model_weights.pth", map_location=device))
model.eval()


# # Utils
def empty_or_not(spot_bgr):
    # Preprocess the input image
    img_resized = resize(spot_bgr, (15, 15, 3))
    img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # Get the model prediction
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)

    # Determine if the spot is empty or not
    if predicted.item() == 0:
        return EMPTY
    else:
        return NOT_EMPTY


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