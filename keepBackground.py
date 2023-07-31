import os
import cv2
import numpy as np

VIDEO_PATH = "random_motorway.mp4"
IMAGE_OUT_DIR_PATH = "violations"
DATASET_DIR = "dataset"

if not os.path.exists(IMAGE_OUT_DIR_PATH):
    os.mkdir(IMAGE_OUT_DIR_PATH)

video = cv2.VideoCapture(VIDEO_PATH)
frames = []

for frameOI in range(150):
    video.set(cv2.CAP_PROP_POS_FRAMES, frameOI)
    ret, frame = video.read()
    frames.append(frame)

result = np.median(frames, axis=0).astype(dtype=np.uint8)
x = 650 / result.shape[0]
y = x
result = cv2.resize(result, None, None, x, y, cv2.INTER_CUBIC)
cv2.imshow("Median filtering result", result)

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

cv2.imshow("background", blur)

cv2.waitKey(0)
cv2.destroyAllWindows()
