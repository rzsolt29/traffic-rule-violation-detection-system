import cv2
import numpy as np
import time

video = cv2.VideoCapture('random_motorway.mp4')
frames = []
start = time.time()
for frameOI in range(150):
    video.set(cv2.CAP_PROP_POS_FRAMES, frameOI)
    ret, frame = video.read()
    frames.append(frame)
    print(frameOI)
end = time.time()
print("eltelt ido:")
print(end-start)
#ez 28 fps-nek felel meg
result = np.median(frames, axis=0).astype(dtype=np.uint8)
x = 650 / result.shape[0]
y = x
result = cv2.resize(result, None, None, x, y, cv2.INTER_CUBIC)
cv2.imshow("Median filtering result",result)

gray = cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray image",gray)

blur = cv2.GaussianBlur(gray, (5,5), 0)
cv2.imshow("Gaussian blur",blur)

canny = cv2.Canny(blur, 50, 150)
cv2.imshow("Canny edge detector",canny)

cv2.waitKey(0)