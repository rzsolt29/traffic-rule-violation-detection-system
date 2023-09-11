import cv2
import numpy as np

background = cv2.imread("result_pictures/newVideoSource/blured.png")

# polynom creation for masking
stencil = np.zeros_like(background[:, :, 0])
polygon = np.array([[80, 80], [250, 80], [500, 650], [-100, 650]])
cv2.fillConvexPoly(stencil, polygon, 1)

# using mask on the background image
masked = cv2.bitwise_and(background[:, :, 0], background[:, :, 0], mask=stencil)

# image thresholding to filter withe colors from the grayscale image
ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)
#cv2.imshow("thresh", thresh)


kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

cv2.imshow("closed", close)


contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

areaArray = []
count = 1

for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    areaArray.append(area)

sortedData = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

biggests = [sortedData[0][1], sortedData[1][1]]

x, y, w, h = cv2.boundingRect(biggests[0])
cv2.drawContours(close, biggests[0], -1, (255, 0, 0), 1)
cv2.rectangle(close, (x, y), (x+w, y+h), (0, 255, 0), 1)
x, y, w, h = cv2.boundingRect(biggests[1])
cv2.drawContours(close, biggests[1], -1, (255, 0, 0), 1)
cv2.rectangle(close, (x, y), (x+w, y+h), (0, 255, 0), 1)
cv2.imshow("before", close)
after = cv2.cvtColor(close, cv2.COLOR_GRAY2BGRA)

for i in range(81, close.shape[0] - 1):
    afterFistContour = False
    contourCounter = 0
    for j in range(close.shape[1] - 1):
        if close[i][j] == 255 and close[i][j+1] != 255:
            afterFistContour = True
            contourCounter += 1
        elif close[i][j] == 0 and afterFistContour and contourCounter < 3:
            after[i][j][0] = 0
            after[i][j][1] = 0
            after[i][j][2] = 255
            background[i][j][0] = 0
            background[i][j][1] = 0
            background[i][j][2] = 255
        elif after[i-1][j][0] == 0 and after[i][j][1] == 0 and after[i-1][j][2] == 255:
            after[i][j][0] = 0
            after[i][j][1] = 0
            after[i][j][2] = 255
            background[i][j][0] = 0
            background[i][j][1] = 0
            background[i][j][2] = 255
cv2.imshow("background", background)


for i in range(81, after.shape[0] - 1):
    for j in range(after.shape[1] - 1):
        if not(after[i][j][0] == 0 and after[i][j][1] == 0 and after[i][j][2] == 255):
            after[i][j][0] = 0
            after[i][j][1] = 0
            after[i][j][2] = 0


after = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

laneChange = []
lengthOfLastLane = 0
lengthOfCurrentLane = 0
for i in range(81, after.shape[0] - 1):
    lengthOfLastLane = lengthOfCurrentLane
    lengthOfCurrentLane = 0
    colorChangedOnce = False
    firstLaneReady = False
    for j in range(after.shape[1] - 2):

        if not colorChangedOnce and after[i][j] != after[i][j+1]:
            colorChangedOnce = True
        elif colorChangedOnce and after[i][j] != 0 and after[i][j+1] == 0:
            firstLaneReady = True
            laneChange.append(j)
            break
        elif lengthOfLastLane != 0 and lengthOfLastLane * 1.00001 < lengthOfCurrentLane:
            break
        elif colorChangedOnce and not firstLaneReady:
            after[i][j] = 255
            lengthOfCurrentLane += 1

cv2.imshow("after", after)

cv2.waitKey(0)
cv2.destroyAllWindows()
