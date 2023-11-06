import cv2
import numpy as np
import math


def find_lanes(background):
    x = 650 / background.shape[0]
    y = x
    background = cv2.resize(background, None, None, x, y, cv2.INTER_CUBIC)
    background = cv2.GaussianBlur(background, (5, 5), 0)
    imgHeight = background.shape[0]
    imgWidth = background.shape[1]

    # polygon creation for masking
    stencil = np.zeros_like(background[:, :])
    polygon = np.array([[int(imgWidth//4.57), int(imgHeight//8.125)],
                        [int(imgWidth//1.464), int(imgHeight//8.125)],
                        [int(math.floor(imgWidth*1.366)), int(imgHeight)],
                       [int(math.floor(imgWidth//-3.66)), int(imgHeight)]])
    cv2.fillConvexPoly(stencil, polygon, 1)

    # using mask on the background image
    masked = cv2.bitwise_and(background[:, :], background[:, :], mask=stencil)

    # image thresholding to filter white colors from the grayscale image
    ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)


    # search for contours
    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    areaArray = []
    count = 1

    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        areaArray.append(area)

    sortedData = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

    # store the two largest contours
    twoLargestContours = [sortedData[0][1], sortedData[1][1]]

    x, y, w, h = cv2.boundingRect(twoLargestContours[0])
    cv2.drawContours(close, twoLargestContours[0], -1, (255, 0, 0), 1)
    cv2.rectangle(close, (x, y), (x+w, y+h), (0, 255, 0), 1)
    x, y, w, h = cv2.boundingRect(twoLargestContours[1])
    cv2.drawContours(close, twoLargestContours[1], -1, (255, 0, 0), 1)
    cv2.rectangle(close, (x, y), (x+w, y+h), (0, 255, 0), 1)

    segmentedLanes = close.copy()

    # mark the lanes without separation
    for i in range(81, close.shape[0] - 1):
        afterFistContour = False
        contourCounter = 0

        for j in range(close.shape[1] - 1):
            if close[i][j] == 255 and close[i][j+1] != 255:
                afterFistContour = True
                contourCounter += 1

            elif close[i][j] == 0 and afterFistContour and contourCounter < 3:
                  segmentedLanes[i][j] = 76

            elif segmentedLanes[i - 1][j] == 76 and segmentedLanes[i][j] == 0:
                  segmentedLanes[i][j] = 76

    # make everything black except the lanes
    for i in range(81, segmentedLanes.shape[0] - 1):
        for j in range(segmentedLanes.shape[1] - 1):
            if segmentedLanes[i][j] != 76:
                segmentedLanes[i][j] = 0


    # find the first lane
    laneChange = []
    lengthOfLastLane = 0
    lengthOfCurrentLane = 0
    for i in range(81, segmentedLanes.shape[0] - 1):
        lengthOfLastLane = lengthOfCurrentLane
        lengthOfCurrentLane = 0
        colorChangedOnce = False
        firstLaneReady = False
        for j in range(segmentedLanes.shape[1] - 2):

            if not colorChangedOnce and segmentedLanes[i][j] != segmentedLanes[i][j + 1]:
                colorChangedOnce = True
            elif colorChangedOnce and segmentedLanes[i][j] != 0 and segmentedLanes[i][j + 1] == 0:
                firstLaneReady = True
                laneChange.append(j)
                break
            elif lengthOfLastLane != 0 and lengthOfLastLane * 1.00001 < lengthOfCurrentLane:
                break
            elif colorChangedOnce and not firstLaneReady:
                segmentedLanes[i][j] = 255
                lengthOfCurrentLane += 1

    x = 3840 / segmentedLanes.shape[0]
    y = x
    segmentedLanes = cv2.resize(segmentedLanes, None, None, x, y, cv2.INTER_CUBIC)

    return segmentedLanes


if __name__ == "__main__":
    background = cv2.imread("result_pictures/newVideoSource/grayscaleFullSize.png")
    background = background[:, :, 0]
    find_lanes(background)
