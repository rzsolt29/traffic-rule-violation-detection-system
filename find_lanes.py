import cv2
import numpy as np

background = cv2.imread("result_pictures/newVideoSource/blured.png")

#cv2.imshow("background", background)

# polynom creation for masking
stencil = np.zeros_like(background[:, :, 0])
polygon = np.array([[80, 80], [250, 80], [500, 650], [-100, 650]])
cv2.fillConvexPoly(stencil, polygon, 1)

# using mask on the background image
masked = cv2.bitwise_and(background[:, :, 0], background[:, :, 0], mask=stencil)

# image thresholding to filter withe colors from the grayscale image
ret, thresh = cv2.threshold(masked, 130, 145, cv2.THRESH_BINARY)
#cv2.imshow("thresh", thresh)

####
####

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=5)

cv2.imshow("closed", close)

####
####

###########

contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

areaArray = []
count = 1

for i, c in enumerate(contours):
    area = cv2.contourArea(c)
    areaArray.append(area)

sorteddata = sorted(zip(areaArray, contours), key=lambda x: x[0], reverse=True)

biggests = [sorteddata[0][1], sorteddata[1][1]]

x, y, w, h = cv2.boundingRect(biggests[0])
cv2.drawContours(close, biggests[0], -1, (255, 0, 0), 1)
cv2.rectangle(close, (x, y), (x+w, y+h), (0,255,0), 1)
x, y, w, h = cv2.boundingRect(biggests[1])
cv2.drawContours(close, biggests[1], -1, (255, 0, 0), 1)
cv2.rectangle(close, (x, y), (x+w, y+h), (0,255,0), 1)
cv2.imshow("before", close)
after = cv2.cvtColor(close, cv2.COLOR_GRAY2BGRA)

for i in range(81, close.shape[0] - 1):
    afterFistContour = False
    contourCounter = 0
    for j in range(close.shape[1] - 1):
        if close[i][j] == 255 and close[i][j+1] != 255:
            afterFistContour = True
            contourCounter += 1
        elif close[i][j] == 0 and afterFistContour and contourCounter<3:
            after[i][j][0] = 0
            after[i][j][1] = 0
            after[i][j][2] = 255
            background[i][j][0] = 0
            background[i][j][1] = 0
            background[i][j][2] = 255
cv2.imshow("background", background)


"""counter = 0
for i in range(close.shape[0] - 1):
    counter=0
    for j in range(close.shape[1] - 1):
        if close[i][j] == 255 :
            counter += 1
        elif counter>1 and counter<3:
            after[i][j][0] = 0
            after[i][j][1] = 0
            after[i][j][2] = 255
        else:
            after[i][j][0] = 255
            after[i][j][1] = 0
            after[i][j][2] = 0"""


cv2.imshow("after", after)

###########

"""img = thresh.copy()

size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

ret, img = cv2.threshold(img, 127, 255, 0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
done = False

while (not done):
    eroded = cv2.erode(img, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(img, temp)
    skel = cv2.bitwise_or(skel, temp)
    img = eroded.copy()

    zeros = size - cv2.countNonZero(img)
    if zeros == size:
        done = True

cv2.imshow("skel", skel)

# Filter using contour area and remove small noise
cnts = cv2.findContours(skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 5500:
        cv2.drawContours(thresh, [c], -1, (0,0,0), -1)

# Morph close and invert image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
close = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, kernel, iterations=2)


cv2.imshow('close', close)


# Canny edges detection
edges = cv2.Canny(thresh, 100, 200)
cv2.imshow("edges", edges)
closedEdges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
cv2.imshow("closed edges", closedEdges)




closedEdgesEdges = cv2.Canny(closedEdges, 100, 200)
cv2.imshow("closed edges edges", closedEdgesEdges)

#####
size = np.size(closedEdges)
closedSkel = np.zeros(closedEdges.shape, np.uint8)

ret, closedEdges = cv2.threshold(closedEdges, 127, 255, 0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
done = False

while not done:
    eroded = cv2.erode(closedEdges, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(closedEdges, temp)
    closedSkel = cv2.bitwise_or(closedSkel, temp)
    closedEdges = eroded.copy()

    zeros = size - cv2.countNonZero(closedEdges)
    if zeros == size:
        done = True

cv2.imshow("closedSkel", closedSkel)"""


cv2.waitKey(0)
cv2.destroyAllWindows()
