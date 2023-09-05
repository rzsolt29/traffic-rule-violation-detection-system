import cv2
import numpy as np

background = cv2.imread("result_pictures/newVideoSource/blured.png")

cv2.imshow("background", background)

# polynom creation for masking
stencil = np.zeros_like(background[:, :, 0])
polygon = np.array([[80, 50], [250, 50], [500, 650], [-100, 650]])
cv2.fillConvexPoly(stencil, polygon, 1)

# using mask on the background image
img = cv2.bitwise_and(background[:, :, 0], background[:, :, 0], mask=stencil)

# image thresholding to filter withe colors from the grayscale image
ret, thresh = cv2.threshold(img, 130, 145, cv2.THRESH_BINARY)
cv2.imshow("thresh", thresh)

# Canny edges detection
edges = cv2.Canny(thresh, 100, 200)
cv2.imshow("edges", edges)

outputImg = np.zeros(background.shape, dtype=np.uint8)

for i in range(thresh.shape[0] - 2):
    for j in range(thresh.shape[1] - 2):

        if abs(int(edges[i][j]) - int(edges[i + 1][j])) > 50:
            # bgr
            outputImg[i + 1][j][0] = 0
            outputImg[i + 1][j][1] = 100
            outputImg[i + 1][j][2] = 0

cv2.imshow("output", outputImg)

cv2.waitKey(0)
cv2.destroyAllWindows()
