import cv2
import numpy as np

image = cv2.imread("result_pictures/grayscaleimg.png")
outputImg = np.zeros((650, 1156, 3), dtype = np.uint8)

for i in range(image.shape[0]-2):
   for j in range(image.shape[1]-2):

        if abs(int(image[i][j][0])-int(image[i+1][j][0])) > 50:
            #bgr
            outputImg[i+1][j][0] = 0
            outputImg[i+1][j][1] = 100
            outputImg[i+1][j][2] = 0

cv2.imshow("", outputImg)

cv2.waitKey(0)
cv2.destroyAllWindows()