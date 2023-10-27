import cv2
import numpy as np
from imageClassifier import image_classifier

VIDEO_PATH = "test_video.mp4"


def moving_object_cutter():
    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # create a background object
    background_object = cv2.createBackgroundSubtractorMOG2(history=2)
    kernel = np.ones((3, 3), np.uint8)
    kernel2 = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = background_object.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 20, 255, cv2.THRESH_BINARY)

        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        fgmask = cv2.medianBlur(fgmask, 5)
        fgmask = cv2.dilate(fgmask, kernel2, iterations=6)

        # contour detection
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frameCopy = frame.copy()

        # loop inside the contour and search for bigger ones
        for cnt in contours:
            # get the area coordinates
            x, y, width, height = cv2.boundingRect(cnt)

            # 600 is enough to avoid False Positive detections with trucks but not too high to miss cars
            if cv2.contourArea(cnt) > 60000 and height > 600:

                # Following line draws a rectangle around the contour area. It's easier to test with this approach.
                #cv2.rectangle(frameCopy, (x,y), (x+width, y+height), (0, 0, 255), 2)

                img = frameCopy[y:y + height, x:x + width]
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

                # here we can call the object classifier function
                image_classifier(img)

        x = 950 / frame.shape[0]
        y = x
        fgmask = cv2.resize(fgmask, None, None, x, y, cv2.INTER_CUBIC)
        frameCopy = cv2.resize(frameCopy, None, None, x, y, cv2.INTER_CUBIC)

        #cv2.imshow("frameCopy", frameCopy)
        #cv2.imshow("fgmask", fgmask)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    moving_object_cutter()
