import cv2
import numpy as np

VIDEO_PATH = "random_motorway.mp4"

video = cv2.VideoCapture(VIDEO_PATH)
# Frames stores only two elements, two frames. The task is to get the differences, because those are the moving objects.
# Every time we have to override the values in this array so the structure will always be the same: last frame and current frame.
# Because of the vibration of the camera we have to make a threshold value which can avoid the False Positive detections.
frames = []

for frameOI in range(40):

    video.set(cv2.CAP_PROP_POS_FRAMES, frameOI)

    ret, frame = video.read()

    x = 650 / frame.shape[0]
    y = x
    frame = cv2.resize(frame, None, None, x, y, cv2.INTER_CUBIC)
    cv2.imshow("frame", frame)

    difference = np.copy(frame)

    if len(frames) == 0 or len(frames) == 1:
        frames.append(frame)
    elif len(frames) == 2:
        cv2.absdiff(frames[0], frames[1], difference)

        blurred = cv2.GaussianBlur(difference, (11, 11), 0)

        blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        ret, tframe = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow("thresholded", tframe)

        sample_frame = frames[1]
        (cnts, _) = cv2.findContours(tframe.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            x, y, w, h = cv2.boundingRect(cnt)
            if y > 200:  # Disregard item that are the top of the picture
                cv2.rectangle(sample_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("sample_frame", sample_frame)

        cv2.waitKey(0)

        frames.pop(0)
        frames.append(frame)
    else:
        raise Exception("Internal logic error")

video.release()
cv2.destroyAllWindows()
