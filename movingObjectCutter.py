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
        cv2.imshow("diff", difference)
        cv2.waitKey(0)

        frames.pop(0)
        frames.append(frame)
    else:
        raise Exception("Internal logic error")


video.release()
cv2.destroyAllWindows()
