import cv2
import numpy as np


def keep_background(video_path):

    cap = cv2.VideoCapture(video_path)
    frames = []

    for frameOI in range(700):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frameOI)
        ret, frame = cap.read()

        if isinstance(video_path, int):
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if frameOI % 6 == 0:
            frames.append(frame)

    median = np.median(frames, axis=0).astype(dtype=np.uint8)

    #cv2.imwrite("result_pictures/newVideoSource/background.png", median)

    gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("result_pictures/newVideoSource/grayscale.png", gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    #cv2.imwrite("result_pictures/newVideoSource/blured.png", blur)

    cap.release()
    return blur


if __name__ == "__main__":
    video_path = "test_video.mp4"
    keep_background(video_path)
