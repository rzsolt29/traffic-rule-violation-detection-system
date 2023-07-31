import cv2
import numpy as np

VIDEO_PATH = "random_motorway.mp4"
video_stream = cv2.VideoCapture(VIDEO_PATH)

# Randomly select 30 frames
frameIds = video_stream.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=100)

frames = []
for fid in frameIds:
    video_stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = video_stream.read()
    x = 650 / frame.shape[0]
    y = x
    frame = cv2.resize(frame, None, None, x, y, cv2.INTER_CUBIC)
    frames.append(frame)

medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
cv2.imshow("background", medianFrame)
cv2.waitKey(0)

video_stream.release()
