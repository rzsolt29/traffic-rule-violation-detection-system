from keepBackground import keep_background
from findLanes import find_lanes
from movingObjectCutter import moving_object_cutter

VIDEO_PATH = "test_video.mp4"

background = keep_background(VIDEO_PATH)
lanes = find_lanes(background)
moving_object_cutter(VIDEO_PATH, lanes)
