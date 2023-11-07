import sys
import datetime
import time

from database_operations.createTable import create_table
from dto.measuringPlace import MeasuringPlace
from keepBackground import keep_background
from findLanes import find_lanes
from movingObjectCutter import moving_object_cutter

if len(sys.argv) == 2:
    VIDEO_PATH = sys.argv[1]
else:
    VIDEO_PATH = 0

print("To start the program, give some necessary information about the control place")

name_of_road = input("Name of the road: ")
kilometric_point = input("Kilometric point: ")
direction = input("Direction of measuring lanes: ")
latitude = input("Geographic coordinates (latitude)")
longitude = input("Geographic coordinates (longitude)")
measuring_place = MeasuringPlace(name_of_road, kilometric_point, direction, latitude, longitude)

create_table()
print("Database table created")

try:
    background = keep_background(VIDEO_PATH)
except BaseException:
    print("Cannot open camera. Camera index out of range")
    exit()

print("Background created")

lanes = find_lanes(background)
print("Lanes found")

while True:
    if 6 < datetime.datetime.now().hour < 22:
        try:
            moving_object_cutter(VIDEO_PATH, lanes, measuring_place)
        except BaseException:
            print("Cannot open camera. Camera index out of range")
            exit()
    else:
        time.sleep(120)
        continue
