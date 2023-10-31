import cv2


def is_object_in_inner_lane(y, height, x, width, lanes):
    x_center = x+width//2
    y_center = y+height//2

    if (lanes[y][x] == 255).any():
        return False
    else:
        return True


if __name__ == "__main__":
    lanes = cv2.imread("result_pictures/newVideoSource/laneLocalizationFullSize.png")
    print(is_object_in_inner_lane(3800, 10, 20, 45, lanes))
