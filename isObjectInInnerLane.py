import cv2


def is_object_in_inner_lane(y, height, x, width, lanes):

    x_center = x+width//2
    y_center = y+height//2

    if lanes[y][x] == 76:
        return True
    else:
        return False


if __name__ == "__main__":
    lanes = cv2.imread("result_pictures/newVideoSource/laneLocalizationFullSize.png")
    lanes = lanes[:, :, 0]

    # outer lane test
    print(is_object_in_inner_lane(3819, 10, 32, 52, lanes))

    # inner lane test
    #print(is_object_in_inner_lane(1170, 10, 1170, 52, lanes))
