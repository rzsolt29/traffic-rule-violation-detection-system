import os
import cv2
import psycopg
import datetime
from dto.measuringPlace import MeasuringPlace


def add_violation(image, measuring_place):
    with psycopg.connect("host="+os.environ['HOST']+" port="+os.environ['PORT']+" dbname="+os.environ['DBNAME']+" user="+os.environ['USER']+" password="+os.environ['PASSWORD']) as conn:
        with conn.cursor() as cur:

            dt = datetime.datetime.now()
            ts = datetime.datetime.timestamp(dt)

            img_path = "D:\\Dev\\Szakdoga\\traffic-rule-violation-detection-system\\violations\\"+str(ts).replace(".", "")+".jpg"

            cv2.imwrite(img_path, image)

            values = "'"+img_path+"', "+measuring_place.getValuesForDB()+", to_timestamp('"+str(dt.replace(microsecond=0))+"', 'yyyy-mm-dd hh24:mi:ss')"
            cur.execute("""
                        INSERT INTO illegal_overtakings (image_path, name_of_road, kilometric_point, direction, coordinates, time)
                        VALUES (
                        """+values+")")
            conn.commit()
    return


if __name__ == "__main__":
    img = cv2.imread("D:\\Dev\\Szakdoga\\traffic-rule-violation-detection-system\\test_img.png")
    measuring_place = MeasuringPlace("M44", 153, "Békéscsaba", 46.740207099204696, 20.818157445847874)
    add_violation(img, measuring_place)
