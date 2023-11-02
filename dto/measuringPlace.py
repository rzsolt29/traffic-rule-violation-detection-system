class MeasuringPlace:
    def __init__(self, name_of_road, kilometric_point, direction, latitude, longitude):
        self.name_of_road = name_of_road
        self.kilometric_point = kilometric_point
        self.direction = direction
        self.latitude = latitude
        self.longitude = longitude

    def getValuesForDB(self):
        return str("'"+self.name_of_road)+"', "+str(self.kilometric_point)+", '"+str(self.direction)+"', "+"point("+str(self.latitude)+","+str(self.longitude)+")"
