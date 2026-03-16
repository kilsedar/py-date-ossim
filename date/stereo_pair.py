import math
import pyossim

from dataclasses import dataclass

@dataclass
class StereoPair:
    id_master: int = 0
    id_slave: int = 0
    raw_master_path: str = ""
    raw_slave_path: str = ""
    # These two may need to be removed
    ortho_master_path: str = ""
    ortho_slave_path: str = ""
    # This may need to be calculated 
    delta_height: float = 0.0
    mean_rotation_angle: float = 0.0
    mean_conversion_factor: float = 0.0

    def set_ids(self, value_1: int, value_2: int):
        self.id_master = value_1
        self.id_slave = value_2

    def set_raw_paths(self, value_1: str, value_2: str):
        self.raw_master_path = value_1
        self.raw_slave_path = value_2

    # This may need to be removed
    def set_ortho_paths(self, value_1: str, value_2: str):
        self.ortho_master_path = value_1
        self.ortho_slave_path = value_2

    def epipolar_direction(self):
        print("EPIPOLAR DIRECTION COMPUTATION")

        grid = 10
        minimum_height= 500.0
        maximum_height= 1000.0
        self.delta_height = maximum_height - minimum_height

        registry = pyossim.ossimImageHandlerRegistry.instance()
        raw_master_handler = registry.open(self.raw_master_path)
        raw_slave_handler = registry.open(self.raw_slave_path)

        if not raw_master_handler or not raw_slave_handler:
            raise RuntimeError("Image files cannot be opened.")

        raw_master_geom = raw_master_handler.getImageGeometry()
        raw_slave_geom = raw_slave_handler.getImageGeometry()

        image_size = raw_master_geom.getImageSize()        
        width = image_size.x
        height = image_size.y    
        print(f"Master image dimensions: {width} x {height}")

        delta_I = image_size.x / (grid +1)
        print(f"delta I: {delta_I}")
        delta_J = image_size.y / (grid +1)
        print(f"delta J: {delta_J}")

        epipolar_direction_logs = open("date/logs/epipolar_direction.txt", "w")
        angles = []
        conversion_factors = []

        for i in range(1, grid + 1): # Columns (x-axis in image)
            for j in range(1, grid + 1): # Rows (y-axis in image)
                image_point_master = pyossim.ossimDpt(delta_I*i, delta_J*j)
                # print(f"Image point master: {image_point_master}")
                ground_point_master_up = raw_master_geom.localToWorld(image_point_master, maximum_height)
                ground_point_master_down = raw_master_geom.localToWorld(image_point_master, minimum_height)
                # print(f"Ground point master: {ground_point_master_up}")
                # print(f"Ground point down: {ground_point_master_down}")

                # Once I've computed the lowest point on the ground, I go to the slave's image plane
                image_point_slave_down = raw_slave_geom.worldToLocal(ground_point_master_down)
                # From the slave's image plane, I go to the highest point on the ground
                ground_point_slave_up = raw_slave_geom.localToWorld(image_point_slave_down, maximum_height)

                # Geographic --> UTM conversion
                utm_ground_point_master_up = pyossim.ossimUtmpt(ground_point_master_up)
                utm_ground_point_slave_up = pyossim.ossimUtmpt(ground_point_slave_up)

                epipolar_direction_logs.write(f"{utm_ground_point_master_up.easting():.12f} {utm_ground_point_master_up.northing():.12f} {utm_ground_point_slave_up.easting():.12f} {utm_ground_point_slave_up.northing():.12f}\n")

                # Calculate easting and northing differences in meters
                difference_easting = utm_ground_point_slave_up.easting() - utm_ground_point_master_up.easting()
                difference_northing = utm_ground_point_slave_up.northing() - utm_ground_point_master_up.northing()
                
                # Rotation angle calculation
                rotation_angle = math.atan2(difference_northing, difference_easting) * 180 / math.pi
                angles.append(rotation_angle)

                # Conversion factor calculation
                conversion_factor = math.sqrt((difference_northing**2) + (difference_easting**2)) / self.delta_height
                conversion_factors.append(conversion_factor)                

        self.mean_rotation_angle = sum(angles) / len(angles)
        self.mean_conversion_factor = sum(conversion_factors) / len(conversion_factors)

        epipolar_direction_logs.close()
                        