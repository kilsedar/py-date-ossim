import math
import pyossim

class StereoPair:
    def __init__(self, id_reference: int, id_target: int, raw_reference_path: str, raw_target_path: str):
        self.id_reference: int = id_reference
        self.id_target: int = id_target
        self.raw_reference_path: str = raw_reference_path
        self.raw_target_path: str = raw_target_path

        self.ortho_reference_path: str | None = None
        self.ortho_target_path: str | None = None
        
        self.delta_height: float | None = None
        self.mean_rotation_angle: float | None = None
        self.mean_conversion_factor: float | None = None
        

    def epipolar_direction(self) -> None:
        """
        Calculate mean rotation angle (epipolar direction) and mean conversion factor
        """
        print("EPIPOLAR DIRECTION COMPUTATION")

        grid = 10
        min_height= 500.0
        max_height= 1000.0
        self.delta_height = max_height - min_height

        registry = pyossim.ossim_image_handler_registry.instance()
        raw_reference_handler = registry.open(self.raw_reference_path)
        raw_target_handler = registry.open(self.raw_target_path)

        if not raw_reference_handler or not raw_target_handler:
            raise RuntimeError("Image files cannot be opened.")

        raw_reference_geom = raw_reference_handler.get_image_geometry()
        raw_target_geom = raw_target_handler.get_image_geometry()

        image_size = raw_reference_geom.get_image_size()        
        width = image_size.x
        height = image_size.y    
        print(f"Reference image dimensions: {width} x {height}")

        # Delta values are currently integer
        # Should we instead use double values? 
        delta_I = image_size.x // (grid +1)
        print(f"Delta I: {delta_I}")
        delta_J = image_size.y // (grid +1)
        print(f"Delta J: {delta_J}")

        epipolar_direction_logs = open("date/logs/epipolar_direction.txt", "w")
        angles = []
        conversion_factors = []

        for i in range(1, grid + 1): # Columns (x-axis in image)
            for j in range(1, grid + 1): # Rows (y-axis in image)
                image_point_reference = pyossim.ossim_dpt(delta_I*i, delta_J*j)
                # print(f"Image point reference: {image_point_reference}")
                ground_point_reference_up = raw_reference_geom.local_to_world(image_point_reference, max_height)
                ground_point_reference_down = raw_reference_geom.local_to_world(image_point_reference, min_height)
                # print(f"Ground point reference: {ground_point_reference_up}")
                # print(f"Ground point down: {ground_point_reference_down}")

                # Once I've computed the lowest point on the ground, I go to the target's image plane
                image_point_target_down = raw_target_geom.world_to_local(ground_point_reference_down)
                # From the target's image plane, I go to the highest point on the ground
                ground_point_target_up = raw_target_geom.local_to_world(image_point_target_down, max_height)

                # Geographic --> UTM conversion
                utm_ground_point_reference_up = pyossim.ossim_utmpt(ground_point_reference_up)
                utm_ground_point_target_up = pyossim.ossim_utmpt(ground_point_target_up)

                epipolar_direction_logs.write(f"{utm_ground_point_reference_up.easting:.12f} {utm_ground_point_reference_up.northing:.12f} {utm_ground_point_target_up.easting:.12f} {utm_ground_point_target_up.northing:.12f}\n")

                # Calculate easting and northing differences in meters
                difference_easting = utm_ground_point_target_up.easting - utm_ground_point_reference_up.easting
                difference_northing = utm_ground_point_target_up.northing - utm_ground_point_reference_up.northing
                
                # Rotation angle calculation
                rotation_angle = math.atan2(difference_northing, difference_easting) * 180 / math.pi
                angles.append(rotation_angle)

                # Conversion factor calculation
                conversion_factor = math.sqrt((difference_northing**2) + (difference_easting**2)) / self.delta_height
                conversion_factors.append(conversion_factor)                

        self.mean_rotation_angle = sum(angles) / len(angles)
        self.mean_conversion_factor = sum(conversion_factors) / len(conversion_factors)

        epipolar_direction_logs.close()
                        