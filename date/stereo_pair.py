import ossim_wrapper

from dataclasses import dataclass

@dataclass
class StereoPair:
    id_master: int = 0
    id_slave: int = 0
    raw_master_path: str = ""
    raw_slave_path: str = ""
    ortho_master_path: str = ""
    ortho_slave_path: str = ""
    delta_height: float = 0.0
    mean_rotation_angle: float = 0.0
    mean_conversion_factor: float = 0.0

    def set_ids(self, value_1: int, value_2: int):
        self.id_master = value_1
        self.id_slave = value_2

    def set_raw_paths(self, value_1: str, value_2: str):
        self.raw_master_path = value_1
        self.raw_slave_path = value_2

    def set_ortho_paths(self, value_1: str, value_2: str):
        self.ortho_master_path = value_1
        self.ortho_slave_path = value_2

    def epipolar_direction(self):
        print("EPIPOLAR DIRECTION COMPUTATION")

        # ossim_wrapper.init() 

        registry = ossim_wrapper.ossimImageHandlerRegistry.instance()
        raw_master_handler = registry.open(self.raw_master_path)
        raw_slave_handler = registry.open(self.raw_slave_path)

        if not raw_master_handler or not raw_slave_handler:
            raise RuntimeError("Could not open image files.")

        # get geometries
        raw_master_geom = raw_master_handler.getImageGeometry()
        raw_slave_geom = raw_slave_handler.getImageGeometry()

        print(f"Master geometry acquired: {raw_master_geom}")
        print(f"Slave geometry acquired: {raw_slave_geom}")

        grid = 1 # change to 10
        # should these heights be determined dynamically?
        minimum_height= 500.0
        maximum_height= 1000.0
        self.delta_height = maximum_height - minimum_height
        image_size = raw_master_geom.getImageSize()
        
        # in OSSIM, for points/sizes, x is width (columns), y is height (rows)
        width = image_size.x
        height = image_size.y    

        print(f"Master image dimensions: {width} x {height}")

        delta_I = image_size.x / (grid +1)
        print(f"delta I: {delta_I}")
        delta_J = image_size.y / (grid +1)
        print(f"delta J: {delta_J}")

        epipolar_direction = open("date/logs/epipolar_direction.txt", "w")
        angles = []
        conversion_factors = []

        for i in range(1, grid + 1): # latitude
            for j in range(1, grid + 1): # longitude
                image_point_master = ossim_wrapper.ossimDpt(delta_I*i, delta_J*j)
                image_point_slave = ossim_wrapper.ossimDpt(0., 0.)
                ground_point_master = ossim_wrapper.ossimGpt(0., 0., maximum_height)
                ground_point_slave = ossim_wrapper.ossimGpt(0., 0., maximum_height)
                ground_point_down = ossim_wrapper.ossimGpt(0., 0., minimum_height)
                print(f"Image point master: {image_point_master}")
                print(f"Image point slave: {image_point_slave}")
                print(f"Ground point master: {ground_point_master}")
                print(f"Ground point slave: {ground_point_slave}")
                print(f"Ground point down: {ground_point_down}")
                # with this transformation, I obtain the ground_point_master
                ground_point_master = raw_master_geom.localToWorld(image_point_master, maximum_height)
                ground_point_down = raw_master_geom.localToWorld(image_point_master, minimum_height)
                print(f"Ground point master: {ground_point_master}")
                print(f"Ground point down: {ground_point_down}")

        raw_master_geom.printGeometry()

        