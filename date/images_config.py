from functools import reduce

import pyproj
from shapely.ops import transform
from shapely.geometry import Polygon

import pyossim

class ImagesConfig:
    def __init__(self, number_levels: int = 2, meters: float = 1.0):
        self.number_levels: int = number_levels
        self.meters: float = meters

        self.min_lat: float | None = None
        self.max_lat: float | None = None
        self.min_lon: float | None = None
        self.max_lon: float | None = None


    @staticmethod
    def shrink_polygon_meters(polygon: Polygon, meters: float) -> Polygon | None:
        """
        polygon: Shapely polygon in latitude and longitude (EPSG:4326)
        meters: distance to shrink inward (positive number)
        """
        
        # The UTM zone is defined according to Trento
        project_to_meters = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform        
        project_to_degrees = pyproj.Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True).transform

        # Transform the polygon from degrees to meters
        polygon_meters = transform(project_to_meters, polygon)

        # Buffer inward 
        shrunk_polygon_meters = polygon_meters.buffer(-meters)

        # Safety check: If the polygon is too small, it might disappear
        if shrunk_polygon_meters.is_empty:
            print(f"Warning: Buffer of {meters}m made the polygon disappear. Consider using a smaller buffer.")
            return None

        # Transform back from meters to degrees
        shrunk_polygon_degrees = transform(project_to_degrees, shrunk_polygon_meters)

        return shrunk_polygon_degrees


    def calculate_extent_from_images(self, image_file_paths: list[str]) -> bool:
        """
        Find the intersection of all images in latitude and longitude, shrink the intersection by the preferred amount of meters, and then calculate the bounding box of the resulting polygon
        """

        registry = pyossim.ossim_image_handler_registry.instance()
        image_polygons = []

        for path in image_file_paths:
            handler = registry.open(path)
            if not handler:
                print(f"Warning: {path} cannot be opened.")
                continue
            
            geom = handler.get_image_geometry()
            if not geom:
                print(f"Warning: No geometry for {path}")
                continue

            image_pixel_dimensions = handler.get_bounding_rect()
            width = image_pixel_dimensions.width()
            height = image_pixel_dimensions.height()

            # Define the 4 corners of the polygon corresponding to the image in pixel space (ossim_dpt)
            # ul=(0,0), ur=(w,0), lr=(w,h), ll=(0,h)
            corners_pixels = [
                pyossim.ossim_dpt(0, 0),
                pyossim.ossim_dpt(width, 0),
                pyossim.ossim_dpt(width, height),
                pyossim.ossim_dpt(0, height)
            ]

            # Convert each pixel to a world point (ossim_gpt)
            world_corners = []
            for corner_pixel in corners_pixels:
                corner_world = geom.local_to_world(corner_pixel)
                # Shapely needs (longitude, latitude)
                world_corners.append((corner_world.lon, corner_world.lat))

            # Create the Shapely polygon for this specific image
            image_polygons.append(Polygon(world_corners))

        try:
            # This intersects all polygons in the list one by one
            overlap = reduce(lambda p1, p2: p1.intersection(p2), image_polygons)
            
            if not overlap.is_empty:
                overlap_shrunk = self.shrink_polygon_meters(overlap, 200)

                if overlap_shrunk is None: 
                    self.min_lat = self.max_lat = self.min_lon = self.max_lon = None  
                    return False

                self.min_lon, self.min_lat, self.max_lon, self.max_lat = overlap_shrunk.bounds

                print(f"\nExtent is calculated for 3 images.")
                print(f"Min lat: {self.min_lat}\nMax lat: {self.max_lat}\nMin lon: {self.min_lon}\nMax lon: {self.max_lon}")  
                print(f"Bounding box (AABB) of the shrunk overlap in WKT: POLYGON(({self.min_lon} {self.min_lat}, {self.max_lon} {self.min_lat}, {self.max_lon} {self.max_lat}, {self.min_lon} {self.max_lat}, {self.min_lon} {self.min_lat}))")      
                return True        
            else:
                self.min_lat = self.max_lat = self.min_lon = self.max_lon = None

                print("No common overlap is found.")                
                return False
        except Exception as e:
            print(f"Error calculating intersection: {e}")
            return False