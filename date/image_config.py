from dataclasses import dataclass
from functools import reduce

import pyproj
from shapely.ops import transform
from shapely.geometry import Polygon

import pyossim

@dataclass
class ImageConfig:
    number_steps: int = 2
    meters: float = 2.0
    min_lat: float = 0.0
    max_lat: float = 0.0
    min_lon: float = 0.0
    max_lon: float = 0.0
    operation: str = 'ortho'
    resampler_filter: str = 'box'
    projection: str = 'utm'


    @staticmethod
    def shrink_polygon_meters(polygon, meters):
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


    def calculate_extent_from_images(self, image_file_paths):
        """
        Find the intersection of all images in latitude and longitude, shrink the intersection by the preferred amount of meters, and then calculate the bounding box of the resulting polygon
        """

        registry = pyossim.ossimImageHandlerRegistry.instance()
        image_polygons = []

        for path in image_file_paths:
            handler = registry.open(path)
            if not handler:
                print(f"Warning: {path} cannot be opened.")
                continue
            
            geom = handler.getImageGeometry()
            if not geom:
                print(f"Warning: No geometry for {path}")
                continue

            image_pixel_dimensions = handler.getBoundingRect()
            width = image_pixel_dimensions.width()
            height = image_pixel_dimensions.height()

            # Define the 4 corners of the polygon corresponding to the image in pixel space (ossimDpt)
            # ul=(0,0), ur=(w,0), lr=(w,h), ll=(0,h)
            corners_pixels = [
                pyossim.ossimDpt(0, 0),
                pyossim.ossimDpt(width, 0),
                pyossim.ossimDpt(width, height),
                pyossim.ossimDpt(0, height)
            ]

            # Convert each pixel to a world point (ossimGpt)
            world_corners = []
            for corner_pixel in corners_pixels:
                corner_world = geom.localToWorld(corner_pixel)
                # Shapely needs (longitude, latitude)
                world_corners.append((corner_world.lon, corner_world.lat))

            # Create the Shapely polygon for this specific image
            image_polygons.append(Polygon(world_corners))

        try:
            # This intersects all polygons in the list one by one
            overlap = reduce(lambda p1, p2: p1.intersection(p2), image_polygons)
            
            if not overlap.is_empty:
                overlap_shrunk = self.shrink_polygon_meters(overlap, 200)

                min_lon, min_lat, max_lon, max_lat = overlap_shrunk.bounds
                
                self.min_lat = min_lat
                self.max_lat = max_lat
                self.min_lon = min_lon
                self.max_lon = max_lon

                print(f"\nBounding box (AABB) of the shrunk overlap in WKT: POLYGON(({self.min_lon} {self.min_lat}, {self.max_lon} {self.min_lat}, {self.max_lon} {self.max_lat}, {self.min_lon} {self.max_lat}, {self.min_lon} {self.min_lat}))")

                print(f"Extent is calculated for 3 images.")
                print(f"Min lat: {self.min_lat}\nMax lat: {self.max_lat}\nMin lon: {self.min_lon}\nMax lon: {self.max_lon}")                
            else:
                print("No common overlap is found.")
                self.min_lat = self.max_lat = self.min_lon = self.max_lon = None
        except Exception as e:
            print(f"Error calculating intersection: {e}")