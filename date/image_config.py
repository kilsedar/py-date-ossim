from dataclasses import dataclass
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

    def calculate_bbox_from_images(self, image_file_paths):
        """
        Finds the intersection of all images in latitude and longitude
        """
        min_lat, max_lat = float('-inf'), float('inf')
        min_lon, max_lon = float('-inf'), float('inf')

        registry = pyossim.ossimImageHandlerRegistry.instance()

        for path in image_file_paths:
            handler = registry.open(path)
            if not handler:
                print(f"Warning: could not open {path}")
                continue
            
            geom = handler.getImageGeometry()
            if not geom:
                print(f"Warning: no geometry for {path}")
                continue
            
            # Use the custom C++ lambda binding to get the rectangle
            image_rect = geom.getGroundBoundingRect()

            if image_rect.hasNans():
                print(f"Warning: image {path} has invalid bounds")
                continue

            min_lat = max(min_lat, image_rect.ll.lat)
            max_lat = min(max_lat, image_rect.ur.lat)
            min_lon = max(min_lon, image_rect.ll.lon)
            max_lon = min(max_lon, image_rect.ur.lon)

        # Final assignment to the class attributes
        self.min_lat = min_lat
        self.max_lat = max_lat
        self.min_lon = min_lon
        self.max_lon = max_lon