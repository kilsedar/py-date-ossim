from dataclasses import dataclass

@dataclass
class RawImage:
    raw_image_id: int
    raw_image_path: str
    orbit: str