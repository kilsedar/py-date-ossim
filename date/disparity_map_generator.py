import os

import numpy as np
import cv2
import rasterio

from .stereo_pair import StereoPair 

class DisparityMapGenerator:
    def __init__(self):
        self.disp_array: np.ndarray | None = None
        self.final_disp: np.ndarray | None = None     


    def execute(self, reference_array: np.ndarray, target_array: np.ndarray, stereo_pair: StereoPair, rows: int, cols: int, ortho_res: float, level: int, reference_handler) -> None:
        print("DISPARITY MAP GENERATION")
        # print(reference_array.dtype)
        # print(target_array.dtype)

        # Configure the semi-global block matching (SGBM) parameters
        cn = 1  # Grayscale images strictly contain 1 channel
        sad_window_size = 5 # This must be an odd number, typically 3, 5, or 7 (used in place of the block size in P1 and P2 calculation as well) 
        p1 = 8 * cn * sad_window_size * sad_window_size # P1 and P2 control the smoothness of the disparity map 
        p2 = 40 * cn * sad_window_size * sad_window_size # P2 must be larger than P1

        if level == 0:     
            sgbm_mode = cv2.StereoSGBM_MODE_SGBM # 5-direction SGBM 
            min_disp = -16 
            num_disp = 64 
        elif level == 1:  
            sgbm_mode = cv2.StereoSGBM_MODE_HH # Full 8-direction matching for cleaner urban building outlines!
            min_disp = -16
            num_disp = 32
        else:            
            sgbm_mode = cv2.StereoSGBM_MODE_HH
            min_disp = -8
            num_disp = 16

        sgbm = cv2.StereoSGBM_create(
            minDisparity=min_disp, # This can be negative, with a conversion factor of 1, use -16*2 (search start) 
            numDisparities=num_disp, # This must be a multiple of 16, try also 64 (maximum disparity - minimum disparity)
            blockSize=sad_window_size, 
            P1=p1,
            P2=p2,
            preFilterCap=63,
            uniquenessRatio=5,
            speckleWindowSize=100,
            speckleRange=1,
            disp12MaxDiff=1, # Maximum allowed difference (in integer pixel units) in the left-right disparity check
            mode=sgbm_mode
        )

        # Execute the dense matching
        # sgbm.compute returns an array of type int16, where values are multiplied by 16 
        self.disp_array = sgbm.compute(reference_array, target_array)

        # Save the sub-pixel disparity map
        os.makedirs("/opt/data/ossim/output/disparity_maps/", exist_ok=True)
        float_disp_name = f"1_float_disparity_level_{level}_reference_{stereo_pair.id_reference}_target_{stereo_pair.id_target}.tif"
        cv2.imwrite("/opt/data/ossim/output/disparity_maps/"+float_disp_name, self.disp_array)

        print(f"Float disparity maps are saved!\n")

        # Normalize the disparity to 8-bit (0-255) for visual debugging
        # min_val, max_val, _, _ = cv2.minMaxLoc(self.disp_array) # Martina's solution
        valid_pixels = self.disp_array[self.disp_array > (min_disp - 0.5) * 16]
        min_val = np.percentile(valid_pixels, 5)
        max_val = np.percentile(valid_pixels, 95) 

        diff = max_val - min_val
        scale = 255.0 / diff if diff > 0 else 1.0
        
        # Create the 8-bit visualization array
        disp_array_uint8 = np.clip((self.disp_array - min_val) * scale, 0, 255).astype(np.uint8)
        
        # Save the visualized disparity
        sgbm_disp_name = f"2_uint8_disparity_level_{level}_reference_{stereo_pair.id_reference}_target_{stereo_pair.id_target}.tif"
        cv2.imwrite("/opt/data/ossim/output/disparity_maps/"+sgbm_disp_name, disp_array_uint8)

        print("Dense matching step is complete.")

        # Calculate rotation center (cols/2.0, rows/2.0)
        center = (self.disp_array.shape[1] / 2.0, self.disp_array.shape[0] / 2.0)
        
        # Get rotation matrix for rotating the image back around its center using the positive rotation angle
        rotation_matrix = cv2.getRotationMatrix2D(center, stereo_pair.mean_rotation_angle, 1.0)

        # Extract original dimensions of the disparity map
        h, w = self.disp_array.shape[:2]

        # Determine bounding rectangle
        rotated_box_points = cv2.boxPoints((center, (w, h), stereo_pair.mean_rotation_angle))
        _, _, bbox_w, bbox_h = cv2.boundingRect(rotated_box_points)

        # Adjust translation offsets in the rotation matrix
        rotation_matrix[0, 2] += (bbox_w / 2.0) - center[0]
        rotation_matrix[1, 2] += (bbox_h / 2.0) - center[1]

        # Rotate the disparity map
        # I used INTER_NEAREST instead of INTER_LINEAR default value
        self.disp_array = cv2.warpAffine(self.disp_array, rotation_matrix, (bbox_w, bbox_h), flags=cv2.INTER_NEAREST)

        # Find the top-left offset to crop out the original, unrotated image
        x_top_left = int((bbox_w - cols) / 2)
        y_top_left = int((bbox_h - rows) / 2)

        # Crop
        self.disp_array = self.disp_array[
            y_top_left : y_top_left + rows,
            x_top_left : x_top_left + cols
        ]

        rotated_float_disp_name = f"3_rotated_float_disparity_level_{level}_reference_{stereo_pair.id_reference}_target_{stereo_pair.id_target}.tif"
        cv2.imwrite("/opt/data/ossim/output/disparity_maps/"+rotated_float_disp_name, self.disp_array)

        # Convert the array to float64 
        self.disp_array = self.disp_array.astype(np.float64)

        # Scale the values back to true pixels (/16.0) and convert to meters
        self.disp_array = ortho_res * (self.disp_array / 16.0) / stereo_pair.mean_conversion_factor

        # Vectorized outlier masking 
        # Any disparity value below this threshold is an occlusion/invalid pixel
        # original: mask_threshold = (min_disp + 0.5 - 1.0) / stereo_pair.mean_conversion_factor
        mask_threshold = ortho_res * (min_disp - 0.5) / stereo_pair.mean_conversion_factor 
        self.disp_array[self.disp_array < mask_threshold] = -9999.0

        # Cast the metric array to 32-bit float 
        self.final_disp = self.disp_array.astype(np.float32)

        valid_mask = (self.final_disp >= -9000).astype(np.uint8)
        min_val, max_val, _, _ = cv2.minMaxLoc(self.final_disp, mask=valid_mask)
        print(f"Disparity => Min: {min_val} | Max: {max_val}\n")

        disp_name = f"4_final_disparity_level_{level}_reference_{stereo_pair.id_reference}_target_{stereo_pair.id_target}.tif"
        disp_path = f"/opt/data/ossim/output/disparity_maps/{disp_name}"

        # Get the path to the original reference ortho image
        reference_image_path = stereo_pair.ortho_reference_path        

        with rasterio.open(reference_image_path) as ref:
            src_crs = ref.crs
            src_transform = ref.transform
            width = ref.width
            height = ref.height

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": src_crs,
            "transform": src_transform,
            "nodata": -9999.0,
            "compress": "lzw",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
        }

        with rasterio.open(disp_path, "w", **profile) as dst:
            dst.write(self.final_disp, 1)

        geom_path = os.path.splitext(disp_path)[0] + ".geom"
        reference_geom = reference_handler.get_image_geometry()
        reference_geom.save_to_file(geom_path)

        print(f"Metric disparity map is saved and georeferenced: {disp_name}\n")