import os

import numpy as np
import cv2
import rasterio

from .stereo_pair import StereoPair 

class DisparityMapGenerator:
    def __init__(self, num_disp: int = 128, min_disp: int = -32, sad_window_size: int = 5):
        self.num_disp: int = num_disp # This must be a multiple of 16, try also 64 (maximum disparity - minimum disparity)      
        self.min_disp: int = min_disp # This can be negative, with a conversion factor of 1, use -16*2 (search start)
        self.sad_window_size: int = sad_window_size # This must be an odd number, typically 3, 5, or 7 (matched block size) 

        self.disp_array: np.ndarray | None = None
        self.final_disp: np.ndarray | None = None     


    def execute(self, reference_array: np.ndarray, target_array: np.ndarray, stereo_pair: StereoPair, rows: int, cols: int, ortho_res: float, level: int) -> None:
        print("DISPARITY MAP GENERATION")

        # Configure the semi-global block matching (SGBM) parameters
        cn = 1  # Grayscale images strictly contain 1 channel
        block_size = self.sad_window_size if self.sad_window_size > 0 else 3        
        p1 = 8 * cn * block_size * block_size # P1 and P2 control the smoothness of the disparity map 
        p2 = 40 * cn * block_size * block_size # P2 must be larger than P1

        sgbm = cv2.StereoSGBM_create(
            minDisparity=self.min_disp, 
            numDisparities=self.num_disp,
            blockSize=block_size,
            P1=p1,
            P2=p2,
            preFilterCap=63,
            uniquenessRatio=10,
            speckleWindowSize=200,
            speckleRange=2,
            disp12MaxDiff=1, # maximum allowed difference (in integer pixel units) in the left-right disparity check
            mode=cv2.StereoSGBM_MODE_SGBM  # 5-direction SGBM 
            # mode=cv2.StereoSGBM_MODE_HH # 8-direction mode for cleaner urban building structures!
        )

        # Execute the dense matching
        # sgbm.compute returns an array of type int16, where values are multiplied by 16 
        self.disp_array = sgbm.compute(reference_array, target_array)

        # Save the sub-pixel disparity map
        os.makedirs("/opt/data/ossim/output/disparity_maps/", exist_ok=True)
        float_disp_name = f"1_float_disparity_level_{level}_ref_{stereo_pair.id_reference}_tar_{stereo_pair.id_target}.tif"
        cv2.imwrite("/opt/data/ossim/output/disparity_maps/"+float_disp_name, self.disp_array)

        print(f"Float disparity maps are saved!\n")

        # Normalize the disparity to 8-bit (0-255) for visual debugging
        min_val, max_val, _, _ = cv2.minMaxLoc(self.disp_array)
        print(f"Disparity => Min: {min_val} | Max: {max_val}\n")

        diff = max_val - min_val
        scale = 255.0 / diff if diff > 0 else 1.0
        
        # Create the 8-bit visualization array
        disp_array_uint8 = np.clip((self.disp_array - min_val) * scale, 0, 255).astype(np.uint8)
        
        # Save the visualized disparity
        sgbm_disp_name = f"2_sgbm_disparity_level_{level}_ref_{stereo_pair.id_reference}_tar_{stereo_pair.id_target}.tif"
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

        rotated_float_disp_name = f"3_rotated_float_disparity_level_{level}_ref_{stereo_pair.id_reference}_tar_{stereo_pair.id_target}.tif"
        cv2.imwrite("/opt/data/ossim/output/disparity_maps/"+rotated_float_disp_name, self.disp_array)

        # Convert the array to float64 
        self.disp_array = self.disp_array.astype(np.float64)

        # Scale the values back to true pixels (/16.0) and convert to meters
        self.disp_array = ortho_res * (self.disp_array / 16.0) / stereo_pair.mean_conversion_factor

        # Vectorized outlier masking 
        # Any disparity value below this threshold is an occlusion/invalid pixel
        # original: mask_threshold = (self.min_disp + 0.5 - 1.0) / stereo_pair.mean_conversion_factor
        mask_threshold = (self.min_disp - 0.5) / stereo_pair.mean_conversion_factor 
        self.disp_array[self.disp_array < mask_threshold] = -9999.0

        self.disp_array[:10, :] = -9999.0   # Top edge
        self.disp_array[-10:, :] = -9999.0  # Bottom edge
        self.disp_array[:, :10] = -9999.0   # Left edge
        self.disp_array[:, -10:] = -9999.0  # Right edge

        # Cast the metric array to 32-bit float 
        self.final_disp = self.disp_array.astype(np.float32)

        disp_name = f"4_disparity_level_{level}_ref_{stereo_pair.id_reference}_tar_{stereo_pair.id_target}.tif"
        disp_path = f"/opt/data/ossim/output/disparity_maps/{disp_name}"

        # Get the path to the original reference ortho image
        reference_image_path = stereo_pair.ortho_reference_path        
        
        # Read the georeferencing directly from the original reference image
        with rasterio.open(reference_image_path) as reference_src:
            # Copy the reference image's metadata profile (CRS, bounds, etc.)
            profile = reference_src.profile
        
            profile.update(
                dtype=rasterio.float32,
                count=1,
                compress='lzw',
                nodata=-9999.0  # Tell QGIS to treat -9999.0 as transparent NoData!
            )
            
        with rasterio.open(disp_path, 'w', **profile) as dst:
            dst.write(self.final_disp, 1)

        print(f"Metric disparity map is saved and georeferenced: {disp_name}\n")