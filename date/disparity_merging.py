import argparse
import os

import numpy as np
import cv2
import rasterio

import pyossim
from .tp_generator import TiePointsGenerator
from .disparity_map_generator import DisparityMapGenerator

class DisparityMerging:
    def __init__(self):
        # OpenCV matrices (cv::Mat) become NumPy arrays in Python
        self.disp_maps: list[np.ndarray] = []  
        self.merged_disp: np.ndarray | None = None   
        self.final_dsm: np.ndarray | None = None             
        
        self.reference_array: np.ndarray | None = None        
        self.target_array: np.ndarray | None = None           
        self.reference_array_uint8: np.ndarray | None = None 
        self.target_array_uint8: np.ndarray | None = None               
        
        self.reference_handler: pyossim.ossim_image_handler | None = None
        self.target_handler: pyossim.ossim_image_handler | None = None

        self.ortho_rows: int | None = None              
        self.ortho_cols: int | None = None     


    def execute(self, stereo_pairs_list: list, level: int, ortho_res: float) -> bool:
        self.disp_maps = [] # Clear the disparity maps list at the start of every single pyramid level so that the old levels don't pollute the new ones!
        pairs_number = len(stereo_pairs_list)        

        for n in range(pairs_number):
            pair = stereo_pairs_list[n]
            print(f"------------------------\n")
            print(f"PAIR TO PROCESS => Reference: {pair.id_reference} | Target: {pair.id_target}\n")

            reference_image_path = stereo_pairs_list[n].ortho_reference_path
            target_image_path = stereo_pairs_list[n].ortho_target_path

            # UTM settings
            reference_geom_path = os.path.splitext(reference_image_path)[0] + ".geom"            
            self._generate_rasterio_to_ossim_geom(reference_image_path, reference_geom_path)
            target_geom_path = os.path.splitext(target_image_path)[0] + ".geom"            
            self._generate_rasterio_to_ossim_geom(target_image_path, target_geom_path)

            registry = pyossim.ossim_image_handler_registry.instance()
            self.reference_handler = registry.open(reference_image_path)
            self.target_handler = registry.open(target_image_path)
          
            self.reference_handler.get_image_geometry()
            self.reference_handler.save_image_geometry()
            self.target_handler.get_image_geometry()
            self.target_handler.save_image_geometry()

            self._image_conversion_to_array(pair.ortho_reference_path, pair.ortho_target_path)

            # Get rotation matrix for rotating the image around its center
            # center: (x, y)
            center = (self.reference_array.shape[1] / 2.0, self.reference_array.shape[0] / 2.0)
            angle = -pair.mean_rotation_angle
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Determine bounding box
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            h, w = self.reference_array.shape[:2]
            bbox_w = int((h * sin_angle) + (w * cos_angle))
            bbox_h = int((h * cos_angle) + (w * sin_angle))

            # Adjust translation parameters in the rotation matrix
            rotation_matrix[0, 2] += (bbox_w / 2.0) - center[0]
            rotation_matrix[1, 2] += (bbox_h / 2.0) - center[1]

            # Apply the affine warp to both arrays
            self.reference_array = cv2.warpAffine(self.reference_array, rotation_matrix, (bbox_w, bbox_h))
            self.target_array = cv2.warpAffine(self.target_array, rotation_matrix, (bbox_w, bbox_h))

            os.makedirs("/opt/data/ossim/output/rotated_ortho_images/", exist_ok=True)
            cv2.imwrite(f"/opt/data/ossim/output/rotated_ortho_images/rotated_ortho_image_level_{level}_pair_{n}_reference_{pair.id_reference}.tif", self.reference_array)
            cv2.imwrite(f"/opt/data/ossim/output/rotated_ortho_images/rotated_ortho_image_level_{level}_pair_{n}_target_{pair.id_target}.tif", self.target_array)

            print(f"Rotated ortho images are saved!\n")

            self._image_conversion_to_uint8(pair.id_reference, pair.id_target, level)

            stereo_tp = TiePointsGenerator(self.reference_array_uint8, self.target_array_uint8)
            alignment_success = stereo_tp.execute(n, pair.id_reference, pair.id_target, level)
            if not alignment_success:
                print(f"WARNING: Image alignment has failed for pair {pair.id_reference} & {pair.id_target} at level {level}. Skipping disparity generation...")
                continue # Safely skip to the next loop iteration instead of crashing!

            disp_map_generator = DisparityMapGenerator()
            disp_map_generator.execute(self.reference_array_uint8, stereo_tp.target_array_warped, pair, self.ortho_rows, self.ortho_cols, ortho_res, level, self.reference_handler)
            self.disp_maps.append(disp_map_generator.disp_array)

        if self.disp_maps:
            # Initialize self.merged_disp as a copy of the first disparity map (Martina's solution)
            # self.merged_disp = np.copy(self.disp_maps[0])

            min_rows = min(disp_map.shape[0] for disp_map in self.disp_maps)
            min_cols = min(disp_map.shape[1] for disp_map in self.disp_maps)

            # Convert the -9999.0 NoData values to NaNs to do math easily
            valid_arrays = []
            for disp_map in self.disp_maps:
                # Crop to the common size to guarantee homogeneous shapes 
                disp_map_cropped = disp_map[:min_rows, :min_cols]
                disp_map_nan = np.where(disp_map_cropped == -9999.0, np.nan, disp_map_cropped)
                valid_arrays.append(disp_map_nan)

            # Compute the average across all disparity maps, safely ignoring NaNs 
            # This combines the data from all 3 images
            # Replaced with the following block!
            # disp_fused = np.nanmean(valid_arrays, axis=0) 

            # Extract the first two maps for error calculation
            disp_0_clean = valid_arrays[0]
            disp_1_clean = valid_arrays[1]
            error_disp = np.abs(disp_0_clean - disp_1_clean)

            all_maps_average = np.nanmean(valid_arrays, axis=0)

            # Vectorized conditional fusion 
            # If the difference is < 5 meters, we use the average of all maps
            # Otherwise, we default strictly to the second map (disp_1_clean)
            disp_fused = np.where(error_disp < 5.0, all_maps_average, disp_1_clean)

            # Convert NaNs back to the standard -9999.0 NoData value
            self.merged_disp = np.where(np.isnan(disp_fused), -9999.0, disp_fused)

            # Save the final fused disparity map 
            # Cast from float64 to float32 for standard GIS TIFF compatibility
            merged_disp_float32 = self.merged_disp.astype(np.float32)

            merged_disp_name = f"5_merged_disparity_level_{level}.tif"
            merged_disp_path = os.path.join("/opt/data/ossim/output/disparity_maps/", merged_disp_name)
            cv2.imwrite(merged_disp_path, merged_disp_float32)
            
            print(f"Saved fused disparity map for level {level}: {merged_disp_name}\n")

            return True
        else:
            print(f"ERROR: No disparity maps were successfully generated at level {level}. The entire level has failed.")
            return False
    

    def compute_dsm(self, args: argparse.Namespace, elev: pyossim.ossim_elev_manager, stereo_pairs_list: list, level: int) -> None:
        """
        Combine the metric disparity map with the coarse elevation model to generate the updated DSM
        """
        print(f"Computing DSM for level {level}...")

        # Delete the old temporary DSM from the previous pyramid level
        temp_dsm_dir = os.path.join(args.output_dir, "temp_dsm")
        temp_dsm_path = os.path.join(temp_dsm_dir, f"{args.output_filename}.tif")        
        if os.path.exists(temp_dsm_path):
            os.remove(temp_dsm_path)
            print(f"Old temporary DSM is removed: {temp_dsm_path}")        

        # Determine output path based on whether this is the final level or a temporary level 
        if level == 0:
            dsm_dir = os.path.join(args.output_dir, "dsm")            
        else:
            dsm_dir = temp_dsm_dir

        dsm_path = os.path.join(dsm_dir, f"{args.output_filename}.tif")

        reference_geom = self.reference_handler.get_image_geometry()

        # Create an 8-bit visualization of the merged disparity map before adding coarse elevation
        valid_mask_bool = self.merged_disp >= -9000 
        valid_mask = valid_mask_bool.astype(np.uint8)
        min_val, max_val, _, _ = cv2.minMaxLoc(self.merged_disp, mask=valid_mask)
        diff = max_val - min_val
        scale = 254.0 / diff if diff > 0 else 1.0        

        merged_disp_computed_0 = np.clip((self.merged_disp - min_val) * scale + 1, 1, 255).astype(np.uint8)
        merged_disp_computed_0[~valid_mask_bool] = 0
        
        debug_name_0 = f"6_merged_disparity_before_elevation_level_{level}.tif"
        debug_path_0 = os.path.join(args.output_dir, "disparity_maps", debug_name_0)
        cv2.imwrite(debug_path_0, merged_disp_computed_0)
        print(f"Saved pre-elevation debug image: {debug_name_0}")

        # We loop through every single pixel to add the base terrain elevation 
        h, w = self.merged_disp.shape[:2]

        for i in range(h): # Rows (y-axis)
            for j in range(w): # Columns (x-axis)
                # Image coordinate (x, y) = (j, i) 
                image_pt = pyossim.ossim_dpt(float(j), float(i))
                # print(image_pt)

                world_pt = reference_geom.local_to_world(image_pt)
                # print(world_pt)

                height_above_msl = elev.get_height_above_msl(world_pt)
                # print(height_above_msl)

                # Check if the pixel has a valid disparity value (not masked out)
                if self.merged_disp[i, j] >= -9000.0:
                    # Add the base MSL elevation to the pixel shift! 
                    self.merged_disp[i, j] += height_above_msl
                
                # Fill holes with the coarse DSM (only for intermediate levels, not the last one)
                elif level != 0:
                    self.merged_disp[i, j] = height_above_msl

        # Create an 8-bit visualization of the merged disparity map after adding coarse elevation
        min_val, max_val, _, _ = cv2.minMaxLoc(self.merged_disp, mask=valid_mask)
        diff = max_val - min_val
        scale = 254.0 / diff if diff > 0 else 1.0        

        merged_disp_computed_1 = np.clip((self.merged_disp - min_val) * scale + 1, 1, 255).astype(np.uint8)
        merged_disp_computed_1[~valid_mask_bool] = 0
        
        debug_name_1 = f"7_merged_disparity_after_elevation_level_{level}.tif"
        debug_path_1 = os.path.join(args.output_dir, "disparity_maps", debug_name_1)
        cv2.imwrite(debug_path_1, merged_disp_computed_1)
        print(f"Saved post-elevation debug image: {debug_name_1}")

        # Cast the metric array to 32-bit float 
        self.final_dsm = self.merged_disp.astype(np.float32)

        reference_image_path = stereo_pairs_list[-1].ortho_reference_path

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

        elev.clear()

        with rasterio.open(dsm_path, "w", **profile) as dst:
            dst.write(self.final_dsm, 1)

        geom_path = os.path.splitext(dsm_path)[0] + ".geom"
        reference_geom.save_to_file(geom_path)

        print(f"DSM is georeferenced and saved successfully: {dsm_path}\n")


    # =============================================
    # "Private" methods (prefixed with _ in Python)
    # =============================================

    def _generate_rasterio_to_ossim_geom(self, tif_path: str, geom_path: str) -> None:
        """
        Generate a minimal UTM zone 32N .geom file using rasterio
        """

        # https://rasterio.readthedocs.io/en/stable/topics/georeferencing.html
        with rasterio.open(tif_path) as src:
            transform = src.transform

        # Upper-Left tie point (Easting, Northing in meters)
        ul_x = transform.c  
        ul_y = transform.f  

        # Pixel resolution (meters per pixel)
        pixel_x = abs(transform.a)
        pixel_y = abs(transform.e)

        with open(geom_path, "w") as f:
            f.write("projection.type: ossimUtmProjection\n")
            f.write("projection.datum: WGE\n")
            f.write("projection.zone: 32\n")
            f.write("projection.hemisphere: N\n")
            f.write("projection.pixel_scale_units: meters\n")
            f.write(f"projection.pixel_scale_xy: ({pixel_x},{pixel_y})\n")
            f.write("projection.tie_point_units: meters\n")
            f.write(f"projection.tie_point_xy: ({ul_x},{ul_y})\n")
            f.write("type: ossimImageGeometry\n")
    

    def _image_conversion_to_array(self, ortho_reference_path: str, ortho_target_path: str) -> bool:
        """
        Open ortho images and convert them to OpenCV format (NumPy arrays)
        """

        self.reference_array = cv2.imread(ortho_reference_path, cv2.IMREAD_ANYDEPTH)
        self.target_array = cv2.imread(ortho_target_path, cv2.IMREAD_ANYDEPTH)

        if self.reference_array is None or self.target_array is None:
            print("ERROR: Images could not be read into NumPy arrays")
            return False

        self.ortho_rows, self.ortho_cols = self.reference_array.shape[:2]
        # print(self.reference_array)
        # print(self.ortho_rows, self.ortho_cols)
        # print(self.reference_array.dtype)
        # print(self.target_array.dtype)

        print(f"OSSIM->NumPy array conversion is done.\n")

        return True
    

    def _image_compute_histogram(self, image_id: int, image: np.ndarray, level: int, threshold: float) -> tuple:
        """
        Compute the histogram, trim off the top and bottom outlier tails by the percentage of the given threshold, return the new min and max
        """        
        print("Histogram computation =>")

        # Get min and max values of the image
        min_val, max_val, _, _ = cv2.minMaxLoc(image)
        print(f"Min: {min_val} | Max: {max_val}")

        # Establish the number of bins
        hist_size = int(max_val - min_val)
        if hist_size <= 0:
            return min_val, max_val
        
        # Compute the histogram, which holds the number of each value, ordered from lowest to highest
        hist = cv2.calcHist([image], [0], None, [hist_size], [min_val, max_val])
        print(f"Histogram size: {hist_size}")

        total_pixels = image.shape[0] * image.shape[1]
        print(f"Rows: {image.shape[0]} | Columns: {image.shape[1]} | Total number of pixels: {total_pixels}")

        # print(hist[785, 0])

        # Trim off the bottom percentage of values
        num_pixels = 0.0
        min_idx = 0
        while min_idx < hist_size:
            num_pixels += hist[min_idx][0]
            min_idx += 1
            if (num_pixels * 100.0 / total_pixels) > threshold:
                break

        # Trim off the top percentage of values 
        num_pixels = 0.0
        max_idx = hist_size
        while max_idx > 0:
            max_idx -= 1
            num_pixels += hist[max_idx][0]
            if (num_pixels * 100.0 / total_pixels) > threshold:
                break

        # Calculate the new min and max values
        new_min_val = min_val + min_idx
        new_max_val = min_val + max_idx        

        print(f"Index of the histogram at the 5th percetile: {min_idx}")
        print(f"Index of the histogram at the 95th percetile: {max_idx}")
        print(f"New min: {new_min_val} | New max: {new_max_val} \n")

        # Recalculate and draw the histogram for debugging
        hist_size_remap = int(new_max_val - new_min_val)
        if hist_size_remap > 0:
            hist_remap = cv2.calcHist([image], [0], None, [hist_size_remap], [new_min_val, new_max_val])
            
            hist_w, hist_h = 512, 400
            bin_w = int(round(hist_w / hist_size_remap)) if hist_size_remap > 0 else 1
            
            hist_image = np.full((hist_h, hist_w, 3), 255, dtype=np.uint8)
            hist_normalized = cv2.normalize(hist_remap, None, 0, hist_image.shape[0], cv2.NORM_MINMAX)
            
            for i in range(1, hist_size_remap):
                pt1 = (int(bin_w * (i - 1)), int(hist_h - round(float(hist_normalized[i - 1][0]))))
                pt2 = (int(bin_w * i), int(hist_h - round(float(hist_normalized[i][0]))))
                cv2.line(hist_image, pt1, pt2, (240, 40, 30), 1, 8, 0)
                    
            os.makedirs("/opt/data/ossim/output/histograms/", exist_ok=True)        
            histogram_image_name = f"histogram_level_{level}_image_id_{image_id}.png"
            cv2.imwrite("/opt/data/ossim/output/histograms/"+histogram_image_name, hist_image)

        return new_min_val, new_max_val

        # A potential shortcut to trim off the bottom and top 5% of pixels
        # min = np.percentile(image, threshold)
        # max = np.percentile(image, 100.0 - threshold)
        # return min, max


    def _image_conversion_to_uint8 (self, id_reference: int, id_target: int, level: int) -> None:
        """
        Convert the reference and target arrays to uint8 (8-bit unsigned integers)
        """

        threshold = 5.0

        # Compute histograms 
        min_val_reference, max_val_reference = self._image_compute_histogram(id_reference, self.reference_array, level, threshold)
        min_val_target, max_val_target = self._image_compute_histogram(id_target, self.target_array, level, threshold)

        print(f"Reference: {min_val_reference}, {max_val_reference} | Target: {min_val_target}, {max_val_target}\n")

        diff_reference = max_val_reference - min_val_reference
        scale_reference = 255.0 / diff_reference if diff_reference > 0 else 1.0
        diff_target = max_val_target - min_val_target
        scale_target = 255.0 / diff_target if diff_target > 0 else 1.0 

        # Perform the scaling, clip to [0, 255], and cast to uint8
        self.reference_array_uint8 = np.clip((self.reference_array - min_val_reference) * scale_reference, 0, 255).astype(np.uint8)
        self.target_array_uint8 = np.clip((self.target_array - min_val_target) * scale_target, 0, 255).astype(np.uint8)