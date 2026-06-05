import numpy as np
import cv2
import argparse
import pyossim

class DisparityMerging:
    def __init__(self):
        # OpenCV matrices (cv::Mat) become NumPy arrays in Python
        self.disp_maps = []                # list of NumPy arrays
        self.mask_ascending_tot = None     # NumPy array
        self.mask_descending_tot = None    # NumPy array
        self.merged_disp = None            # NumPy array
        
        self.reference_array = None           # NumPy array
        self.target_array = None            # NumPy array
        self.reference_array_uint8 = None     # NumPy array (uint8)
        self.target_array_uint8 = None      # NumPy array (uint8)
        
        # OSSIM objects
        self.final_dsm = None              # pyossim.ossim_image_data
        self.reference_handler = None         # pyossim.ossim_image_handler
        self.target_handler = None          # pyossim.ossim_image_handler
        
        # Primitive types
        self.null_disp_threshold = 0.0     # float
        self.ortho_rows = 0                # int
        self.ortho_columns = 0             # int


    def execute(self, stereo_pairs_list: list, step: int, ortho_res: float) -> bool:
        # registry = pyossim.ossimImageHandlerRegistry.instance()
        pairs_number = len(stereo_pairs_list)

        for n in range(pairs_number):
            pair = stereo_pairs_list[n]
            print(f"PAIR PROCESSED => Reference: {pair.id_reference} | Target: {pair.id_target}\n")

            # ortho_reference_path = pair.ortho_reference_path
            # ortho_target_path = pair.ortho_target_path
            # self.reference_handler = registry.open(ortho_reference_path)
            # self.target_handler = registry.open(ortho_target_path)

            self._img_conversion_to_array(pair.ortho_reference_path, pair.ortho_target_path)

            # Get rotation matrix for rotating the image around its center
            # center: (x, y)
            center = (self.reference_array.shape[1] / 2.0, self.reference_array.shape[0] / 2.0)
            angle = -pair.mean_rotation_angle
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            # print(rotation_matrix)

            # Determine bounding box
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            h, w = self.reference_array.shape[:2]
            bbox_w = int((h * sin_angle) + (w * cos_angle))
            bbox_h = int((h * cos_angle) + (w * sin_angle))

            # Adjust translation parameters in the rotation matrix
            rotation_matrix[0, 2] += (bbox_w / 2.0) - center[0]
            rotation_matrix[1, 2] += (bbox_h / 2.0) - center[1]

            # Apply the affine warp (rotation and translation) to both arrays
            self.reference_array = cv2.warpAffine(self.reference_array, rotation_matrix, (bbox_w, bbox_h))
            self.target_array = cv2.warpAffine(self.target_array, rotation_matrix, (bbox_w, bbox_h))

            cv2.imwrite("/opt/data/ossim/output/rotated_reference.tiff", self.reference_array)
            cv2.imwrite("/opt/data/ossim/output/rotated_target.tiff", self.target_array)

            self._img_conversion_to_uint8(pair.id_reference, pair.id_target, step)


    def get_merged_disp(self) -> np.ndarray:
        return self.merged_disp
    

    def compute_dsm(self, args: argparse.Namespace, elev: pyossim.ossim_elev_manager, step: int) -> bool:
        print(f"Computing DSM for step {step}...")
        return True
    

    # =============================================
    # "Private" methods (prefixed with _ in Python)
    # =============================================
    def _img_conversion_to_array(self, ortho_reference_path: str, ortho_target_path: str) -> bool:
        """
        Open ortho images and convert them to OpenCV format (NumPy arrays)
        """

        self.reference_array = cv2.imread(ortho_reference_path, cv2.IMREAD_ANYDEPTH)
        self.target_array = cv2.imread(ortho_target_path, cv2.IMREAD_ANYDEPTH)

        if self.reference_array is None or self.target_array is None:
            print("ERROR: Images could not be read into NumPy arrays")
            return False

        self.ortho_rows, self.ortho_columns = self.reference_array.shape[:2]
        # print(self.reference_array)
        # print(self.ortho_rows, self.ortho_columns)

        print(f"OSSIM->NumPy array conversion is done\n")

        return True
    

    def _img_compute_histogram(self, image_id: int, image: np.ndarray, step: int, threshold: float) -> tuple:
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
        
        # Compute the histogram, which holds the number of each value, ordered from lowest to heightest
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
                    
            histogram_image_name = f"histogram_level_{step}_image_{image_id}.png"
            cv2.imwrite("/opt/data/ossim/output/"+histogram_image_name, hist_image)

        return new_min_val, new_max_val

        # A potential shortcut to trim off the bottom and top 5% of pixels
        # min = np.percentile(image, threshold)
        # max = np.percentile(image, 100.0 - threshold)
        # return min, max


    def _img_conversion_to_uint8 (self, id_reference: int, id_target: int, step: int) -> bool:
        """
        Convert the reference and target arrays to uint8 (8-bit unsigned integers)
        """

        threshold = 5.0

        # Compute histograms 
        min_val_reference, max_val_reference = self._img_compute_histogram(id_reference, self.reference_array, step, threshold)
        min_val_target, max_val_target = self._img_compute_histogram(id_target, self.target_array, step, threshold)

        print(f"Reference: {min_val_reference}, {max_val_reference} | Target: {min_val_target}, {max_val_target}\n")
        print(f"*************************************\n")

        diff_reference = max_val_reference - min_val_reference
        scale_reference = 255.0 / diff_reference if diff_reference > 0 else 1.0
        diff_target = max_val_target - min_val_target
        scale_target = 255.0 / diff_target if diff_target > 0 else 1.0 

        # Perform the scaling, clip to [0, 255], and cast to uint8
        self.reference_array_8U = np.clip((self.reference_array - min_val_reference) * scale_reference, 0, 255).astype(np.uint8)
        self.target_array_8U  = np.clip((self.target_array - min_val_target) * scale_target, 0, 255).astype(np.uint8)

        return True