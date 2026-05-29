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
        
        self.master_array = None           # NumPy array
        self.slave_array = None            # NumPy array
        self.master_array_uint8 = None     # NumPy array (uint8)
        self.slave_array_uint8 = None      # NumPy array (uint8)
        
        # OSSIM objects
        self.final_dsm = None              # pyossim.ossim_image_data
        self.master_handler = None         # pyossim.ossim_image_handler
        self.slave_handler = None          # pyossim.ossim_image_handler
        
        # Primitive types
        self.null_disp_threshold = 0.0     # float
        self.ortho_rows = 0                # int
        self.ortho_columns = 0             # int


    def execute(self, stereo_pairs_list: list, ortho_images_mask_list: list, images_list: list, ortho_res: float) -> bool:
        # registry = pyossim.ossimImageHandlerRegistry.instance()
        pairs_number = len(stereo_pairs_list)

        for n in range(pairs_number):
            pair = stereo_pairs_list[n]
            print(f"PAIR PROCESSED: master: {pair.id_master} | slave: {pair.id_slave}")

            # ortho_master_path = pair.ortho_master_path
            # ortho_slave_path = pair.ortho_slave_path
            # self.master_handler = registry.open(ortho_master_path)
            # self.slave_handler = registry.open(ortho_slave_path)

            self._img_conversion_to_array(pair.ortho_master_path, pair.ortho_slave_path)

            # Get rotation matrix for rotating the image around its center
            # center: (x, y)
            center = (self.master_array.shape[1] / 2.0, self.master_array.shape[0] / 2.0)
            angle = -pair.mean_rotation_angle
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            # print(rotation_matrix)

            # Determine bounding box
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            h, w = self.master_array.shape[:2]
            bbox_w = int((h * sin_angle) + (w * cos_angle))
            bbox_h = int((h * cos_angle) + (w * sin_angle))

            # Adjust translation parameters in the rotation matrix
            rotation_matrix[0, 2] += (bbox_w / 2.0) - center[0]
            rotation_matrix[1, 2] += (bbox_h / 2.0) - center[1]

            # Apply the affine warp (rotation and translation) to both arrays
            self.master_array = cv2.warpAffine(self.master_array, rotation_matrix, (bbox_w, bbox_h))
            self.slave_array = cv2.warpAffine(self.slave_array, rotation_matrix, (bbox_w, bbox_h))

            cv2.imwrite("/opt/data/ossim/output/rotated_master.tiff", self.master_array)
            cv2.imwrite("/opt/data/ossim/output/rotated_slave.tiff", self.slave_array)


    def get_merged_disp(self) -> np.ndarray:
        return self.merged_disp
    

    def compute_dsm(self, stereo_pairs_list: list, elev: pyossim.ossim_elev_manager, step: int, args: argparse.Namespace) -> bool:
        print(f"Computing DSM for step {step}...")
        return True
    

    # =============================================
    # "Private" methods (prefixed with _ in Python)
    # =============================================
    def _img_conversion_to_array(self, ortho_master_path: str, ortho_slave_path: str) -> bool:
        """
        Open ortho images and convert them to OpenCV format (NumPy arrays)
        """

        self.master_array = cv2.imread(ortho_master_path, cv2.IMREAD_ANYDEPTH)
        self.slave_array = cv2.imread(ortho_slave_path, cv2.IMREAD_ANYDEPTH)

        if self.master_array is None or self.slave_array is None:
            print("ERROR: Images could not be read into NumPy arrays")
            return False

        self.ortho_rows, self.ortho_columns = self.master_array.shape[:2]
        # print(self.master_array)
        # print(self.ortho_rows, self.ortho_columns)

        print(f"OSSIM->NumPy array conversion is done\n")

        return True
    

    def _img_get_histo(self, image: np.ndarray, threshold: float, min_histo: float, max_histo: float) -> bool:
        pass


    def _img_conversion_to_uint8(self) -> bool:
        """
        Convert internal arrays to 8-bit unsigned integers
        """
        pass