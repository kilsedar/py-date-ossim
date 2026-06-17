import os
import math

import numpy as np
import cv2

class TiePointsGenerator:
    def __init__(self, reference_array: np.ndarray, target_array: np.ndarray):
        self.reference_array: np.ndarray = reference_array
        self.target_array: np.ndarray = target_array
        self.target_array_warped: np.ndarray | None = None

        self.transformation_matrix: np.ndarray | None = None
        
        self.keypoints_reference: list = []
        self.keypoints_target: list = []
        self.good_matches: list = []
        
        # Average coordinates of the matches
        self.reference_x: float | None = None
        self.reference_y: float | None = None
        self.target_x: float | None = None
        self.target_y: float | None = None


    def execute(self, id_reference: int, id_target: int, level: int) -> bool:
        """
        Generate, draw, and warp coordinates
        """

        if not self._tp_gen():
            print("ERROR: Keypoint generation failed.")
            return False
        
        self._tp_draw(id_reference, id_target, level)

        success = self._tp_warp(id_reference, id_target, level)

        return success


    # =============================================
    # "Private" methods (prefixed with _ in Python)
    # =============================================

    def _tp_gen(self) -> bool:
        """
        Detect keypoints and match them between reference and target
        """
        print("TIE POINTS GENERATION")

        # Apply CLAHE (contrast-limited adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=8.0)
        self.reference_array = clahe.apply(self.reference_array)
        self.target_array = clahe.apply(self.target_array)

        # Feature detection and extraction
        # In modern OpenCV, ORB computes rotated BRIEF descriptors natively (The legacy code used a 5x5 grid with 500 points each (max 12500 features))
        orb = cv2.ORB_create(nfeatures=12500)

        # Detect keypoints and compute descriptors simultaneously
        self.keypoints_reference, descriptors_reference = orb.detectAndCompute(self.reference_array, None)
        self.keypoints_target, descriptors_target = orb.detectAndCompute(self.target_array, None)

        print(f"Features found: {len(self.keypoints_reference)} in reference, {len(self.keypoints_target)} in target") 

        if descriptors_reference is None or descriptors_target is None:
            print("ERROR: No descriptors are found!")
            return False

        # Match descriptors
        # I use NORM_HAMMING instead of NORM_L2!
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors_reference, descriptors_target)

        if not matches:
            print("ERROR: No matches were found!")
            return False
        
        print(f"{len(matches)} matches!")

        # Extract the visual similarity distances of all matches
        distances = [m.distance for m in matches]
        min_dist = min(distances)
        max_dist = max(distances)

        print(f"Initial maximum match distance: {max_dist}")
        print(f"Initial minimum match distance: {min_dist}")

        # Perform a numerical binary search (bisection) to find the exact distance threshold that retains 1% of the matches and discards the rest
        total_reference_features = len(self.keypoints_reference)
        distance_cutoff_threshold = (max_dist + min_dist) / 2.0
        retained_matches_ratio = 1.0  # placeholder that represents 100%
        search_iter = 0

        while abs(retained_matches_ratio - 0.01) > 0.001 and search_iter <= 200:
            retained_matches_count = 0
            for match in matches:
                if match.distance <= distance_cutoff_threshold:
                    retained_matches_count += 1
            
            # Calculate the ratio of matches we are keeping
            retained_matches_ratio = retained_matches_count / total_reference_features if total_reference_features > 0 else 0.0
            
            if retained_matches_ratio >= 0.01:
                # We kept too many matches, so we shrink the threshold ceiling
                max_dist = distance_cutoff_threshold
            else:
                # We kept too few matches, so we raise the threshold floor
                min_dist = distance_cutoff_threshold
                
            distance_cutoff_threshold = (max_dist + min_dist) / 2.0
            search_iter += 1
            
        print(f"Final distance cutoff threshold: {distance_cutoff_threshold}\n")

        # Initial parallax filtering (vertical displacement < 100 pixels) 
        for match in matches:
            # Only process the matches that pass the distance threshold
            if match.distance <= distance_cutoff_threshold:
                
                # Query index maps to the reference image keypoints (queryIdx: index of the matched feature in the reference (query) image)
                # Train index maps to the target image keypoints (trainIdx: index of the matched feature in the target (train) image)
                reference_pixel_coords = self.keypoints_reference[match.queryIdx].pt 
                target_pixel_coords = self.keypoints_target[match.trainIdx].pt

                # Calculate the vertical shift (parallax) in pixels (pt[0]: x, pt[1]: y)
                vertical_parallax = reference_pixel_coords[1] - target_pixel_coords[1]

                # Limit initial vertical displacement to 100 pixels
                # Since the images are aligned, vertical parallax should be close to 0!
                if abs(vertical_parallax) < 100.0:
                    self.good_matches.append(match)

        print(f"Number of good matches after removing the most distant features (99%) and features with vertical parallax in y-axis > 100: {len(self.good_matches)}")
        print(f"Percentage of points found: {((len(self.good_matches) / len(matches)) * 100.0):.1f}\n")
       
        # The 3-sigma iterative outlier test
        print("3-sigma test =>")
        sigma_test_iter = 0

        while True:
            sigma_test_iter += 1
            outliers_detected = False

            print(f"Iteration: {sigma_test_iter}")

            if not self.good_matches:
                print("Warning: No matches are left during the 3-sigma test!")
                break

            # Collect the vertical parallax (y-displacement) for every remaining good match
            vertical_parallax_values = []
            for match in self.good_matches:
                reference_pixel_y = self.keypoints_reference[match.queryIdx].pt[1]
                target_pixel_y = self.keypoints_target[match.trainIdx].pt[1]
                
                vertical_parallax = reference_pixel_y - target_pixel_y
                vertical_parallax_values.append(vertical_parallax)

            # Convert to a NumPy array for fast statistical calculations
            vertical_parallax_array = np.array(vertical_parallax_values, dtype=np.float64)

            # Calculate the mean and standard deviation of the vertical parallax
            parallax_mean = np.mean(vertical_parallax_array)
            parallax_std_dev = np.std(vertical_parallax_array) 
            if parallax_std_dev < 1e-9:
                print("Standard deviation is 0; no outliers remain. Skipping 3-sigma trim...")
                break

            print(f"Parallax standard deviation: {parallax_std_dev}")
            print(f"Parallax mean: {parallax_mean}")

            lower_bound = parallax_mean - (3.0 * parallax_std_dev)
            upper_bound = parallax_mean + (3.0 * parallax_std_dev)
            statistically_valid_matches = []       

            # Filter out any matches that lie outside the 3 standard deviations of the mean      
            for match in self.good_matches:
                reference_pixel_y = self.keypoints_reference[match.queryIdx].pt[1]
                target_pixel_y = self.keypoints_target[match.trainIdx].pt[1]
                current_vertical_parallax = reference_pixel_y - target_pixel_y

                # If the parallax lies within the 3-sigma bounds, we keep the match 
                if lower_bound < current_vertical_parallax < upper_bound:
                    statistically_valid_matches.append(match)
                else:
                    # An outlier was found! We flag that we need to run another iteration
                    outliers_detected = True

            # Update the good matches list with the newly filtered, statistically valid ones
            self.good_matches = statistically_valid_matches

            # If an entire iteration runs without finding a single outlier, we are done!
            if not outliers_detected:
                break

        print(f"\nNumber of good points found after the 3-sigma test: {len(self.good_matches)}\n")   

        if len(self.good_matches) < 3:
            print("ERROR: Not enough statistically valid matches survived the 3-sigma test.")
            return False

        return True   


    def _tp_draw(self, id_reference: int, id_target: int, level: int) -> None:
        """
        Draw the matched keypoints on a debug canvas and save it
        """
        print("Drawing the [filtered matched features (CV term)] / [tie points (photogrammetry term)]...")

        # cv2.drawMatches draws the reference and target images side by side and connects the matching keypoints with colored lines
        # https://docs.opencv.org/4.13.0/d4/d5d/group__features2d__draw.html#gad8f463ccaf0dc6f61083aca581df14dd
        image_matches = cv2.drawMatches(
            self.reference_array, self.keypoints_reference,
            self.target_array, self.keypoints_target,
            self.good_matches, 
            None,
            matchColor=(-1, -1, -1), # random colors for matched keypoints
            singlePointColor=(-1, -1, -1), # random colors for keypoints without matches
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        os.makedirs("/opt/data/ossim/output/tie_points/", exist_ok=True)
        tp_image_name = f"tp_level_{level}_ref_{id_reference}_tar_{id_target}.png"
        cv2.imwrite("/opt/data/ossim/output/tie_points/"+tp_image_name, image_matches)
        
        print(f"Tie points visualization is saved!\n")


    def _estimate_transformation_matrix(self, reference_points: list, target_points: list) -> np.ndarray:
        """
        Estimate the quasi-epipolar rotation & translation 2x3 affine matrix
        """

        num_matches = len(reference_points)
        if num_matches != len(target_points):
            raise ValueError("Reference and target point lists must have the same size.")

        # Extract coordinates into 1D NumPy arrays
        reference_pixel_x_coords = np.array([p[0] for p in reference_points], dtype=np.float64)
        reference_pixel_y_coords = np.array([p[1] for p in reference_points], dtype=np.float64)
        target_pixel_x_coords = np.array([p[0] for p in target_points], dtype=np.float64)
        target_pixel_y_coords = np.array([p[1] for p in target_points], dtype=np.float64)

        # Compute means (barycenters)
        self.reference_x = float(np.mean(reference_pixel_x_coords))
        self.reference_y = float(np.mean(reference_pixel_y_coords))
        self.target_x = float(np.mean(target_pixel_x_coords))
        self.target_y = float(np.mean(target_pixel_y_coords))

        # Calculate coordinate shifts
        horizontal_shifts = reference_pixel_x_coords - target_pixel_x_coords
        vertical_shifts = reference_pixel_y_coords - target_pixel_y_coords

        # Calculate standard deviation of coordinate shifts
        std_dev_horizontal_shifts = float(np.std(horizontal_shifts, ddof=0))
        std_dev_vertical_shifts = float(np.std(vertical_shifts, ddof=0))

        # Print out the stats
        print(f"Mean of reference pixel x coordinates: {self.reference_x}")
        print(f"Mean of reference pixel y coordinates: {self.reference_y}")
        print(f"Mean of target pixel x coordinates: {self.target_x}")
        print(f"Mean of target pixel y coordinates: {self.target_y}")
        print(f"Shift in x: {self.reference_x - self.target_x} pixels")
        print(f"Shift in y: {self.reference_y - self.target_y} pixels")
        print(f"Standard deviation of shift in x: {std_dev_horizontal_shifts}")
        print(f"Standard deviation of shift in y: {std_dev_vertical_shifts}\n")

        # Calculate barycentric (centered) coordinates
        centered_reference_x = reference_pixel_x_coords - self.reference_x
        centered_reference_y = reference_pixel_y_coords - self.reference_y
        centered_target_x = target_pixel_x_coords - self.target_x
        centered_target_y = target_pixel_y_coords - self.target_y

        # Rigorous non-linear SVD (singular value decomposition) 
        # SVD is the mathematical tool that calculates the best, most consistent average rotation and translation matrix that fits all of the matching points as closely as possible
        # In least-squares adjustment, we solve: design_matrix * adjustment_vector = observation_vector => adjustment_vector in the last iteration is extremely close to 0, unknowns_approximation holds the rotation angle (in radians) and vertical translation (in pixels) values
        unknowns_approximation = np.zeros((2 + num_matches, 1), dtype=np.float64)
        design_matrix = np.zeros((2 * num_matches, 2 + num_matches), dtype=np.float64)
        observation_vector = np.zeros((2 * num_matches, 1), dtype=np.float64)

        mean_horizontal_shift = self.reference_x - self.target_x

        for _ in range(3):
            for i in range(num_matches):
                # Populate row 2*i of the design matrix
                design_matrix[2 * i, 0] = centered_target_y[i]
                design_matrix[2 * i, 1] = 0.0
                design_matrix[2 * i, 2 + i] = 1.0

                # Populate row 2*i+1 of the design matrix
                design_matrix[2 * i + 1, 0] = -centered_target_x[i] - mean_horizontal_shift
                design_matrix[2 * i + 1, 1] = 1.0
                design_matrix[2 * i + 1, 2 + i] = 0.0

                # Extract current approximation values to evaluate the non-linear observation vector
                estimated_rotation_angle = unknowns_approximation[0, 0]
                estimated_vertical_translation = unknowns_approximation[1, 0]
                estimated_z_offset = unknowns_approximation[2 + i, 0]

                # Populate row 2*i of the observation vector
                observation_vector[2 * i, 0] = (
                    centered_reference_x[i] 
                    - math.cos(estimated_rotation_angle) * (centered_target_x[i] + mean_horizontal_shift + estimated_z_offset) 
                    - math.sin(estimated_rotation_angle) * centered_target_y[i]
                )
                
                # Populate row 2*i+1 of the observation vector
                observation_vector[2 * i + 1, 0] = (
                    centered_reference_y[i] 
                    + math.sin(estimated_rotation_angle) * (centered_target_x[i] + mean_horizontal_shift + estimated_z_offset) 
                    - math.cos(estimated_rotation_angle) * centered_target_y[i] 
                    - estimated_vertical_translation
                )

            # Solve the linear system using SVD decomposition
            success, adjustment_vector = cv2.solve(design_matrix, observation_vector, flags=cv2.DECOMP_SVD)
            
            if not success:
                print("Warning: SVD decomposition failed to converge.")
            
            # Update the approximations with the newly calculated adjustments
            unknowns_approximation += adjustment_vector

        # Assemble final 2x3 affine rotation and translation matrix
        # Rotation is applied around the barycenter of the target image
        rotation_center = (self.target_x, self.target_y)        
        # Calculate rotation angle in degrees using math.pi
        rotation_angle_degrees = -unknowns_approximation[0, 0] * 180.0 / math.pi        
        # Get standard 2D rotation matrix
        transformation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_angle_degrees, 1.0)        
        # Apply the final vertical translation adjustment
        transformation_matrix[1, 2] += unknowns_approximation[1, 0] - self.reference_y + self.target_y

        return transformation_matrix
 

    def _tp_warp(self, id_reference: int, id_target: int, level: int) -> bool:
        """
        Warp the target image using the estimated transformation matrix
        """

        reference_points = []
        target_points = []

        # Get the keypoints from the good matches
        for match in self.good_matches:
            # Query index points to the reference keypoints
            reference_pixel_coords = self.keypoints_reference[match.queryIdx].pt
            # Train index points to the target keypoints
            target_pixel_coords = self.keypoints_target[match.trainIdx].pt

            reference_points.append(reference_pixel_coords)
            target_points.append(target_pixel_coords)

        # Estimate the quasi-epipolar transformation model
        self.transformation_matrix = self._estimate_transformation_matrix(reference_points, target_points)

        if self.transformation_matrix is None:
            print("ERROR: Transformation matrix cannot be estimated!")
            return False

        # Retrieve the dimensions of the reference image
        h, w = self.reference_array.shape[:2]

        # Warp the target image to align with the reference
        self.target_array_warped = cv2.warpAffine(self.target_array, self.transformation_matrix, (w, h))

        # Save aligned images
        os.makedirs("/opt/data/ossim/output/warped/", exist_ok=True)
        reference_name = f"reference_level_{level}_id_{id_reference}.tif"
        target_name = f"target_aligned_level_{level}_id_{id_target}.tif"
        cv2.imwrite("/opt/data/ossim/output/warped/"+reference_name, self.reference_array)
        cv2.imwrite("/opt/data/ossim/output/warped/"+target_name, self.target_array_warped)

        return True