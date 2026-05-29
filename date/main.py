import sys
import argparse
import os
import time

import pyossim

from .images_config import ImagesConfig
from .raw_image import RawImage
from .stereo_pair import StereoPair
from .disparity_merging import DisparityMerging


def ortho(kwl_dict: dict) -> bool:
    chipper = pyossim.ossim_chipper_util()
    chipper.initialize(kwl_dict) # The pybind11 function automatically converts the dictionary to an ossimKeywordList!

    start_time = time.perf_counter()

    try:
        chipper.execute() # Execute the orthorectification        

        elapsed_time = time.perf_counter() - start_time
        print(f"Elapsed time in seconds: {elapsed_time:.3f}\n")        
    except RuntimeError as e:
        print(f"OSSIM exception: {e}", file=sys.stderr)
        sys.exit(1)

    return True


def main():
    pyossim.init()

    # Check if the number of arguments passed is less than 4 (the minimum expected)
    if len(sys.argv) < 4: 
        print("ERROR: Few arguments... At least 5 arguments are expected!")
        print("Usage: main.py <configuration_file> <output_results_directory> <output_dsm_name>")
        print("Options:")
        print("--number-steps <number_steps> ===> Specify the number of steps for pyramidal processing.")
        print("--meters <meters> ===> Specify a size in meters for resampling.")
        print("--cut-extent-ll <min_lat> <max_lat> <min_lon> <max_lon> ===> Specify an extent with the minimum and maximum latitude and longitude in decimal degrees.")
        sys.exit(1)

    parser = argparse.ArgumentParser()

    # Fixed arguments
    parser.add_argument("input_filename", type=str, help="Input images configuration text file")
    parser.add_argument("output_dir", type=str, help="Output directory path")
    parser.add_argument("output_filename", type=str, help="Output filename")

    # Optional arguments (flags)
    parser.add_argument("--number-steps", type=int, default=1, help="Number of steps for pyramidal processing")
    parser.add_argument("--meters", type=float, default=5.0, help="Grid spacing in meters")
    parser.add_argument("--cut-extent-ll", nargs=4, type=float, metavar=("min_lat", "max_lat", "min_lon", "max_lon"), help="Extent coordinates") 

    args = parser.parse_args()
    print(f"\nArguments: {args}\n")

    # Read the input images configuration text file
    try:
        f_input = open(args.input_filename, "r")
    except FileNotFoundError:
        print("Missing input file")
        sys.exit(1)

    images_list = []
    stereo_pairs_list = []

    lines = f_input.readlines()

    tokens = []
    for line in lines:
        tokens.extend(line.strip().split())
    # Skip empty tokens
    tokens = [t for t in tokens if t]

    idx = 0

    # Read images
    images_number = int(tokens[idx])
    idx += 1
    print(f"Number of images: {images_number}")

    for _ in range(images_number):
        id = tokens[idx]
        file_path = tokens[idx + 1]
        orbit = tokens[idx + 2]
        idx += 3

        image = RawImage(raw_image_id=id, raw_image_path=file_path, orbit=orbit)
        images_list.append(image)

        print(f"Id: {id}")
        print(f"Path: {file_path}")
        print(f"Orbit: {orbit}")

    configuration = ImagesConfig(
        meters = args.meters or 1.0,
        number_steps = args.number_steps or 2
    )

    if args.cut_extent_ll:
        configuration.min_lat, configuration.max_lat, configuration.min_lon, configuration.max_lon = args.cut_extent_ll
    else:
        configuration.calculate_extent_from_images([image.raw_image_path for image in images_list])

    print(f"\nConfiguration of images: {configuration}")

    # Read pairs
    pairs_number = int(tokens[idx])
    idx += 1
    print(f"\nNumber of pairs: {pairs_number}\n")

    for _ in range(pairs_number):
        id_master = tokens[idx]
        id_slave = tokens[idx + 1]
        idx += 2

        stereo_pair = StereoPair()
        stereo_pair.set_ids(int(id_master), int(id_slave))
        stereo_pair.set_raw_paths(images_list[int(id_master)].raw_image_path, images_list[int(id_slave)].raw_image_path)

        stereo_pair.epipolar_direction()
        stereo_pairs_list.append(stereo_pair)

        print(f"Pair: {id_master} | {id_slave}")
        print(f"Master path: {stereo_pair.raw_master_path}")
        print(f"Slave path: {stereo_pair.raw_slave_path}")  
        print(f"Conversion factor of the pair {stereo_pair.id_master} | {stereo_pair.id_slave} : {stereo_pair.mean_conversion_factor}")
        print(f"Rotation angle of the pair {stereo_pair.id_master} | {stereo_pair.id_slave} : {stereo_pair.mean_rotation_angle}\n")

    kwl = {
        "meters": configuration.meters,
        "cut_min_lat": configuration.min_lat,
        "cut_max_lat": configuration.max_lat,  
        "cut_min_lon": configuration.min_lon,      
        "cut_max_lon": configuration.max_lon,        
        "operation": "ortho",
        "resampler_filter": "box",
        "projection": "utm"
    }

    # Pyramidal iteration
    for s in range(configuration.number_steps-1, -1, -1):
        # temp_dsm_path = os.path.join(args.output_dir, "temp_dsm")
        # elev = pyossim.ossim_elev_manager.instance()

        # if s != configuration.number_steps - 1:
        #     elev.load_elevation_path(str(temp_dsm_path), True)

        print(f"STEP: {s}")
        # print(f"Temporary DSM path: {temp_dsm_path}")
        # print(f"Number of elevation databases: {elev.getNumberOfElevationDatabases()}")

        # gpt = pyossim.ossim_gpt(46.07334640, 11.12284482, 0.00)
        # orthometric_height = elev.get_height_above_msl(gpt)
        # ellipsoidal_height = elev.get_height_above_ellipsoid(gpt)
        # print(f"Orthometric height (SRTM only): {orthometric_height} meters")
        # print(f"Ellipsoidal height (SRTM + EGM96): {ellipsoidal_height} meters")
        # print(f"Geoid offset: {ellipsoidal_height - orthometric_height} meters")   

        ortho_res = configuration.meters * (2 ** s)
        kwl["meters"] = ortho_res

        print(f"{ortho_res} m: resolution of this level")
        print(f"{configuration.meters} m: final DSM resolution\n")        

        ortho_images_dict = {}
        ortho_images_mask_list = [] # What is this?

        for n in range(int(images_number)):
            kwl["image1.file"] = images_list[n].raw_image_path

            raw_image_id = images_list[n].raw_image_id
            ortho_file_name = f"ortho_image_level_{s}_image_{raw_image_id}_ortho.TIF"
            ortho_file_path = os.path.join(args.output_dir, "ortho_images", ortho_file_name)
            ortho_images_dict[raw_image_id] = ortho_file_path
            kwl["output_file"] = ortho_file_path

            print(f"Keyword list for orthorectification: {kwl}")

            ortho(kwl)
        
        print(f"Dictionary of orthorectified images: {ortho_images_dict}\n")
            
        for n in range(int(pairs_number)):
            pair = stereo_pairs_list[n]
            pair.ortho_master_path = ortho_images_dict[str(pair.id_master)]
            pair.ortho_slave_path = ortho_images_dict[str(pair.id_slave)]
            
        print(f"List of stereo pairs: {stereo_pairs_list}\n")

        merged_disp = DisparityMerging()
        merged_disp.execute(stereo_pairs_list, ortho_images_mask_list, images_list, ortho_res)
        # final_disp = merged_disp.get_merged_disp()      


if __name__ == "__main__":
    main()