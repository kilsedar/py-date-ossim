import sys
import argparse
import os
import time
import shutil
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
        print("--number-levels <number_levels> ===> Specify the number of levels for pyramidal processing.")
        print("--meters <meters> ===> Specify a size in meters for resampling.")
        print("--cut-extent-ll <min_lat> <max_lat> <min_lon> <max_lon> ===> Specify an extent with the minimum and maximum latitude and longitude in decimal degrees.")
        sys.exit(1)

    parser = argparse.ArgumentParser()

    # Fixed arguments
    parser.add_argument("input_filename", type=str, help="Input images configuration text file")
    parser.add_argument("output_dir", type=str, help="Output directory path")
    parser.add_argument("output_filename", type=str, help="Output filename")

    # Optional arguments (flags)
    parser.add_argument("--number-levels", type=int, default=2, help="Number of levels for pyramidal processing")
    parser.add_argument("--meters", type=float, default=1.0, help="Grid spacing in meters")
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

    configuration = ImagesConfig(meters = args.meters, number_levels = args.number_levels)

    if args.cut_extent_ll:
        configuration.min_lat, configuration.max_lat, configuration.min_lon, configuration.max_lon = args.cut_extent_ll
    else:
        configuration.calculate_extent_from_images([image.raw_image_path for image in images_list])

    # Read pairs
    pairs_number = int(tokens[idx])
    idx += 1
    print(f"\nNumber of pairs: {pairs_number}\n")

    for _ in range(pairs_number):
        id_reference = tokens[idx]
        id_target = tokens[idx + 1]
        idx += 2

        stereo_pair = StereoPair(id_reference=int(id_reference), id_target=int(id_target), raw_reference_path=images_list[int(id_reference)].raw_image_path, raw_target_path=images_list[int(id_target)].raw_image_path)

        stereo_pair.epipolar_direction()
        stereo_pairs_list.append(stereo_pair)

        print(f"Pair: {id_reference} | {id_target}")
        print(f"Reference path: {stereo_pair.raw_reference_path}")
        print(f"Target path: {stereo_pair.raw_target_path}")  
        print(f"Conversion factor of the pair {stereo_pair.id_reference} | {stereo_pair.id_target}: {stereo_pair.mean_conversion_factor}")
        print(f"Rotation angle of the pair {stereo_pair.id_reference} | {stereo_pair.id_target}: {stereo_pair.mean_rotation_angle}\n")

    kwl = {
        "cut_min_lat": str(configuration.min_lat),
        "cut_max_lat": str(configuration.max_lat),  
        "cut_min_lon": str(configuration.min_lon),      
        "cut_max_lon": str(configuration.max_lon),   
        "operation": "ortho",
        "resampler_filter": "box", # https://manpages.debian.org/experimental/ossim-core/ossim-chipper.1.en.html => ossim-info --resampler-filters
        "projection": "utm",
        "zone": "32",
        "hemisphere": "N",
        "snap_tie_to_origin": "true" # Ensure grid alignment
    }
    
    srtm_dir = "/opt/data/ossim/dsm"
    srtm_path_cache = os.path.join(srtm_dir, "elev_cell_map.kwl")
    temp_dsm_dir = os.path.join(args.output_dir, "temp_dsm")
    final_dsm_dir = os.path.join(args.output_dir, "dsm")
    if os.path.exists(srtm_path_cache):
        os.remove(srtm_path_cache)
    if os.path.exists(temp_dsm_dir):
        shutil.rmtree(temp_dsm_dir)
    os.makedirs(temp_dsm_dir)
    if os.path.exists(final_dsm_dir):
        shutil.rmtree(final_dsm_dir)
    os.makedirs(final_dsm_dir)

    # Pyramidal iteration
    for l in range(configuration.number_levels-1, -1, -1):
        elev = pyossim.ossim_elev_manager.instance()

        if l != configuration.number_levels - 1:
            elev.clear()
            elev.load_elevation_path(srtm_dir, True)   
            elev.load_elevation_path(str(temp_dsm_dir), True)

        print(f"*************************************\n")
        print(f"LEVEL: {l}")

        print(f"Number of elevation databases: {elev.get_number_of_elevation_databases()}")

        gpt = pyossim.ossim_gpt(46.054, 11.125, 0.00)
        orthometric_height = elev.get_height_above_msl(gpt)
        ellipsoidal_height = elev.get_height_above_ellipsoid(gpt)
        print(f"Orthometric height (SRTM/temp DSM only): {orthometric_height} meters")
        print(f"Ellipsoidal height (SRTM/temp DSM + EGM96): {ellipsoidal_height} meters")
        print(f"Geoid offset: {ellipsoidal_height - orthometric_height} meters")   

        ortho_res = configuration.meters * (2 ** l)
        kwl["meters"] = str(ortho_res)

        print(f"{ortho_res} m: resolution of this level")
        print(f"{configuration.meters} m: final DSM resolution\n")        

        ortho_images_dict = {}

        for n in range(int(images_number)):
            kwl["image0.file"] = images_list[n].raw_image_path

            raw_image_id = images_list[n].raw_image_id
            ortho_file_name = f"ortho_image_level_{l}_id_{raw_image_id}.tif"
            ortho_file_path = os.path.join(args.output_dir, "ortho_images", ortho_file_name)
            ortho_images_dict[raw_image_id] = ortho_file_path
            kwl["output_file"] = ortho_file_path

            print(f"Keyword list for orthorectification: {kwl}")

            ortho(kwl)
        
        print(f"Dictionary of orthorectified images: {ortho_images_dict}\n")
            
        for n in range(int(pairs_number)):
            pair = stereo_pairs_list[n]
            pair.ortho_reference_path = ortho_images_dict[str(pair.id_reference)]
            pair.ortho_target_path = ortho_images_dict[str(pair.id_target)]

        merged_disp = DisparityMerging()
        success = merged_disp.execute(stereo_pairs_list, l, ortho_res)
        if success:
            merged_disp.compute_dsm(args, elev, stereo_pairs_list, l) 

    print("A digital surface model from your triplet is successfully generated!")   


if __name__ == "__main__":
    main()