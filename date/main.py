import sys
import argparse

import pyossim

from .image_config import ImageConfig
from .raw_image import RawImage
from .stereo_pair import StereoPair


def main():
    pyossim.init()

    # Check if the number of arguments passed is less than 4 (the minimum expected)
    if len(sys.argv) < 4: 
        print("ERROR: Few arguments... At least 5 arguments are expected!")
        print("Usage: main.py <configuration_file> <output_results_directory> <output_dsm_name>")
        print("Options:")
        print("--number-steps <number_steps> ===> Specify the number of steps for pyramidal processing.")
        print("--meters <meters> ===> Specify a size in meters for resampling.")
        print("--cut-bbox-ll <min_lat> <max_lat> <min_lon> <max_lon> ===> Specify a bounding box with the minimum and maximum latitude and longitude in decimal degrees.")
        sys.exit(1)

    parser = argparse.ArgumentParser()

    # Fixed arguments
    parser.add_argument('input_filename', type=str, help="Input images configuration text file")
    parser.add_argument('output_dir', type=str, help="Output directory path")
    parser.add_argument('output_filename', type=str, help="Output filename")

    # Optional arguments (flags)
    parser.add_argument('--number-steps', type=int, default=1, help="Number of steps for pyramidal processing")
    parser.add_argument('--meters', type=float, default=5.0, help="Grid spacing in meters")
    parser.add_argument('--cut-bbox-ll', nargs=4, type=float, metavar=('min_lat', 'max_lat', 'min_lon', 'max_lon'), help="Bounding box coordinates") 

    args = parser.parse_args()
    print(f"\nArguments: {args}\n")

    # Read the input images configuration text file
    try:
        f_input = open(args.input_filename, 'r')
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

    configuration = ImageConfig(
        meters = args.meters or 2.0,
        number_steps = args.number_steps or 2
    )

    if args.cut_bbox_ll:
        configuration.min_lat, configuration.max_lat, configuration.min_lon, configuration.max_lon = args.cut_bbox_ll
    else:
        configuration.calculate_bbox_from_images([image.raw_image_path for image in images_list])

    print(f"\nImage configuration: {configuration}")

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


if __name__ == "__main__":
    main()