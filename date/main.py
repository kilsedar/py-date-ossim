import os
import sys
import argparse

# Add the 'libs' folder to Python's search path
current_dir = os.path.dirname(os.path.abspath(__file__))
libs_path = os.path.join(current_dir, 'libs')
sys.path.append(libs_path)
import ossim_wrapper
# ossim_wrapper.init() => is this necessary?

from .raw_image import RawImage
from .stereo_pair import StereoPair


# Constants
CUT_MIN_LAT_KW = 'cut_min_lat'
CUT_MAX_LAT_KW = 'cut_max_lat'
CUT_MIN_LON_KW = 'cut_min_lon'
CUT_MAX_LON_KW = 'cut_max_lon'
METERS_KW = 'meters'
OP_KW  = 'operation'
RESAMPLER_FILTER_KW = 'resampler_filter'
PROJECTION_KW = 'projection'


# Check if the number of arguments passed is less than 4 (the minimum expected)
if len(sys.argv) < 4: 
  print("ERROR: Few arguments... At least 5 arguments are expected!")
  print("Usage: main.py <configuration_file> <output_results_directory> <output_dsm_name>")
  print("Options:")
  print("--number-steps <number_steps> ===> Specify the number of steps for pyramidal processing.")
  print("--cut-bbox-ll <min_lat> <max_lat> <min_lon> <max_lon> ===> Specify a bounding box with the minimum and maximum latitude and longitude in decimal degrees.")
  print("--meters <meters> ===> Specify a size in meters for resampling.")
  sys.exit(1)


###################################################################
################## BEGINNING OF ARGUMENT PARSING ##################
###################################################################
parser = argparse.ArgumentParser()

# Fixed Arguments
parser.add_argument('input_filename', type=str, help="Input images configuration text file")
parser.add_argument('output_dir', type=str, help="Output directory path")
parser.add_argument('output_filename', type=str, help="Output filename")

# Optional Arguments (Flags)
parser.add_argument('--number-steps', type=int, default=1, help="Number of steps for pyramidal processing")
parser.add_argument('--cut-bbox-ll', nargs=4, type=float, metavar=('min_lat', 'max_lat', 'min_lon', 'max_lon'), help="Bounding box coordinates")
parser.add_argument('--meters', type=float, default=5.0, help="Grid spacing in meters")

args = parser.parse_args()
print(f"\nArguments: {args}\n")

image_key = {}

number_steps = args.number_steps
print(f"Number of steps for pyramidal: {number_steps}\n")

image_key[METERS_KW] = args.meters
print(f"Orthoimages resolution: {args.meters} meters\n")

# Minimum and maximum latitude and longitude computation for the tile defined by the provided bounding box
if args.cut_bbox_ll:
    min_lat, max_lat, min_lon, max_lon = args.cut_bbox_ll

    print(f"Tile extent:\tmin_lat = {min_lat}\n" + f"\t\tmax_lat = {max_lat}\n" + f"\t\tmin_lon = {min_lon}\n" + f"\t\tmax_lon = {max_lon}\n")

    min_height, max_height = ossim_wrapper.get_tile_min_max_elevation(min_lat, max_lat, min_lon, max_lon, 0.001)

    print(f"Minimum height for this tile is {min_height:.6g} m")
    print(f"Maximum height for this tile is {max_height:.6g} m\n")

# Default keyword for orthorectification
image_key[OP_KW] = 'ortho'

# Resampling filter 
image_key[RESAMPLER_FILTER_KW] = 'box'

# Output DSM projection
image_key[PROJECTION_KW] = 'utm'

print(f"{image_key}\n")
#############################################################
################## END OF ARGUMENT PARSING ##################
#############################################################


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
# Skip comments and empty tokens
tokens = [t for t in tokens if t]

idx = 0

# Read number of images
images_number = int(tokens[idx])
idx += 1
print(f"Number of images: {images_number}")

# Read images
for i in range(images_number):
    id = tokens[idx]
    file_path = tokens[idx + 1]
    orbit = tokens[idx + 2]
    idx += 3

    image = RawImage(raw_image_id=id, raw_image_path=file_path, orbit=orbit)
    images_list.append(image)    

    print(f"Id: {id}")
    print(f"Path: {file_path}")
    print(f"Orbit: {orbit}")

# Read number of pairs
pairs_number = int(tokens[idx])
idx += 1
print(f"\nNumber of pairs: {pairs_number}\n")

for i in range(pairs_number):
    id_master = tokens[idx]
    id_slave = tokens[idx + 1]
    idx += 2

    stereo_pair = StereoPair()
    stereo_pair.set_ids(int(id_master), int(id_slave))
    stereo_pair.set_raw_paths(images_list[int(id_master)].raw_image_path, images_list[int(id_slave)].raw_image_path)

    stereo_pair.epipolar_direction()
    stereo_pairs_list.append(stereo_pair)

    print(f"Pair: {id_master} | {id_slave}")

print(f"\nStereo pairs: {stereo_pairs_list}")


chipper = ossim_wrapper.ossimChipperUtil()
print('ChipperUtil loaded successfully!')