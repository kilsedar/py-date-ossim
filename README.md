# DATE Plugin for OSSIM: Python Implementation

This project brings the [DATE (Digital Automatic Terrain Extractor) plugin](https://github.com/martidi/opencv_dsm/tree/imageStack) for OSSIM back to life. It uses `pybind11` to rely on OSSIM for geomatic operations in Python. 

* Check `date/command.txt` to run the Python code.
* To regenerate the `.so` file for `pyossim` after editing pyossim/ossim_bindings.cpp, check `pyossim/command.txt`.

### Environment & Data Setup

**1. Build the Docker Image**
To have access to OSSIM, the project must be run in an Ubuntu-based Docker container. Build the base Docker image before starting the project:
```bash
docker build -t ossim:0.2 -f docker/Dockerfile .
```

**2. Configure the Data Directory**
Create a folder named `data/` in the root of this project. Inside `data/`, create the following four subfolders:
* `input/`: Place your stereo images here.
* `output/`: This folder is automatically populated.
* `dsm/`: Place a low-resolution DSM, such as SRTM, covering your AOI.
* `geoid/`: Place the [EGM96 geoid file](https://download.osgeo.org/ossim/data/geoids/geoid1996/) here.

**3. Launch the Project**
This repository includes a `.devcontainer` configuration. Once you have built the image and created the `data/` folder, open this project in any IDE that supports dev containers to automatically launch the container, mount your data, and start developing.