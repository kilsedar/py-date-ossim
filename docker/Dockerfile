# Use a stable modern base
FROM ubuntu:24.04

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
	cmake \
    git \
	wget \
	bzip2 \
    libopenthreads-dev \
    libtiff-dev \
	libgeotiff-dev \
    libjpeg-dev \
	libopencv-dev \
	libgdal-dev \
    libfreetype6-dev \
	&& apt-get autoremove

WORKDIR /usr/local/src	
RUN wget https://download.osgeo.org/geos/geos-3.14.1.tar.bz2
RUN tar xvfj geos-3.14.1.tar.bz2
WORKDIR /usr/local/src/geos-3.14.1
RUN mkdir _build
WORKDIR /usr/local/src/geos-3.14.1/_build
# Set up the build
RUN cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    ..
# Run the build, test, install
RUN make
RUN ctest
RUN make install
RUN ldconfig

# 2. Clone OSSIM and OSSIM Plugins (Crucial for GDAL support, etc.)
WORKDIR /usr/local/src
RUN git clone https://github.com/ossimlabs/ossim.git     
RUN git clone https://github.com/ossimlabs/ossim-plugins.git 

# 3. Build OSSIM Core
WORKDIR /usr/local/src/ossim/scripts
RUN BUILD_JPEG12_PLUGIN=OFF BUILD_GDAL_PLUGIN=ON BUILD_CNES_PLUGIN=ON ./build.sh

RUN echo "plugin.file1: /usr/local/src/build/lib/libossim_cnes_plugin.so" > /usr/local/src/ossim_preferences.txt && \
	echo "plugin.file2: /usr/local/src/build/lib/libossim_gdal_plugin.so" >> /usr/local/src/ossim_preferences.txt && \
    echo "elevation_manager.elevation_source0.enabled: true" >> /usr/local/src/ossim_preferences.txt && \
    echo "elevation_manager.elevation_source0.type: image_directory" >> /usr/local/src/ossim_preferences.txt && \
    echo "elevation_manager.elevation_source0.connection_string: /opt/data/ossim/dsm" >> /usr/local/src/ossim_preferences.txt && \	
	echo "elevation_manager.elevation_source0.geoid.type: geoid1996" >> /usr/local/src/ossim_preferences.txt && \
	echo "elevation_manager.use_geoid_if_null: true" >> /usr/local/src/ossim_preferences.txt && \
	echo "geoid_egm_96_grid: /opt/data/ossim/geoid/egm96.grd" >> /usr/local/src/ossim_preferences.txt 

ENV OSSIM_PREFS_FILE=/usr/local/src/ossim_preferences.txt
ENV PATH="/usr/local/src/build/bin:${PATH}"
ENV LD_LIBRARY_PATH=""
ENV LD_LIBRARY_PATH="/usr/local/src/build/lib:${LD_LIBRARY_PATH}" 

CMD ["ossim-cli", "--version"]