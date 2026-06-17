#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ossim/init/ossimInit.h>

#include <ossim/base/ossimFilename.h>
#include <ossim/base/ossimKeywordlist.h> 
#include <ossim/base/ossimIpt.h>
#include <ossim/base/ossimGpt.h>
#include <ossim/base/ossimIrect.h>

#include <ossim/imaging/ossimImageHandlerRegistry.h>
#include <ossim/imaging/ossimImageHandler.h>
#include <ossim/imaging/ossimImageGeometry.h>

#include <ossim/projection/ossimUtmpt.h>
#include <ossim/util/ossimChipperUtil.h>
#include <ossim/elevation/ossimElevManager.h>


namespace py = pybind11;


/* py::tuple get_tile_min_max_elevation(double min_lat, double max_lat, double min_lon, double max_lon, double step = 0.001) {
    ossimInit::instance()->initialize();
    
    double min_height = std::numeric_limits<double>::infinity();
    double max_height = -std::numeric_limits<double>::infinity();

    ossimElevManager* elev_manager = ossimElevManager::instance();

    for(double lat = min_lat; lat < max_lat; lat += step) {
        for(double lon = min_lon; lon < max_lon; lon += step) {
            
            ossimGpt world_point(lat, lon, 0.00);
            double height = elev_manager->getHeightAboveMSL(world_point);
            // std::cout << lat << " " << lon << " " << height << std::endl;
            
            if (height < min_height) min_height = height;
            if (height > max_height) max_height = height;
        }
    }

    // Return a Python tuple natively
    return py::make_tuple(min_height, max_height);
} */


// This macro tells pybind11 to use ossimRefPtr as a holder type
PYBIND11_DECLARE_HOLDER_TYPE(T, ossimRefPtr<T>, true);

PYBIND11_MODULE(pyossim, m) {
    m.doc() = "pybind11 bindings for various OSSIM classes and functions and a wrapper for my functions";

    // Initialize OSSIM 
    m.def("init", []() {
        ossimInit::instance()->initialize();
        /* int argc = 1;
        char* arg0 = strdup("python");
        char* argv[] = {arg0, nullptr};
        ossimInit::instance()->initialize(argc, argv);
        free(arg0); */
    });


    // Bind ossimIpt: 2D integer point (used for discrete pixel coordinates and image grid dimensions)
    py::class_<ossimIpt>(m, "ossim_ipt")
        .def(py::init<int, int>(), py::arg("x")=0, py::arg("y")=0)
        .def_readwrite("x", &ossimIpt::x) // def_readwrite lets access to public member variables in Python (read and write access)
        .def_readwrite("y", &ossimIpt::y)
        .def("__repr__", [](const ossimIpt& p) {
            return "Width: " + std::to_string(p.x) + ", Height: " + std::to_string(p.y);
        });


    // Bind ossimDpt: 2D double point (used for sub-pixel accuracy and PCS coordinates (e.g., in easting and northing))
    py::class_<ossimDpt>(m, "ossim_dpt")
        .def(py::init<double, double>(), py::arg("x")=0.0, py::arg("y")=0.0)
        .def_readwrite("x", &ossimDpt::x)
        .def_readwrite("y", &ossimDpt::y)
        .def("__repr__", [](const ossimDpt& p) {
            return "X: " + std::to_string(p.x) + ", Y: " + std::to_string(p.y);
        });


    // Bind ossimGpt: 3D double point (used for GCS coordinates)
    py::class_<ossimGpt>(m, "ossim_gpt")
        .def(py::init<double, double, double>(), py::arg("lat")=0.0, py::arg("lon")=0.0, py::arg("height")=0.0)
        .def_readwrite("lat", &ossimGpt::lat)
        .def_readwrite("lon", &ossimGpt::lon)
        .def_readwrite("height", &ossimGpt::hgt)
        .def("__repr__", [](const ossimGpt& p) {
            return "Latitude: " + std::to_string(p.lat) + ", Longitude: " + std::to_string(p.lon) +  ", Height: " + std::to_string(p.hgt);
        });


    py::class_<ossimUtmpt>(m, "ossim_utmpt")
        .def(py::init<const ossimGpt&>()) 
        .def_property_readonly("easting", &ossimUtmpt::easting) // def_property_readonly makes a function look like a variable in Python (only read access)
        .def_property_readonly("northing", &ossimUtmpt::northing)
        .def_property_readonly("zone", &ossimUtmpt::zone)
        .def_property_readonly("hemisphere", &ossimUtmpt::hemisphere) // 'N' or 'S'
        .def("__repr__", [](const ossimUtmpt& p) {
            return "UTM " + std::to_string(p.zone()) + std::string(1, p.hemisphere()) + " E: " + std::to_string(p.easting()) + " N: " + std::to_string(p.northing());
        });


    // Bind ossimIrect: rectangle in pixel space
    py::class_<ossimIrect>(m, "ossim_irect")
        .def(py::init<int, int, int, int>()) // ul_x, ul_y, lr_x, lr_y
        .def("width", &ossimIrect::width)
        .def("height", &ossimIrect::height)
        .def("ul", &ossimIrect::ul) // Return ossimIpt (upper-left)
        .def("lr", &ossimIrect::lr) // Return ossimIpt (lower-right)
        .def("__repr__", [](const ossimIrect& r) {
            return "Width: " + std::to_string(r.width()) + ", Height: " + std::to_string(r.height());
        });


    py::class_<ossimImageGeometry, ossimRefPtr<ossimImageGeometry>>(m, "ossim_image_geometry")
        .def(py::init<>())

        .def("get_image_size", &ossimImageGeometry::getImageSize, "Return the image dimensions as an ossimIpt")
            
         // Bind worldToLocal, return an ossimDpt
        .def("world_to_local", [](ossimImageGeometry& self, const ossimGpt& worldPt) {
            ossimDpt localPt;
            self.worldToLocal(worldPt, localPt);
            return localPt;
        }, py::arg("world_point"), "Convert world (latitude, longitude, height) to local (x, y) coordinates")

        // Bind localToWorld, return an ossimGpt using height
        .def("local_to_world", [](ossimImageGeometry& self, const ossimDpt& localPt, double height) {
            ossimGpt worldPt;
            self.localToWorld(localPt, height, worldPt);
            return worldPt;
        }, py::arg("local_point"), py::arg("height"), "Convert local (x, y) at a specific height to world (latitude, longitude, height) coordinates")
        
        // Bind localToWorld, return an ossimGpt using the internal elevation model if available
        .def("local_to_world", [](ossimImageGeometry& self, const ossimDpt& localPt) {
            ossimGpt worldPt;
            self.localToWorld(localPt, worldPt);
            return worldPt;
        }, py::arg("local_point"), "Convert local (x, y) to world (latitude, longitude, height) coordinates using default elevation");
    

    py::class_<ossimImageHandler, ossimRefPtr<ossimImageHandler>>(m, "ossim_image_handler")
        .def("get_image_geometry", &ossimImageHandler::getImageGeometry, "Return the image geometry object")

        .def("get_bounding_rect", &ossimImageHandler::getBoundingRect, py::arg("res_level")=0, "Return the bounding rectangle in pixel space")
        
        .def("save_image_geometry", [](ossimImageHandler& self) {
            return self.saveImageGeometry();
        }, "Save the image geometry to a .geom file");


    py::class_<ossimImageHandlerRegistry>(m, "ossim_image_handler_registry")
        .def_static("instance", &ossimImageHandlerRegistry::instance, py::return_value_policy::reference)

        .def("open", [](ossimImageHandlerRegistry& self, const std::string& filename) {
            ossimRefPtr<ossimImageHandler> handler = self.open(ossimFilename(filename));
            return handler;
        }, "Open an image file and return a handler");


    py::class_<ossimChipperUtil>(m, "ossim_chipper_util")
        .def(py::init<>())

        .def("execute", &ossimChipperUtil::execute)
 
         // Lambda adapter for Python dictionaries
        .def("initialize", [](ossimChipperUtil& self, py::dict args) {
            ossimKeywordlist kwl;

            // Loop through the Python dictionary
            for (auto item : args) {
                // Convert keys and values to strings
                std::string key = py::str(item.first);
                std::string val = py::str(item.second);
                
                // Add to the OSSIM object
                kwl.add(key.c_str(), val.c_str());
            }

            // Call the actual C++ function
            self.initialize(kwl);
        });      

        
    py::class_<ossimElevManager>(m, "ossim_elev_manager")
        // Expose the singleton instance
        .def_static("instance", &ossimElevManager::instance, py::return_value_policy::reference)

        // Accept a Python string and convert it to ossimFilename on the fly
        .def("load_elevation_path", [](ossimElevManager& self, const std::string& path, bool obtain_lock=true) {
            return self.loadElevationPath(ossimFilename(path), obtain_lock);
        }, py::arg("path"), py::arg("obtain_lock")=true, "Load a temporary DSM path into the elevation manager")

        .def("get_number_of_elevation_databases", &ossimElevManager::getNumberOfElevationDatabases, "Get the number of loaded elevation databases")

        .def("get_height_above_ellipsoid", &ossimElevManager::getHeightAboveEllipsoid, py::arg("gpt"), "Get height above ellipsoid for a given ground point")

        .def("get_height_above_msl", &ossimElevManager::getHeightAboveMSL, py::arg("gpt"), "Get height above MSL for a given ground point");


        // m.def("get_tile_min_max_elevation", &get_tile_min_max_elevation, py::arg("min_lat"), py::arg("max_lat"), py::arg("min_lon"), py::arg("max_lon"), py::arg("step")=0.001, "Calculate minimum and maximum elevation for a given extent");
}
