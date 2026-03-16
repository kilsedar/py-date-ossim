#include <Python.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ossim/init/ossimInit.h>

#include <ossim/base/ossimFilename.h>
#include <ossim/base/ossimKeywordlist.h> 
#include <ossim/base/ossimGpt.h>
#include <ossim/base/ossimIpt.h>

#include <ossim/imaging/ossimImageHandlerRegistry.h>
#include <ossim/imaging/ossimImageHandler.h>
#include <ossim/imaging/ossimImageGeometry.h>

#include <ossim/projection/ossimUtmpt.h>
#include <ossim/util/ossimChipperUtil.h>
#include <ossim/elevation/ossimElevManager.h>


namespace py = pybind11;


py::tuple get_tile_min_max_elevation(double min_lat, double max_lat, double min_lon, double max_lon, double step = 0.001) {
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
}


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


    // Bind ossimIpt (integer point for image / pixel dimensions)
    py::class_<ossimIpt>(m, "ossimIpt")
        .def(py::init<int, int>(), py::arg("x")=0, py::arg("y")=0)

        .def_readwrite("x", &ossimIpt::x)

        .def_readwrite("y", &ossimIpt::y)

        .def("__repr__", [](const ossimIpt& p) {
            return "Width: " + std::to_string(p.x) + ", Height: " + std::to_string(p.y);
        });


    // Bind ossimDpt (double point for local / image / pixel coordinates)
    py::class_<ossimDpt>(m, "ossimDpt")
        .def(py::init<double, double>(), py::arg("x")=0.0, py::arg("y")=0.0)

        .def_readwrite("x", &ossimDpt::x)

        .def_readwrite("y", &ossimDpt::y)

        .def("__repr__", [](const ossimDpt& p) {
            return "X: " + std::to_string(p.x) + ", Y: " + std::to_string(p.y);
        });


    // Bind ossimGpt (double point for world / ground coordinates)
    py::class_<ossimGpt>(m, "ossimGpt")
        .def(py::init<double, double, double>(), py::arg("lat")=0.0, py::arg("lon")=0.0, py::arg("height")=0.0)

        .def_readwrite("lat", &ossimGpt::lat)

        .def_readwrite("lon", &ossimGpt::lon)

        .def_readwrite("height", &ossimGpt::hgt)

        .def("__repr__", [](const ossimGpt& p) {
            return "Latitude: " + std::to_string(p.lat) + ", Longitude: " + std::to_string(p.lon) +  ", Height: " + std::to_string(p.hgt);
        });


    py::class_<ossimUtmpt>(m, "ossimUtmpt")
        .def(py::init<const ossimGpt&>())

        .def("easting", &ossimUtmpt::easting)

        .def("northing", &ossimUtmpt::northing)
        
        .def("zone", &ossimUtmpt::zone);


    py::class_<ossimImageGeometry, ossimRefPtr<ossimImageGeometry>>(m, "ossimImageGeometry")
        .def(py::init<>())

        .def("getImageSize", &ossimImageGeometry::getImageSize, "Return the image dimensions as an ossimIpt")
            
         // Bind worldToLocal, return an ossimDpt
        .def("worldToLocal", [](ossimImageGeometry& self, const ossimGpt& worldPt) {
            ossimDpt localPt;
            self.worldToLocal(worldPt, localPt);
            return localPt;
        }, py::arg("world_point"), "Convert world (latitude, longitude, height) to local (x, y) coordinates")

        // Bind localToWorld, return an ossimGpt using height
        .def("localToWorld", [](ossimImageGeometry& self, const ossimDpt& localPt, double height) {
            ossimGpt worldPt;
            self.localToWorld(localPt, height, worldPt);
            return worldPt;
        }, py::arg("local_point"), py::arg("height"), "Convert local (x, y) at a specific height to world (latitude, longitude, height) coordinates")
        
        // Bind localToWorld, return an ossimGpt using the internal elevation model if available
        .def("localToWorld", [](ossimImageGeometry& self, const ossimDpt& localPt) {
            ossimGpt worldPt;
            self.localToWorld(localPt, worldPt);
            return worldPt;
        }, py::arg("local_point"), "Convert local (x, y) to world (latitude, longitude, height) coordinates using default elevation");
    

    py::class_<ossimImageHandler, ossimRefPtr<ossimImageHandler>>(m, "ossimImageHandler")
        .def("getImageGeometry", &ossimImageHandler::getImageGeometry, "Return the image geometry object");


    py::class_<ossimImageHandlerRegistry>(m, "ossimImageHandlerRegistry")
        .def_static("instance", &ossimImageHandlerRegistry::instance, py::return_value_policy::reference)

        .def("open", [](ossimImageHandlerRegistry& self, const std::string& filename) {
            ossimRefPtr<ossimImageHandler> handler = self.open(ossimFilename(filename));
            return handler;
        }, "Open an image file and return a handler");


    py::class_<ossimChipperUtil>(m, "ossimChipperUtil")
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
        

    m.def("get_tile_min_max_elevation", &get_tile_min_max_elevation, py::arg("min_lat"), py::arg("max_lat"), py::arg("min_lon"), py::arg("max_lon"), py::arg("step")=0.001, "Calculate minimum and maximum elevation for a given bounding box");

        
    /* py::class_<ossimElevManager>(m, "ossimElevManager")
        // Expose the singleton instance
        .def_static("instance", &ossimElevManager::instance, py::return_value_policy::reference)

        // Example of adding a method
        // .def("getHeightAboveEllipsoid", &ossimElevManager::getHeightAboveEllipsoid);
        .def("getHeightAboveMSL", [](ossimElevManager& self, const ossimGpt& gpt) {
            return self.getHeightAboveMSL(gpt);
        }); */
}