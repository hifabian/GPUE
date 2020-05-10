#include "pybind11/pybind11.h"
#include "pybind11/iostream.h"
#include "pybind11/complex.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"

#include <cuda.h>

#include "parser.h"
#include "fileIO.h"

#define PYBIND11_EXPORT __attribute__ ((visibility("default")))

namespace py = pybind11;
using namespace FileIO;

void fileio_binding(py::module &m){
    m.def("init", &init);
    m.def("load", &load);
    m.def("loadA", &loadA);
    m.def("writeOutEnergy", &writeOutEnergy);
    m.def("writeOutWfc", &writeOutWfc);
    m.def("writeOutV", &writeOutV);
    m.def("writeOutEdges", &writeOutEdges);
    m.def("writeOutAx", &writeOutAx);
    m.def("writeOutAy", &writeOutAy);
    m.def("writeOutAz", &writeOutAz);
    m.def("writeOutX", &writeOutX);
    m.def("writeOutY", &writeOutY);
    m.def("writeOutZ", &writeOutZ);
    m.def("writeOutParams", &writeOutParams);
    m.def("destroy", &destroy);
    m.def("writeOutInt", &writeOutInt);
    m.def("writeOutVortex", &writeOutVortex);
    m.def("writeOutAdjMat", &writeOutAdjMat);
    m.def("writeOutAdjMat", py::overload_cast<std::string, int*, unsigned int*, int, int>(&writeOutAdjMat), "");
    m.def("writeOutAdjMat", py::overload_cast<std::string, double*, unsigned int*, int, int>(&writeOutAdjMat), "");
}

PYBIND11_MODULE(_PyGPUE_IO, m){
    fileio_binding(m);
    //parser.h
    m.def("parseArgs",&parseArgs);
}

