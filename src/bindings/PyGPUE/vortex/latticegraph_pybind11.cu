#include "pybind11/pybind11.h"
#include "pybind11/iostream.h"
#include "pybind11/complex.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"

#include <cuda.h>

#include "vort.h"
#include "node.h"
#include "edge.h"
#include "lattice.h"

#include <tuple>

#define PYBIND11_EXPORT __attribute__ ((visibility("default")))

namespace py = pybind11;
using namespace LatticeGraph;
using namespace Vtx;

template <class T>
void vortex_binding(py::module &m){
    py::class_<T>(m, "Vortex")
        .def(py::init<>())
        .def(py::init<int2, double2, int, bool, std::size_t>())
        .def("updateUID", &T::updateUID)
        .def("updateWinding", &T::updateWinding)
        .def("updateIsOn", &T::updateIsOn)
        .def("updateCoords", &T::updateCoords)
        .def("updateCoordsD", &T::updateCoordsD)
        .def("updateTimeStep", &T::updateTimeStep)
        .def("getUID", &T::getUID)
        .def("getWinding", &T::getWinding)
        .def("getIsOn", &T::getIsOn)
        .def("getCoords", &T::getCoords)
        .def("getCoordsD", &T::getCoordsD)
        .def("getTimeStep", &T::getTimeStep);
}

template <class T>
void vtxlist_binding(py::module &m){
    py::class_<T>(m, "VtxList")
        .def(py::init<>())
        .def(py::init<std::size_t>())
        .def("addVtx", py::overload_cast<std::shared_ptr<Vortex>>(&T::addVtx), "")
        .def("addVtx", py::overload_cast<std::shared_ptr<Vortex>,std::size_t>(&T::addVtx), "")
        .def("removeVtx", &T::removeVtx)
        .def("getVortices", &T::getVortices) //may need to edit for refs
        .def("getVtx_Uid", &T::getVtx_Uid)
        .def("getVtx_Idx", &T::getVtx_Idx)
        .def("getVtxIdx_Uid", &T::getVtxIdx_Uid)
        .def("getMax_Uid", &T::getMax_Uid)
        .def("getVtxMinDist", &T::getVtxMinDist)
        .def("swapUid", &T::swapUid)
        .def("swapUid_Idx", &T::swapUid_Idx)
        .def("sortVtxUID", &T::sortVtxUID)
        .def("arrangeVtx", &T::arrangeVtx)
        .def("setUIDs", &T::setUIDs)
        .def("minDistPair", &T::minDistPair);
        //.def("vortOff", &T::vortOff) //to be defined
}

template <class T>
void node_binding(py::module &m){
    py::class_<T>(m, "Node")
        .def(py::init<>())
        .def(py::init<Vortex&>())
        .def("getUid", &T::getUid)
        .def("getData", &T::getData)
        .def("getEdges", &T::getEdges)
        .def("getEdge", &T::getEdge)
        .def("getConnectedNode", &T::getConnectedNode)
        .def("setData", &T::setData)
        .def("addEdge", &T::addEdge)
        .def("removeEdge", py::overload_cast<std::shared_ptr<Node>>(&T::removeEdge), "")
        .def("removeEdge", py::overload_cast<std::weak_ptr<Edge>>(&T::removeEdge), "")
        .def("removeEdgeUid", &T::removeEdgeUid)
        .def("removeEdgeIdx", &T::removeEdgeIdx)
        .def("removeEdges", &T::removeEdges);
        //.def("getSuid", &T::getSuid)
        //.def("getConnectedNodes", &T::getConnectedNodes) //to be defined
}

template <class T>
void edge_binding(py::module &m){
    py::class_<T>(m, "Edge")
        .def(py::init<>())
        .def(py::init<std::weak_ptr<Node>, std::weak_ptr<Node> >())
        .def(py::init<std::weak_ptr<Node>, std::weak_ptr<Node>, int, double >())
        .def("getUid", &T::getUid)
        .def("getDirection", &T::getDirection)
        .def("getVortex", &T::getVortex)
        .def("setDirection", &T::setDirection)
        .def("setWeight", &T::setWeight)
        .def("updateVortex", &T::updateVortex)
        .def("isMember", &T::isMember);
        //.def("getSuid", &T::getSuid)
}

template <class T>
void lattice_binding(py::module &m){
    py::class_<T>(m, "Lattice")
        .def(py::init<>())
        .def("getVortices", &T::getVortices)
        .def("getEdges", &T::getEdges)
        .def("getVortexIdx", &T::getVortexIdx)
        .def("getEdgeIdx", &T::getEdgeIdx)
        .def("getVortexIdxUid", &T::getVortexIdxUid)
        .def("getEdgeIdxUid", &T::getEdgeIdxUid)
        .def("getVortexUid", &T::getVortexUid)
        .def("getEdgeUid", &T::getEdgeUid)
        .def("getVortexDistance", &T::getVortexDistance)
        .def("getVortexDistanceD", &T::getVortexDistanceD)
        .def("setVortex", &T::setVortex)
        .def("setEdge", &T::setEdge)
        .def("addVortex", &T::addVortex)
        .def("addEdge", py::overload_cast<std::shared_ptr<Edge>>(&T::addEdge), "")
        //.def("addEdge", py::overload_cast<std::shared_ptr<Node>,std::shared_ptr<Node>>(&T::addEdge), "")
        .def("addEdge", py::overload_cast<std::shared_ptr<Edge>,std::shared_ptr<Node>,std::shared_ptr<Node>>(&T::addEdge), "")
        .def("removeVortex", &T::removeVortex)
        .def("removeVortexIdx", &T::removeVortexIdx)
        .def("removeVortexUid", &T::removeVortexUid)
        .def("removeEdge", py::overload_cast<std::shared_ptr<Node>,std::shared_ptr<Node>>(&T::removeEdge), "")
        .def("removeEdge", py::overload_cast<std::shared_ptr<Edge>>(&T::removeEdge), "")
        .def("removeEdgeUid", &T::removeEdgeUid)
        .def("removeEdgeIdx", &T::removeEdgeIdx)
        .def("removeEdges", &T::removeEdges)
        .def("createVortex", &T::createVortex)
        .def("destroyVortex", &T::destroyVortex)
        .def("createEdges", py::overload_cast<unsigned int>(&T::createEdges), "")
        .def("createEdges", py::overload_cast<double>(&T::createEdges), "")
        .def("genAdjMat", py::overload_cast<unsigned int*>(&T::genAdjMat), "")
        .def("genAdjMat", py::overload_cast<double*>(&T::genAdjMat), "")
        .def("adjMatMtca", py::overload_cast<unsigned int*>(&T::adjMatMtca), "")
        .def("adjMatMtca", py::overload_cast<double*>(&T::adjMatMtca), "")
        .def("swapIdxUid", &T::swapIdxUid)
        .def("swapIdx", &T::swapIdx)
        .def("isConnected", &T::isConnected);
}

PYBIND11_MODULE(_PyGPUE_VTX, m){
    vortex_binding<Vortex>(m);
    vtxlist_binding<VtxList>(m);
    node_binding<Node>(m);
    edge_binding<Edge>(m);
    lattice_binding<Lattice>(m);
}

