#include "pybind11/pybind11.h"
#include "pybind11/iostream.h"
#include "pybind11/complex.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"

#include <cuda.h>

#include "ds.h"
#include "dynamic.h"
#include "manip.h"
#include "init.h"
#include "evolution.h"
#include "minions.h"
#include "operators.h"
#include "split_op.h"

#include <tuple>

#define PYBIND11_EXPORT __attribute__ ((visibility("default")))

namespace py = pybind11;

void test_all(){
    printf("Please run unit tests in 'gpue' binary with flag '-u'\n");
}

// ############# ds.h ################ //

void pos_binding(py::module &m){
    py::class_<pos>(m, "Pos")
    .def(py::init<>())
    .def(py::init<double,double,double>())
    .def_readwrite("x", &pos::x)
    .def_readwrite("y", &pos::y)
    .def_readwrite("z", &pos::z);
}

void eqnnode_binding(py::module &m){
    py::class_<EqnNode>(m, "EqnNode")
    .def(py::init<>())
    .def_readwrite("val", &EqnNode::val)
    .def_readwrite("is_dynamic", &EqnNode::is_dynamic)
    .def_readwrite("var", &EqnNode::var)
    .def_readwrite("l_ptr", &EqnNode::left)
    .def_readwrite("r_ptr", &EqnNode::right)
    .def_readwrite("op_num", &EqnNode::op_num)
    .def_readwrite("has_op", &EqnNode::has_op);
}

void eqnnode_gpu_binding(py::module &m){
    py::class_<EqnNode_gpu>(m, "EqnNode_gpu")
    .def(py::init<>())
    .def_readwrite("val", &EqnNode_gpu::val)
    .def_readwrite("is_dynamic", &EqnNode_gpu::is_dynamic)
    .def_readwrite("var", &EqnNode_gpu::var)
    .def_readwrite("l", &EqnNode_gpu::left)
    .def_readwrite("r", &EqnNode_gpu::right)
    .def_readwrite("op_num", &EqnNode_gpu::op_num);
}

void grid_binding(py::module &m){
    py::class_<Grid>(m, "Grid")
        .def(py::init<>())
        .def("store", py::overload_cast<std::string, cufftDoubleComplex*>(&Grid::store), "")
        .def("store", py::overload_cast<std::string, int>(&Grid::store), "")
        .def("store", py::overload_cast<std::string, double>(&Grid::store), "")
        .def("store", py::overload_cast<std::string, double*>(&Grid::store), "")
        .def("store", py::overload_cast<std::string, bool>(&Grid::store), "")
        .def("store", py::overload_cast<std::string, std::string>(&Grid::store), "")
        .def("store", py::overload_cast<std::string, EqnNode>(&Grid::store), "")
        .def("store", py::overload_cast<std::string, std::vector<double> >(&Grid::store), "")
        .def("store", py::overload_cast<std::string, std::vector<double*>>(&Grid::store), "")
        .def("store", py::overload_cast<std::string, std::vector<double2*>>(&Grid::store), "")
        .def("store_new", py::overload_cast<std::string, int>(&Grid::store), "")
        .def("store_new", py::overload_cast<std::string, double>(&Grid::store), "")
        .def("store_new", py::overload_cast<std::string, bool>(&Grid::store), "")
        .def("store_new", py::overload_cast<std::string, std::string>(&Grid::store), "")
        .def("ival_default", &Grid::ival_default)
        .def("dval_default", &Grid::dval_default)
        .def("bval_default", &Grid::bval_default)
        .def("sval_default", &Grid::sval_default)
        .def("ival", &Grid::ival)
        .def("dval", &Grid::dval)
        .def("dsval", &Grid::dsval, py::return_value_policy::reference)
        .def("dvecval", &Grid::dvecval)
        .def("dsvecval", &Grid::dsvecval)
        .def("bval", &Grid::bval)
        .def("sval", &Grid::sval)
        .def("cufftDoubleComplexval", &Grid::cufftDoubleComplexval, py::return_value_policy::reference)
        .def("d2svecval", &Grid::d2svecval)
        .def("astval", &Grid::astval, py::return_value_policy::reference)
        .def("ast_cpuval", &Grid::ast_cpuval)
        .def("is_double", &Grid::is_double)
        .def("is_dstar", &Grid::is_dstar)
        .def("is_ast_gpu", &Grid::is_ast_gpu)
        .def("is_ast_cpu", &Grid::is_ast_cpu)
        .def("print_map", &Grid::print_map)
        .def("set_A_fn", &Grid::set_A_fn)
        .def("set_V_fn", &Grid::set_V_fn)
        .def("set_wfc_fn", &Grid::set_wfc_fn)
        .def("ast_cpuval", &Grid::ast_cpuval)
        .def("ast_cpuval", &Grid::ast_cpuval)
        .def("ast_cpuval", &Grid::ast_cpuval)
        .def("ast_cpuval", &Grid::ast_cpuval)
        .def("getDoubleMap", &Grid::getDoubleMap)
        .def("getIntMap", &Grid::getIntMap)
        .def("getBoolMap", &Grid::getBoolMap)
        .def("getStringMap", &Grid::getStringMap)
        .def_readwrite("Kfn", &Grid::Kfn)
        .def_readwrite("Vfn", &Grid::Vfn)
        .def_readwrite("Afn", &Grid::Afn)
        .def_readwrite("Axfile", &Grid::Axfile)
        .def_readwrite("Ayfile", &Grid::Ayfile)
        .def_readwrite("Azfile", &Grid::Azfile)
        .def_readwrite("Wfcfn", &Grid::Wfcfn)
        .def_readwrite("x", &Grid::x, py::return_value_policy::reference)
        .def_readwrite("y", &Grid::y, py::return_value_policy::reference)
        .def_readwrite("z", &Grid::z, py::return_value_policy::reference)
        .def_readwrite("xp", &Grid::xp, py::return_value_policy::reference)
        .def_readwrite("yp", &Grid::yp, py::return_value_policy::reference)
        .def_readwrite("zp", &Grid::zp, py::return_value_policy::reference);
}







// ############# Multiple ################### //

void freefunc_binding(py::module &m){
    // ds.h
    m.def("generate_plan_other2d", &generate_plan_other2d);
    m.def("generate_plan_other3d", &generate_plan_other3d);
    m.def("set_fns", &set_fns);

    // dynamic.h
    m.def("parse_eqn", &parse_eqn);
    m.def("find_element_num", &find_element_num);
    m.def("tree_to_array", &tree_to_array);
    //m.def("allocate_equation", &allocate_equation);i //undefined
    m.def("parse_param_file", &parse_param_file);

    // manip.h
    m.def("phaseWinding", py::overload_cast<double*, int, double*, double*, double, double, double, double, int>(&WFC::phaseWinding), "");
    m.def("phaseWinding", py::overload_cast<double*, int, double*, double*, double, double, double*, double*, int, int>(&WFC::phaseWinding), "");
    m.def("applyPhase", &WFC::applyPhase);

    // init.h
    m.def("check_memory", &check_memory);
    m.def("init", &init);
    m.def("set_variables", &set_variables);

    // evolution.h
    m.def("evolve", &evolve);
    m.def("apply_gauge", py::overload_cast<Grid&, double2*, double2*, double2*, double2*, double, double, double, bool, cufftHandle, cufftHandle, cufftHandle, double, double, double, double, int, int>(&apply_gauge), "");
    m.def("apply_gauge", py::overload_cast<Grid&, double2*, double2*, double2*, double, double, bool, cufftHandle, cufftHandle, double, double, double, double>(&apply_gauge), "");

    // minions.h
    m.def("psi2", &Minions::psi2);
    m.def("minValue", &Minions::minValue);
    m.def("maxValue", &Minions::maxValue);
    m.def("sumAvg", &Minions::sumAvg);
    m.def("fInvSqRt", &Minions::fInvSqRt);
    m.def("coordSwap", &Minions::coordSwap);
    m.def("complexMag", &Minions::complexMag);
    m.def("complexMag2", &Minions::complexMag2);
    m.def("complexMult", &Minions::complexMult);
    m.def("complexScale", &Minions::complexScale);
    m.def("conj", &Minions::conj);
    m.def("complexDiv", &Minions::complexDiv);

    // operators.h
    m.def("laplacian", py::overload_cast<Grid&, double2*, double2*, int, int, int, double, double, double>(&laplacian), "");
    m.def("laplacian", py::overload_cast<Grid&, double2*, double2*, int, int, double, double>(&laplacian), "");
    m.def("laplacian", py::overload_cast<Grid&, double2*, double2*, int, double>(&laplacian), "");
    m.def("curl2d", &curl2d, py::return_value_policy::reference);
    m.def("curl3d_x", &curl3d_x, py::return_value_policy::reference);
    m.def("curl3d_y", &curl3d_y, py::return_value_policy::reference);
    m.def("curl3d_z", &curl3d_z, py::return_value_policy::reference);
    m.def("curl3d_r", &curl3d_r, py::return_value_policy::reference);
    m.def("curl3d_phi", &curl3d_phi, py::return_value_policy::reference);
    m.def("filecheck", &filecheck);
    //m.def("file_A", &file_A); //undefined
    m.def("generate_p_space", &generate_p_space);
    m.def("generate_K", &generate_K);
    m.def("generate_gauge",generate_gauge);
    m.def("generate_fields", &generate_fields);
    m.def("generate_grid", &generate_grid);

    // split_op.h
    m.def("gpuReduce", &gpuReduce);
    m.def("parSum", &parSum);
    m.def("optLatSetup", &optLatSetup);
    m.def("energy_calc", &energy_calc);

    // Override test calls
    m.def("test_all", &test_all);
}

// ############# Create binding module ################### //

PYBIND11_MODULE(_PyGPUE_SIM, m){
    pos_binding(m);
    eqnnode_binding(m);
    eqnnode_gpu_binding(m);
    grid_binding(m);
    freefunc_binding(m);
}
