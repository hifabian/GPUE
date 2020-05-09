#include "pybind11/pybind11.h"
#include "pybind11/iostream.h"
#include "pybind11/complex.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"

#include "init.h"
#include "dynamic.h"
#include "split_op.h"

#include <tuple>

#define PYBIND11_EXPORT __attribute__ ((visibility("default")))

namespace py = pybind11;

class GPUEPy{
    private:
    int device, dimnum, wfc_num;
    int xDim, yDim, zDim;
    int gsteps, esteps;
    bool is_init;
    Grid par;

    public:

    GPUEPy() { }
    GPUEPy(int argc, char** argv) {
        par = parseArgs(argc,argv);
        device = par.ival("device");
        dimnum = par.ival("dimnum");
        wfc_num = par.ival("wfc_num");
        cudaHandleError(cudaSetDevice(device));

        time_t start,fin;
        time(&start);
        printf("Start: %s\n", ctime(&start));

        xDim = par.ival("xDim");
        yDim = par.ival("yDim");
        zDim = par.ival("zDim");
        if(par.bval("read_file") == true){
            FileIO::load(par);
        }

        if(par.bval("corotating_override")){
            std::cout << "Overriding rotational flag set by file\n";
            par.store("corotating", false);
        }
        init(par);

        gsteps = par.ival("gsteps");
        esteps = par.ival("esteps");

        std::string data_dir = par.sval("data_dir");
        std::cout << "variables re-established" << '\n';

        if (par.bval("write_file")){
            FileIO::writeOutParams(par);
        }

        if(gsteps > 0){
            std::cout << "Imaginary-time evolution started..." << '\n';
            par.store("gstate", true);
            set_variables(par);

            evolve(par, gsteps, true);
        }

        if(esteps > 0){
            std::cout << "real-time evolution started..." << '\n';
            par.store("gstate", false);
            set_variables(par);

            evolve(par, esteps, false);
        }

        par.store("found_sobel",false);
        FileIO::writeOutParams(par);

        // Close the output file
        FileIO::destroy();

        std::cout << "done evolving" << '\n';
        time(&fin);
        printf("Finish: %s\n", ctime(&fin));
        printf("Total time: %ld seconds\n ",(long)fin-start);
        std::cout << '\n';
    }
    ~GPUEPy(){}

    void evolve(Grid& par, int num_steps, bool g_state){
        par.store("gstate", g_state);
        set_variables(par);
        if (g_state){
            std::cout << "Imaginary-time evolution started..." << '\n';
        }
        else{
            std::cout << "Real-time evolution started..." << '\n';
        }
    }
    void run(){
        if(!is_init){
            std::cout << "Please initialise the environment before running";
            return;
        }
    }
    void setDevice(int device){
        this->device = device;
    }
    void setDimNum(int dimnum){
        this->dimnum = dimnum;
    }
    void setWfcNum(int wfc_num){
        this->wfc_num = wfc_num;
    }
    void setDims(int xDim, int yDim, int zDim){
        this->xDim = xDim;
        this->yDim = yDim;
        this->zDim = zDim;
    }
    void setGsteps(int gsteps){
        this->gsteps = gsteps;
    }
    void setEsteps(int esteps){
        this->esteps = esteps;
    }

    int getEsteps(){
        return this->esteps;
    }
    int getGsteps(){
        return this->gsteps;
    }
    std::tuple<int,int,int> getDims(){
        return std::make_tuple(this->xDim, this->yDim, this->zDim);
    }
};

template <class GPUE_T>
void gpue_binding(py::module &m){

    py::class_<GPUE_T>(m, "GPUE")
        .def(py::init<>())
        .def("getDims", &GPUE_T::getDims)
        .def("setDims", &GPUE_T::setDims)
        .def("setGsteps", &GPUE_T::setGsteps)
        .def("setEsteps", &GPUE_T::setEsteps)
        .def("getGsteps", &GPUE_T::getGsteps)
        .def("getEsteps", &GPUE_T::getEsteps);
        //.def(py::init<const std::size_t &, const bool &>())
        //.def("printStates", &SimulatorType::PrintStates, py::call_guard<py::scoped_ostream_redirect,py::scoped_estream_redirect>())

}

PYBIND11_MODULE(_PyGPUE, m){
    gpue_binding<GPUEPy>(m);
}

