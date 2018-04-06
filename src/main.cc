/*
* main.cu - GPUE2: Split Operator based GPU solver for Nonlinear
* Schrodinger Equation, Copyright (C) 2018, Lee J. O'Riordan, James Schloss
*/

//###########################################################################################################//

#include "../include/host.h"

//###########################################################################################################//
/*
*	Get to the choppah!
*/
//###########################################################################################################//

int main(int argc, char **argv){
    Host::parseArgs();

    double dt;
    unsigned int gridSize[3];
    double omega_V[3];
    double mass=1.0;

    //These structs contain the addresses of all the essential arrays for both CPU and GPU.
    State::addr_grid addr_grid;
    State::addr_op addr_op_host;
    State::addr_Uop addr_Uop_host, addr_Uop_gpu;

    Host::defineGrid(&addr_grid);

    Host::allocateMemoryHost(0x1ff, &addr_grid, &addr_op_host, &addr_Uop_host);
    Host::allocateMemoryDevice(&addr_grid, &addr_Uop_gpu);

    //Imaginary time evolution
    Host::initHamiltonianGnd( addr_grid, addr_op_host, addr_Uop_host, addr_Uop_gpu );
    Host::splitOp(steps, dt, wfc, wfc_gpu, Uq_gpu, Up_gpu, Uxpy_gpu, Uypx_gpu, buffer);

    //Real time evolution
    Host::initHamiltonianEv( gridSize, X, Y, Z, V, K, XPy, YPx, opV, opK, opXPy, opYPx );
    Host::splitOp(steps, dt, wfc, wfc_gpu, Uq_gpu, Up_gpu, Uxpy_gpu, Uypx_gpu, buffer);

    //Free the memory and go home.
    Host::freeMemoryHost(V,K,XPy,YPx,opV,opK,opXPy,opYPx,wfc);
    Host::freeMemoryDevice(wfc_gpu,Uq_gpu,Up_gpu,Uxpy_gpu,Uypx_gpu,buffer);
}

//###########################################################################################################//
