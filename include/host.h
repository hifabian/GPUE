/*
* host.h - GPUE2: GPU Split Operator solver for Nonlinear
* Schrodinger Equation, Copyright (C) 2018, Lee J. O'Riordan, James Schloss
*/

//###########################################################################################################//
//
#ifndef HOST_H
#define HOST_H

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cufft.h>
#include "../include/gpu_functions.h"
#include "../include/constants.h"
#include "../include/host.h"

class Host{
public:
	//###########################################################################################################//
	static void setupFFT( struct addr_grid *grid, cufftHandle plan_xyz, cufftHandle plan_xy, cufftHandle plan_x_batchY);
	static void allocateMemoryDevice( struct addr_grid *grid, struct addr_Uop *U_op );
	static void freeMemoryDevice( struct addr_Uop *U_op);
	static void allocateMemoryHost( unsigned int selection, struct addr_grid *grid, struct addr_op *op, struct addr_Uop *U_op );
	static void freeMemoryHost( struct addr_op *op, struct addr_Uop *U_ops );
	static void initHamiltonianGnd( struct addr_grid *grid, struct addr_op *op, struct addr_Uop *U_op );
	static void initHamiltonianEv( struct addr_grid *grid, struct addr_op *op, struct addr_Uop *Uop );
	static void defineGrid( struct addr_grid *grid );
	static void parseArgs( int argc, char **argv );
	static void splitOp( unsigned int steps, double dt, struct addr_grid grid*, struct addr_op *op, struct addr_Uop *Uop);
	//###########################################################################################################//
}
#endif

//###########################################################################################################//
