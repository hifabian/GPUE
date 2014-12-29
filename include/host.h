/*
* host.h - GPUE2: Split Operator based GPU solver for Nonlinear
* Schrodinger Equation, Copyright (C) 2014, Lee J. O'Riordan.

* This library is free software; you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as
* published by the Free Software Foundation; either version 2.1 of the
* License, or (at your option) any later version. This library is
* distributed in the hope that it will be useful, but WITHOUT ANY
* WARRANTY; without even the implied warranty of MERCHANTABILITY or
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
* License for more details. You should have received a copy of the GNU
* Lesser General Public License along with this library; if not, write
* to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
* Boston, MA 02111-1307 USA
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
