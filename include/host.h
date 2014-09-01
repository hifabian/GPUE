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

//###########################################################################################################//
/*
*  These structs define the addresses to the grid arrays and operators respectively.
*/
//###########################################################################################################//

struct addr_grid {
	unsigned int dim;
	unsigned int *gridSize;
	unsigned int gridMax;
	double *qMax, *pMax, *dq, *dp;//These guys should depend on the size of dim
	double **Q, **P; //First index represents the respective cartesian dimension space. The outer represents the associated value at that point.
};

struct addr_op{
	double *V, *K, *XPy, *YPx;
};

struct addr_Uop {
	double2 *wfc, *opV, *opK, *opXPy, *opYPx, *buffer;
};

//###########################################################################################################//

void allocateMemoryDevice( struct addr_grid *grid, struct addr_Uop *U_op );
void freeMemoryDevice( struct addr_Uop *U_op);
void allocateMemoryHost( unsigned int selection, struct addr_grid *grid, struct addr_op *op, struct addr_Uop *U_op );
void freeMemoryHost( struct addr_op *op, struct addr_Uop *U_ops );
void initHamiltonianGnd( struct addr_grid *grid, struct addr_op *op, struct addr_Uop *U_op );
void initHamiltonianEv( struct addr_grid *grid, struct addr_op *op, struct addr_Uop *Uop );
void defineGrid( struct addr_grid *grid );
void parseArgs( int argc, char **argv );
void splitOp( unsigned int steps, double dt, struct addr_grid grid*, struct addr_op *op, struct addr_Uop *Uop);

#endif

//###########################################################################################################//

