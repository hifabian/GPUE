//###########################################################################################################//

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cufft.h>
#include "../include/gpu_functions.h"
#include "../include/constants.h"


//###########################################################################################################//
/*
*  These structs define the addresses to the grid arrays and operators respectively.
*/
//###########################################################################################################//

struct addr_grid {
	unsigned int dim;
	unsigned int *gridSize;
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
