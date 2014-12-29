/*
* operators.cu - GPUE2: Split Operator based GPU solver for Nonlinear
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

#include "../include/gpu_functions.h"
#include "../include/operators.h"

//###########################################################################################################//
/*
*  Functions used to generate operators. Slower than createHamiltonian due to consecutive looping.
*/
//###########################################################################################################//


double* generate_V(struct addr_grid *grid, double mass, double[] omega){
	unsigned int gridMax = 1;
	for(int i=0; i<addr_grid->dim;++i){
		gridMax *= grid->gridSize[i];
	}
	double* V = (double*) malloc( sizeof(double) * gridMax );
	for (int d = 0; d < grid->dim; ++i){

	}

	#if defined D3
	for(int k=0; k < grid->gridSize[2]; ++k){
	#endif
		#if defined D2 || defined D3
		for(int j=0; j < grid->gridSize[1]; ++j){
		#endif
			for(int i=0; i < grid->gridSize[0]; ++i){
				#if defined D3
				V[(k*gridSize[1] + j)*gridSize[0] + i] = operator_V(gridX[i], gridY[j], gridZ[k], mass,omega);
				#endif
				#if defined D2
				V[j*gridSize[0] + i] = operator_V(gridX[i], gridY[j], 0, mass, omega);
				#endif
				#if defined D1
				V[i] = operator_V(gridX[i], 0, 0, mass, omega);
				#endif
			}
		}
	}
	return *V;
}

double* generate_K(struct addr_grid *grid, double mass){
	double* K = (double*) malloc(sizeof(double)*gridSize[0]*gridSize[1]*gridSize[2]);
	for(int k = 0; k < gridSize[2]; ++k){
		for(int j=0; j < gridSize[1]; ++j){
			for(int i = 0; i < gridSize[0]; ++i){
				K[(k*gridSize[1] + j)*gridSize[0] + i] = operator_K(gridPX[i],gridPY[j],gridPZ[k],mass);
			}
		}
	}
	return *K;
}

double2* generate_gndOperator(double *operator, struct addr_grid *grid, double dt_hbar){
	double2 *gnd_op = (double2*) malloc(sizeof(double2)**gridSize[0]*gridSize[1]*gridSize[2]);

	for(int k=0; k<gridSize[2]; ++k){
		for(int j=0; j<gridSize[1]; ++j){
			for(int i=0; i<gridSize[0]; ++i){
				gnd_op[(k*gridSize[1] + j)*gridSize[0] + i] = operator_gnd(operator[(k*gridSize[1] + j)*gridSize[0] + i], dt_hbar);
			}
		}
	}
	return *gnd_op;
}

double2* generate_evOperator(double *operator, struct addr_grid *grid, double dt_hbar){
	double2 *ev_op = (double2*) malloc(sizeof(double2)**gridSize[0]*gridSize[1]*gridSize[2]);

	for(int k=0; k<gridSize[2]; ++k){
		for(int j=0; j<gridSize[1]; ++j){
			for(int i=0; i<gridSize[0]; ++i){
				ev_op[(k*gridSize[1] + j)*gridSize[0] + i].x = cos(-operator[(k*gridSize[1] + j)*gridSize[0] + i]*dt_hbar);
				ev_op[(k*gridSize[1] + j)*gridSize[0] + i].y = sin(-operator[(k*gridSize[1] + j)*gridSize[0] + i]*dt_hbar);
			}
		}
	}
	return *ev_op;
}

//###########################################################################################################//
