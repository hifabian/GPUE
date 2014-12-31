/*
* host.cu - GPUE2: Split Operator based GPU solver for Nonlinear
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

#include "../include/host.h"
#include "../include/state.hpp"
#include "../include/operators.h"
#include "../include/gpu_functions.h"

//###########################################################################################################//

//###########################################################################################################//
/*
*  Define CUDA variables and set-up required routines.
*/
//###########################################################################################################//

static void Host::setupFFT( State::addr_grid *grid,
		cufftHandle plan_xyz,
		cufftHandle plan_xy,
		cufftHandle plan_x_batchY)
{
	if( grid->dim = 3)
        	result = cufftPlan3d(&plan_xyz, xDim, yDim, zDim, CUFFT_Z2Z);
	else if( grid->dim = 2){
        	result = cufftPlan2d(&plan_xy, xDim, yDim, CUFFT_Z2Z);
	if( grid->gridSize[0] == grid->gridSize[1] )
			result = cufftPlan1d(&plan_x_batchY, xDim, CUFFT_Z2Z, yDim);
	}
	else{
		printf("Error: Runtype not currently supported.\n");
		exit(1);
	}
	if(result != CUFFT_SUCCESS){
		printf("Result:=%d\n",result);
		printf("Error: Could not set-up CUFFT plan.\n");
     		exit (2);
	}
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Allocate and free memory for the device. Non-required memory can be NULL.
*/
//###########################################################################################################//

static void Host::allocateMemoryDevice( State::addr_grid *grid, State::addr_Uop *U_op ){
	unsigned int gridMax = 1;
	for(int i=0; i<addr_grid->dim;++i){
		gridMax *= addr_grid->gridSize[i];
	}
	cudaMalloc((void**) &(U_op->wfc), sizeof(double2)*gridMax);
	cudaMalloc((void**) &(U_op->opV), sizeof(double2)*gridMax);
	cudaMalloc((void**) &(U_op->opK), sizeof(double2)*gridMax);
	cudaMalloc((void**) &(U_op->opXPy), sizeof(double2)*gridMax);
	cudaMalloc((void**) &(U_op->opYPx), sizeof(double2)*gridMax);
	cudaMalloc((void**) &(U_op->buffer), sizeof(double2)*gridMax);
}

static void Host::freeMemoryDevice( State::addr_Uop *U_op){
	cudaFree( U_op->wfc );
	cudaFree( U_op->opV );
	cudaFree( U_op->opK );
	cudaFree( U_op->opXPy );
	cudaFree( U_op->opYPx );
	cudaFree( U_op->buffer );
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Allocate and free memory for the host. Non-required memory can be NULL.
*/
//###########################################################################################################//


/*
* @selection Used as a bitwise operation to select which pointers to allocate memory for. 0b1111111111 (0x1ff, 511) selects all
*/
static void Host::allocateMemoryHost( unsigned int selection, State::addr_grid *grid, State::addr_op *op, State::addr_Uop *U_op ){

	ops->dq = (double*) malloc(sizeof(double)*grid->dim);
	ops->dp = (double*) malloc(sizeof(double)*grid->dim);
	ops->qMax = (double*) malloc(sizeof(double)*grid->dim);
	ops->pMax = (double*) malloc(sizeof(double)*grid->dim);

	if(selection & 0b000000001)
		op->V = (double*) malloc(sizeof(double)*grid->gridMax );
	if(selection & 0b000000010)
		op->K = (double*) malloc(sizeof(double)*grid->gridMax );
	if(selection & 0b000000100)
		op->XPy = (double*) malloc(sizeof(double)*grid->gridMax );
	if(selection & 0b000001000)
		op->YPx = (double*) malloc(sizeof(double)*grid->gridMax );

	if(selection & 0b000010000)
		U_op->opV = (double2*) malloc(sizeof(double2)*grid->gridMax );
	if(selection & 0b000100000)
		U_op->opK = (double2*) malloc(sizeof(double2)*grid->gridMax );
	if(selection & 0b001000000)
		U_op->opXPy = (double2*) malloc(sizeof(double2)*grid->gridMax );
	if(selection & 0b010000000)
		U_op->opYPx = (double2*) malloc(sizeof(double2)*grid->gridMax );
	if(selection & 0b100000000)
		U_op->wfc = (double2*) malloc(sizeof(double2)*grid->gridMax );
}

/*
* Frees memory blocks. Use NULL in place of blocks to ignore.
*/
static void Host::freeMemoryHost( State::addr_op *op, State::addr_Uop *U_op){

	free(op->V);		free(op->K);		free(op->XPy);		free(op->YPx);
	free(U_op->opV);	free(U_op->opK);	free(U_op->opXPy);	free(U_op->opYPx);
	free(U_op->wfc);
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Initialise operators for imaginary time evolution Hamiltonian
*/
//###########################################################################################################//

static void Host::initHamiltonianGnd( State::addr_grid *grid, State::addr_op *op, State::addr_Uop *U_op ){
	for(int d=0; d<){

	}

	for(int k=0; k<gridSize[2]; ++k){
		for(int j=0; j<gridSize[1]; ++j){
			for(int i=0; i<gridSize[0]; ++i){

				V[(k*gridSize[1] + j)*gridSize[0] + i] = operator_V( X[i], Y[j], Z[k], mass, omega_V); //may need to set Z[k] to 0 here
				K[(k*gridSize[1] + j)*gridSize[0] + i] = operator_V( X[i], Y[j], Z[k], mass);
				XPy[(k*gridSize[1] + j)*gridSize[0] + i] = X[i]*PY[j];
				YPx[(k*gridSize[1] + j)*gridSize[0] + i] = -Y[j]*PX[i];

				opV[(k*gridSize[1] + j)*gridSize[0] + i] = operator_gnd( V[(k*gridSize[1] + j)*gridSize[0] + i], dt_hbar);
				opK[(k*gridSize[1] + j)*gridSize[0] + i] = operator_gnd( K[(k*gridSize[1] + j)*gridSize[0] + i], dt_hbar);

				opXPy[(k*gridSize[1] + j)*gridSize[0] + i] = operator_gnd( Omega*XPy[(k*gridSize[1] + j)*gridSize[0] + i], dt_hbar);
				opYPx[(k*gridSize[1] + j)*gridSize[0] + i] = operator_gnd( Omega*YPx[(k*gridSize[1] + j)*gridSize[0] + i], dt_hbar);
			}
		}
	}
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Initialise operators for real time evolution Hamiltonian.
*/
//###########################################################################################################//

/*
*  Must have V, K, XPY YPX in memory, otherwise this routine will fall over.
*/
static void Host::initHamiltonianEv( State::addr_grid *grid, State::addr_op *op, State::addr_Uop *Uop )
{
	if(V==NULL || K==NULL || XPy==NULL || YPx==NULL){
		printf("The required arrays were not stored in memory. Please load them before use.");
		exit(1);
	}
	for(int k=0; k<gridSize[2]; ++k){
		for(int j=0; j<gridSize[1]; ++j){
			for(int i=0; i<gridSize[0]; ++i){

				opV[(k*gridSize[1] + j)*gridSize[0] + i] = operator_gnd( V[(k*gridSize[1] + j)*gridSize[0] + i], dt_hbar);
				opK[(k*gridSize[1] + j)*gridSize[0] + i] = operator_gnd( K[(k*gridSize[1] + j)*gridSize[0] + i], dt_hbar);

				opXPy[(k*gridSize[1] + j)*gridSize[0] + i] = operator_gnd( Omega*XPy[(k*gridSize[1] + j)*gridSize[0] + i], dt_hbar);
				opYPx[(k*gridSize[1] + j)*gridSize[0] + i] = operator_gnd( Omega*YPx[(k*gridSize[1] + j)*gridSize[0] + i], dt_hbar);
			}
		}
	}
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Initialise grids based upon system size and dimensionality.
*/
//###########################################################################################################//

static void Host::defineGrid( State::addr_grid *grid ){
	grid->Q = (double **) malloc( sizeof(double*)*grid->dim );
	grid->P = (double **) malloc( sizeof(double*)*grid->dim );

	for ( int i=0; i<grid->dim; ++i ){
		grid->dp[i] = PI/(grid->qMax[i]);
		grid->dq[i] = grid->qMax[i]/(grid->gridSize[i]>>1);

		grid->pMax[i] = grid->dp[i]*(grid->gridSize[i]>>1);

		grid->Q[i] = (double *) malloc( sizeof(double)*grid->gridSize[i] );
		grid->P[i] = (double *) malloc( sizeof(double)*grid->gridSize[i] );

		for( int j = 0; j < grid->gridSize[i]>>1; ++j){
			grid->Q[i][j] = -grid->qMax[i] + (j+1)*grid->dq[i];
			grid->Q[i][j + grid->gridSize[i]>>1] = (j+1)*grid->dq[i];

			grid->P[i][j] = (j+1)*grid->dp[i];
			grid->P[i][j + grid->gridSize[i]>>1] = -grid->pMax[i] + (j+1)*grid->dp[i];
		}
	}
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Parse command-line arguments.
*/
//###########################################################################################################//

static void Host::parseArgs(int argc, char **argv){
	while((c = get_opt(argc,argv,)))
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Do the evolution.
*/
//###########################################################################################################//

static void Host::splitOp(unsigned int steps, double dt, double2 *wfc, double *wfc_gpu, double2 *Uq, double2 *Up, double *Uxpy, double *Uypx){
	double2 *Uq_gpu_half =
	//1 half step in position space
	vecVecMult_d2d2<<<,>>>(wfc_gpu, Uq_gpu, wfc_gpu);

	//Steps -1 momentum & position

	//1 full step momentum & 1 half step in position
}

//###########################################################################################################//
