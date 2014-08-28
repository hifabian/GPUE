//###########################################################################################################//
#include "../host.c"
//###########################################################################################################//

//###########################################################################################################//
/*
*  Define CUDA variables and set-up required routines.
*/
//###########################################################################################################//

void setupFFT(	unsigned int *gridSize, 
		cufftHandle plan_xyz, 
		cufftHandle plan_xy, 
		cufftHandle plan_x_batchY)
{
	if( gridSize[2] != 0)
        	result = cufftPlan3d(&plan_xyz, xDim, yDim, zDim, CUFFT_Z2Z);
	if( gridSize[1] != 0)
        	result = cufftPlan2d(&plan_xy, xDim, yDim, CUFFT_Z2Z);
	if( gridSize[0] == gridSize[1] )
		result = cufftPlan1d(&plan_x_batchY, xDim, CUFFT_Z2Z, yDim);
	
	if(result != CUFFT_SUCCESS){
		printf("Result:=%d\n",result);
		printf("Error: Could not set-up CUFFT plan.\n");
     		exit (-1);
	}
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Allocate and free memory for the device. Non-required memory can be NULL.
*/
//###########################################################################################################//

void allocateMemoryDevice( struct addr_grid *grid, struct addr_Uop *U_op ){
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

void freeMemoryDevice(struct addr_Uop *U_op){

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
void allocateMemoryHost(	unsigned int selection, 
				unsigned int *gridSize, 
				struct addr_op *op, 
				struct addr_Uop *U_op ){

	ops->dq = (double*) malloc(sizeof(double)*grid->dim);
	ops->dp = (double*) malloc(sizeof(double)*grid->dim);
	ops->qMax = (double*) malloc(sizeof(double)*grid->dim);
	ops->pMax = (double*) malloc(sizeof(double)*grid->dim);

	if(selection & 0b00000001)
		op->V = (double*) malloc(sizeof(double)*gridSize[0]*gridSize[1]*gridSize[2]);
	if(selection & 0b00000010)
		op->K = (double*) malloc(sizeof(double)*gridSize[0]*gridSize[1]*gridSize[2]);
	if(selection & 0b00000100)
		op->XPy = (double*) malloc(sizeof(double)*gridSize[0]*gridSize[1]*gridSize[2]);
	if(selection & 0b00001000)
		op->YPx = (double*) malloc(sizeof(double)*gridSize[0]*gridSize[1]*gridSize[2]);
	
	if(selection & 0b00010000)
		U_op->opV = (double2*) malloc(sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	if(selection & 0b00100000)
		U_op->opK = (double2*) malloc(sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	if(selection & 0b01000000)
		U_op->opXPy = (double2*) malloc(sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	if(selection & 0b10000000)
		U_op->opYPx = (double2*) malloc(sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	if(selection & 0b100000000)
		U_op->wfc = (double2*) malloc(sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
}

/*
* Frees memory blocks. Use NULL in place of blocks to ignore.
*/
void freeMemoryHost( struct addr_op *op, struct addr_Uop *U_ops){

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

void initHamiltonianGnd( struct addr_op *op, struct addr_Uop *U_op ){
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
void initHamiltonianEv( struct addr_grid *grid, struct addr_op *op, struct addr_Uop *Uop )
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

void defineGrid(struct addr_grid *grid){
	
	for ( int i=0; i<grid->dim; ++i ){
		grid->dp[i] = PI/(grid->qMax[i]);
		grid->dq[i] = grid->qMax[i]/(grid->gridSize[i]>>1);

		grid->pMax[i] = grid->dp[i]*(grid->gridSize[i]>>1);

		grid->Q[i] = (double *) malloc(sizeof(double)*grid->gridSize[i]); 
		grid->P[i] = (double *) malloc(sizeof(double)*grid->gridSize[i]); 

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

void parseArgs(int argc, char **argv){
	while((c = get_opt(argc,argv,)))
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Do the evolution.
*/
//###########################################################################################################//

void splitOp(unsigned int steps, double dt, double2 *wfc, double *wfc_gpu, double2 *Uq, double2 *Up, double *Uxpy, double *Uypx){
	double2 *Uq_gpu_half = 
	//1 half step in position space
	vecVecMult_d2d2<<<,>>>(wfc_gpu, Uq_gpu, wfc_gpu);

	//Steps -1 momentum & position

	//1 full step momentum & 1 half step in position
}

//###########################################################################################################//

//###########################################################################################################//
/*
*	Get to the choppah!
*/
//###########################################################################################################//

int main(int argc, char **argv){
	parseArgs();
	double dt;
	unsigned int gridSize[3];
	double omega_V[3];
	double mass=1.0;

	//These contains the addresses of all the essential arrays for both CPU and GPU.
	struct addr_grid addr_grid;
	struct addr_op addr_op_host; 
	struct addr_Uop addr_Uop_host, addr_Uop_gpu; 

	defineGrid(&addr_grid);

	allocateMemoryHost(0x1ff, &addr_grid, &addr_op_host, &addr_Uop_host);
	allocateMemoryDevice(&addr_grid, &addr_Uop_gpu);

	//Imaginary time evolution
	initHamiltonianGnd( addr_grid, addr_op_host, addr_Uop_host, addr_Uop_gpu );
	splitOp(steps, dt, wfc, wfc_gpu, Uq_gpu, Up_gpu, Uxpy_gpu, Uypx_gpu, buffer);
	//Real time evolution
	initHamiltonianEv( gridSize, X, Y, Z, V, K, XPy, YPx, opV, opK, opXPy, opYPx );

	//Free the memory and go home.
	freeMemoryHost(V,K,XPy,YPx,opV,opK,opXPy,opYPx,wfc);
	freeMemoryDevice(wfc_gpu,Uq_gpu,Up_gpu,Uxpy_gpu,Uypx_gpu,buffer);
}

//###########################################################################################################//
