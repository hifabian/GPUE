#include "../include/gpu_functions.h"

#define hbar 1.0
double dt_hbar;

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

void allocateMemoryDevice(	int *gridSize, 
				double2* wfc_gpu, 
				double2* Uq_gpu, 
				double2* Up_gpu, 
				double2* Uxpy_gpu, 
				double2* Uypx_gpu, 
				double2* buffer)
{
	cudaMalloc((void**) &wfc_gpu, sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	cudaMalloc((void**) &Uq_gpu, sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	cudaMalloc((void**) &Up_gpu, sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	cudaMalloc((void**) &XPy_gpu, sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	cudaMalloc((void**) &YPx_gpu, sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	cudaMalloc((void**) &buffer, sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
}

void freeMemoryDevice(	double2* wfc_gpu, 
			double2* Uq_gpu, 
			double2* Up_gpu, 
			double2* Uxpy_gpu, 
			double2* Uypx_gpu, 
			double2* buffer )
{
	cudaFree(double2* wfc_gpu);
	cudaFree(double2* Uq_gpu);
	cudaFree(double2* Up_gpu);
	cudaFree(double2* Uxpy_gpu);
	cudaFree(double2* Uypx_gpu);
	cudaFree(double2* Uypx_gpu);
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Allocate and free memory for the problem. Non-required memory can be NULL.
*/
//###########################################################################################################//


/*
* @selection Used as a bitwise operation to select which pointers to allocate memory for. 0b1111111111 (0x1ff, 511) selects all
*/
void allocateMemoryHost(	unsigned int selection, 
				unsigned int *gridSize, 
				double *V, double *K, 
				double *XPy, double *YPx, 
				double2 *opV, double2 *opK, 
				double2 *opXPy, double2 *opYPx, 
				double2 *wfc){

	if(selection & 0b00000001)
		double* V = (double*) malloc(sizeof(double)*gridSize[0]*gridSize[1]*gridSize[2]);
	if(selection & 0b00000010)
		double* K = (double*) malloc(sizeof(double)*gridSize[0]*gridSize[1]*gridSize[2]);
	if(selection & 0b00000100)
		double* XPy = (double*) malloc(sizeof(double)*gridSize[0]*gridSize[1]*gridSize[2]);
	if(selection & 0b00001000)
		double* YPx = (double*) malloc(sizeof(double)*gridSize[0]*gridSize[1]*gridSize[2]);
	
	if(selection & 0b00010000)
		double2 *opV = (double2*) malloc(sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	if(selection & 0b00100000)
		double2 *opK = (double2*) malloc(sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	if(selection & 0b01000000)
		double2 *opXPy = (double2*) malloc(sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	if(selection & 0b10000000)
		double2 *opYPx = (double2*) malloc(sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	if(selection & 0b100000000)
		double2 *wfc = (double2*) malloc(sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
}

/*
* Frees memory blocks. Use NULL in place of blocks to ignore.
*/
void freeMemoryHost(	double *V, 
			double *K, 
			double *XPy, 
			double *YPx, 
			double2 *opV, 
			double2 *opK, 
			double2 *opXPy, 
			double2 *opYPx )
{
	free(V); free(K); free(XPy); free(YPx); free(opV); free(opK); free(opXPy); free(opYPx); free(wfc); 
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Initialise operators for imaginary time evolution Hamiltonian
*/
//###########################################################################################################//

void initHamiltonianGnd(	unsigned int* gridSize, 
				double *X, 
				double *Y, 
				double *Z, 
				double *V, 
				double *K, 
				double *XPy, 
				double *YPx, 
				double2 *opV, 
				double2 *opK, 
				double2 *opXPy, 
				double2 *opYPx )
{
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
void initHamiltonianEv(	unsigned int* gridSize, 
			double *X, 
			double *Y, 
			double *Z, 
			double *V, 
			double *K, 
			double *XPy, 
			double *YPx, 
			double2 *opV, 
			double2 *opK, 
			double2 *opXPy, 
			double2 *opYPx )
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
*  Initialise operators for real time evolution Hamiltonian.
*/
//###########################################################################################################//

void splitOp(double dt, double2 *wfc, double2 *Uq, double2 *Up){
	
}

void parseArguments(int argc, char **argv){
	while((c = get_opt(argc,argv,)))
}


void defineGrid(	unsigned int *gridSize, 
			double *qMax, 
			double *pMax, 
			double *dq, 
			double *dp, 
			double *X, 
			double *Y, 
			double *Z, 
			double *PX, 
			double *PY, 
			double *PZ)
{
	dp[0] = PI/(qMax[0]);
	dp[1] = PI/(qMax[1]);
	dp[2] = PI/(qMax[2]);

	dq[0] = qMax[0]/(gridSize[0]>>1);
	dq[1] = qMax[1]/(gridSize[1]>>1);
	dq[2] = qMax[2]/(gridSize[2]>>1);

	pMax[0] = dp[0]*(gridSize[0]>>1);
	pMax[1] = dp[1]*(gridSize[1]>>1);
	pMax[2] = dp[2]*(gridSize[2]>>1);

	X = (double *) malloc(sizeof(double) * gridSize[0]);
	Y = (double *) malloc(sizeof(double) * gridSize[1]);
	Z = (double *) malloc(sizeof(double) * gridSize[2]);

	PX = (double *) malloc(sizeof(double) * gridSize[0]);
	PY = (double *) malloc(sizeof(double) * gridSize[1]);
	PZ = (double *) malloc(sizeof(double) * gridSize[2]);

/*
* Pos and Mom grids. Assumes the grids are equally sized.
*/	if(gridSize[0]==gridSize[1])
	for(i=0; i < gridSize[0]/2; ++i){
		X[i] = -qMax[0] + (i+1)*dq[0];
		X[i + (gridSize[0]/2)] = (i+1)*dq[0];

		Y[i] = -qMax[1] + (i+1)*dq[1];
		Y[i + (gridSize[1]/2)] = (i+1)*dq[1];

		Z[i] = -qMax[2] + (i+1)*dq[2];
		Z[i + (gridSize[2]/2)] = (i+1)*dq[2];

		XP[i] = (i+1)*dp[0];
		XP[i + (gridSize[0]/2)] = -pMax[0] + (i+1)*dp[0];

		YP[i] = (i+1)*dp[1];
		YP[i + (gridSize[1]/2)] = -pMax[1] + (i+1)*dp[1];

		ZP[i] = (i+1)*dp[2];
		ZP[i + (gridSize[2]/2)] = -pMax[2] + (i+1)*dp[2];

	}
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Do the magic.
*/
//###########################################################################################################//

int main(int argc, char **argv){
	double qMax[3]; double pMax[3];
	double dq[3]; double dp[3];

	double omega_V[3];
	unsigned int gridSize[3];
	double mass=1.0;
	double *X, *Y, *Z, *V, *K, *XPy, *YPx; 
	double2 *wfc, *opV, *opK, *opXPy, *opYPx, *buffer;
	double2 *wfc_gpu, *Uq_gpu, *Up_gpu, *Uxpy_gpu, *Uypx_gpu, *buffer;
	
	//Allocate memory on host and device
	allocateMemoryHost(0x1ff, gridSize, V, K, XPy, YPx, opV, opK, opXPy, opYPx, wfc);
	allocateMemoryDevice(gridSize, wfc_gpu, Uq_gpu, Up_gpu, Uxpy_gpu, Uypx_gpu, buffer);

	//Imaginary time evolution
	initHamiltonianGnd( gridSize, X, Y, Z, V, K, XPy, YPx, opV, opK, opXPy, opYPx );

	//Real time evolution
	initHamiltonianEv( gridSize, X, Y, Z, V, K, XPy, YPx, opV, opK, opXPy, opYPx );

	//Free the memory and go home.
	freeMemoryHost(V,K,XPy,YPx,opV,opK,opXPy,opYPx,wfc);
	freeMemoryDevice(wfc_gpu,Uq_gpu,Up_gpu,Uxpy_gpu,Uypx_gpu,buffer);
}

//###########################################################################################################//
