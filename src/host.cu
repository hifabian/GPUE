#define hbar 1.0
double dt_hbar;



//###########################################################################################################//
//###########################################################################################################//
/*
*  Define CUDA variables and set-up required routines.
*/
//###########################################################################################################//


void setupFFT(){
	if(zDim!=0)
        	result = cufftPlan3d(&plan_xyz, xDim, yDim, zDim, CUFFT_Z2Z);
	if(yDim!=0)
        	result = cufftPlan2d(&plan_xy, xDim, yDim, CUFFT_Z2Z);
	if(xDim==yDim)
		result = cufftPlan1d(&plan_x_batchy, xDim, CUFFT_Z2Z, yDim);
	if(result != CUFFT_SUCCESS){
		printf("Result:=%d\n",result);
		printf("Error: Could not set-up CUFFT plan.\n");
     		exit (-1);
	}
}

void allocateMemoryDevice(int *gridSize, double2* wfc_gpu, double2* Uq_gpu, double2* Up_gpu, double2* XPy_gpu, double2* YPx_gpu){
	cudaMalloc((void**) &wfc_gpu, sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	cudaMalloc((void**) &Uq_gpu, sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	cudaMalloc((void**) &Up_gpu, sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	cudaMalloc((void**) &XPy_gpu, sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
	cudaMalloc((void**) &YPx_gpu, sizeof(double2)*gridSize[0]*gridSize[1]*gridSize[2]);
}

void freeMemoryDevice(double2* wfc_gpu, double2* Uq_gpu, double2* Up_gpu, double2* XPy_gpu, double2* YPx_gpu){
	cudaFree(double2* wfc_gpu);
	cudaFree(double2* Uq_gpu);
	cudaFree(double2* Up_gpu);
	cudaFree(double2* XPy_gpu);
	cudaFree(double2* YPx_gpu);
}

//###########################################################################################################//
/*
*  Allocate and free memory for the problem. Non-required memory can be NULL.
*/
//###########################################################################################################//


/*
* @selection Used as a bitwise operation to select which pointers to allocate memory for. 0b11111111 (0xff, 255) selects all
*/
void allocateMemoryHost(unsigned int selection, int *gridSize, double *V, double *K, double *XPy, double *YPx, double2 *opV, double2 *opK, double2 *opXPy, double2 *opYPx){

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
}

/*
* Frees memory blocks. Use NULL in place of blocks to ignore.
*/
void freeMemoryHost( double *V, double *K, double *XPy, double *YPx, double2 *opV, double2 *opK, double2 *opXPy, double2 *opYPx ){
	free(V); free(K); free(XPy); free(YPx); free(opV); free(opK); free(opXPy); free(opYPx);  
}

//###########################################################################################################//

int createHamiltonianGnd( int* gridSize,  ){
	double omega_V[3];
	double mass=1.0;
	double *V, *K, *XPy, *YPx; 
	double2 *opV, *opK, *opXPy, *opYPx;

	for(int k=gridSize[2]; k>0; --k){
		for(int j=gridSize[1]; j>0; --j){
			for(int i=gridSize[0]; i>0; --i){

				V[(k*gridSize[1] + j)*gridSize[0] + i] = operator_V(X[i],Y[j],Z[k],mass,omega_V);
				K[(k*gridSize[1] + j)*gridSize[0] + i] = operator_V(X[i],Y[j],Z[k],mass);

				opV[(k*gridSize[1] + j)*gridSize[0] + i] = operator_gnd(V[(k*gridSize[1] + j)*gridSize[0] + i],dt_hbar);
				opK[(k*gridSize[1] + j)*gridSize[0] + i] = operator_gnd(K[(k*gridSize[1] + j)*gridSize[0] + i],dt_hbar);

			}
		}
	}

	return 0;
}


int splitOp(double dt, double2 *wfc, double2 *Uq, double2 *Up){
	
}



void main(){

}

