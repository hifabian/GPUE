#include"../include/gpu_functions.h"

//###########################################################################################################//
/*
*  Functions used to generate operators. Slower than createHamiltonian due to consecutive looping.
*/
//###########################################################################################################//

double* generate_V(int[] gridSize, double *gridX, double *gridY, double *gridZ, double mass, double[] omega){
	double* V = (double*) malloc(sizeof(double)*gridSize[0]*gridSize[1]*gridSize[2]);
	for(int k=gridSize[2]; k>0; --k){
		for(int j=gridSize[1]; j>0; --j){
			for(int i=gridSize[0]; i>0; --i){
				V[(k*gridSize[1] + j)*gridSize[0] + i] = operator_V(gridX[i],gridY[j],gridZ[k],mass,omega);
			}
		}
	}
	return *V;
}

double* generate_K(int[] gridSize, double *gridPX, double *gridPY, double *gridPZ, double mass){
	double* K = (double*) malloc(sizeof(double)*gridSize[0]*gridSize[1]*gridSize[2]);
	for(int k=gridSize[2]; k>0; --k){
		for(int j=gridSize[1]; j>0; --j){
			for(int i=gridSize[0]; i>0; --i){
				K[(k*gridSize[1] + j)*gridSize[0] + i] = operator_K(gridX[i],gridY[j],gridZ[k],mass);
			}
		}
	}
	return *K;
}

double2* generate_gndOperator(double *operator, int[] gridSize, double dt_hbar){
	double2 *gnd_op = (double2*) malloc(sizeof(double2)**gridSize[0]*gridSize[1]*gridSize[2]);
	
	for(int k=gridSize[2]; k>0; --k){
		for(int j=gridSize[1]; j>0; --j){
			for(int i=gridSize[0]; i>0; --i){
				gnd_op[(k*gridSize[1] + j)*gridSize[0] + i] = operator_gnd(operator[(k*gridSize[1] + j)*gridSize[0] + i], dt_hbar);
			}
		}
	}
	return *gnd_op;
}

double2* generate_evOperator(double *operator, int[] gridSize, double dt_hbar){
	double2 *ev_op = (double2*) malloc(sizeof(double2)**gridSize[0]*gridSize[1]*gridSize[2]);
	
	for(int k=gridSize[2]; k>0; --k){
		for(int j=gridSize[1]; j>0; --j){
			for(int i=gridSize[0]; i>0; --i){
				ev_op[(k*gridSize[1] + j)*gridSize[0] + i].x = cos(-operator[(k*gridSize[1] + j)*gridSize[0] + i]*dt_hbar);
				ev_op[(k*gridSize[1] + j)*gridSize[0] + i].y = sin(-operator[(k*gridSize[1] + j)*gridSize[0] + i]*dt_hbar);
			}
		}
	}
	return *ev_op;
}

//###########################################################################################################//
