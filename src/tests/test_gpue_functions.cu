/*
* test_gpue_functions.cu - GPUE2: Split Operator based GPU solver for Nonlinear 
* Schrodinger Equation, Copyright (C) 2015, Lee J. O'Riordan. 

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

#include<assert.h>
#include<cuda.h>
#include<stdio.h>
#include<cuda_runtime.h>
#include "../colonel/gpu_functions.cu"


void test_scalVectMult(){
	int xDim, yDim;
	xDim=256;
	yDim=256;
        double *v1, *v1_gpu;
        v1 = (double*) malloc(sizeof(double)*xDim*yDim);
        cudaMalloc((void**) &v1_gpu, sizeof(double)*xDim*yDim);

        for(int i=0; i<xDim; ++i){
                for(int j=0; j<yDim; ++j){
                        v1[i*yDim + j] = 1.0;
                }
        }
        cudaMemcpy(v1_gpu, v1, sizeof(double)*xDim*yDim, cudaMemcpyHostToDevice);
        scalVecMult_dd<<<256,256>>>(v1_gpu, 2.0, v1_gpu);       
        cudaMemcpy(v1, v1_gpu, sizeof(double)*xDim*yDim, cudaMemcpyDeviceToHost);
        printf("%e\n",v1[0]);
        vecVecMult_dd<<<256,256>>>(v1_gpu, v1_gpu, v1_gpu);       
        cudaMemcpy(v1, v1_gpu, sizeof(double)*xDim*yDim, cudaMemcpyDeviceToHost);
        printf("%e\n",v1[0]);
	free(v1);cudaFree(v1_gpu);

	//#######################################################################

        double2 *v2, *v2_gpu;
        v2 = (double2*) malloc(sizeof(double2)*xDim*yDim);
        cudaMalloc((void**) &v2_gpu, sizeof(double2)*xDim*yDim);

        for(int i=0; i<xDim; ++i){
                for(int j=0; j<yDim; ++j){
                        v2[i*yDim + j].x = 1.0;
                        v2[i*yDim + j].y = 1.0;
                }
        }
        cudaMemcpy(v2_gpu, v2, sizeof(double2)*xDim*yDim, cudaMemcpyHostToDevice);
        scalVecMult_d2d<<<256,256>>>(v2_gpu, 2.0, v2_gpu);       
        cudaMemcpy(v2, v2_gpu, sizeof(double2)*xDim*yDim, cudaMemcpyDeviceToHost);
        printf("Re=%e	Im=%e\n",v2[0].x,v2[0].y);
        vecVecMult_d2d2<<<256,256>>>(v2_gpu, v2_gpu, v2_gpu);       
        cudaMemcpy(v2, v2_gpu, sizeof(double2)*xDim*yDim, cudaMemcpyDeviceToHost);
        printf("Re=%e	Im=%e\n",v2[0].x,v2[0].y);
}

void test_sum(){
	int xDim, yDim;
	const int threads = 128;
	xDim=256;
	yDim=256;
        double *v1, *v1_gpu;
        v1 = (double*) malloc(sizeof(double)*xDim*yDim);
        cudaMalloc((void**) &v1_gpu, sizeof(double)*xDim*yDim);

        for(int i=0; i<xDim; ++i){
                for(int j=0; j<yDim; ++j){
                        v1[i*yDim + j] = 1.0;
                }
        }
        cudaMemcpy(v1_gpu, v1, sizeof(double)*xDim*yDim, cudaMemcpyHostToDevice);
        for(int i=0; i<xDim; ++i){
                for(int j=0; j<yDim; ++j){
                       v1[0] += v1[i*yDim + j];
                }
        }
       	printf("%e\n",v1[0]);
	sumVector_d<threads><<<xDim*yDim/threads,threads,threads*sizeof(double)>>>(v1_gpu, v1_gpu, (unsigned int)threads*8);
        cudaMemcpy(v1, v1_gpu, sizeof(double)*xDim*yDim, cudaMemcpyDeviceToHost);
	for(int i=0; i<xDim; i++)
		for(int j=0; j<yDim; ++j)
        		printf("[%d,%d]=%e\n",i,j,v1[i*yDim + j]);
       	printf("%e\n",v1[0]);
}
void test_transpose(){
	int xDim, yDim;
	int temp;
	const int threads = 16;
	xDim=20;
	yDim=20;
        int *v1, *v2, *v1_gpu, *v2_gpu;
        v1 = (int*) malloc(sizeof(int)*xDim*yDim);
        v2 = (int*) malloc(sizeof(int)*xDim*yDim);
        cudaMalloc((void**) &v1_gpu, sizeof(int)*xDim*yDim);
        cudaMalloc((void**) &v2_gpu, sizeof(int)*xDim*yDim);

	//printf("Allocate blocks\n");
        for(int i=0; i<xDim; ++i){
                for(int j=0; j<yDim; ++j){
                        v1[i*yDim + j] = i*yDim + j;
                        v2[i*yDim + j] = i*yDim + j;
			//printf("V1[%d,%d]=%d	",i,j, v1[i*yDim +j]);
			printf("%d,%d,%d\n",i,j, v1[i*yDim +j]);
                }
        }
	printf("\n");

	printf("CPU Transpose 1 OP\n");
        cudaMemcpy(v1_gpu, v1, sizeof(int)*xDim*yDim, cudaMemcpyHostToDevice);
        for(int i=0; i<xDim; ++i){
                for(int j=0; j<yDim; ++j){
			v2[i*yDim + j] = v1[j*xDim + i];
			printf("V2[%d,%d]=%d	",i,j, v2[i*yDim + j]);
                }
		printf("\n");
        }
	printf("\n");

	printf("CPU Transpose 2 IP\n");
        for(int i=0; i<xDim; ++i){
                for(int j=0; j<yDim; ++j){
			temp = v1[i*yDim + j];
			v1[i*yDim + j] = v1[j*yDim + i];
			v1[j*xDim + i] = temp;
			printf("V1[%d,%d]=%d	",i,j, v1[i*yDim + j]);
                }
		printf("\n");
        }
	printf("\n");
	printf("GPU Transpose 1 IP\n");
	matTrans<<<1,128>>>(v1_gpu, v2_gpu);
	matTrans<<<1,128>>>(v1_gpu, v1_gpu);
        cudaMemcpy(v1, v1_gpu, sizeof(int)*xDim*yDim, cudaMemcpyDeviceToHost);
        cudaMemcpy(v2, v2_gpu, sizeof(int)*xDim*yDim, cudaMemcpyDeviceToHost);
	for(int i=0; i<xDim; i++){
		for(int j=0; j<yDim; ++j){
        		printf("G1[%d,%d]=%d	",i,j,v1[i*yDim + j]);
		}
		printf("\n");
	}
	printf("\n");
	printf("GPU Transpose 2 OP\n");
	for(int i=0; i<xDim; i++){
		for(int j=0; j<yDim; ++j){
        		printf("G2[%d,%d]=%d	",i,j,v2[i*yDim + j]);
		}
		printf("\n");
	}
}

int main(){
	//test_scalVectMult();
	//test_sum();
	test_transpose();
	return 0;
}
