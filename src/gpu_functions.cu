/*
* gpu_functions.cu - GPUE2: Split Operator based GPU solver for Nonlinear
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
#include "../include/constants.h"

#ifndef T32B4
	#define TILE_DIM 32 //small segment to be computed
	#define BLOCK_ROW 8 // sum of the two should match threads
#endif

//###########################################################################################################//
/*
*  Returns the global (not grid) index for the relevant thread in a 3d grid 3d block fashion. I will use 1d 1d mostly here though.
*/
//###########################################################################################################//

__device__ unsigned int getGid3d3d(){
	int gid = blockDim.x * ( ( blockDim.y * ( ( blockIdx.z * blockDim.z + threadIdx.z ) + blockIdx.y ) + threadIdx.y ) + blockIdx.x ) + threadIdx.x;
	return gid;
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Defines values for V, K and subsequent operators at the specified point.
*/
//###########################################################################################################//

inline __host__ __device__ double operator_V(double X, double Y, double Z, double mass, double *omega){
	return 0.5*mass*( pow(X*omega[0],2) + pow(Y*omega[1],2) + pow(Z*omega[2],2) );
}

inline __host__ __device__ double operator_K(double PX, double PY, double PZ, double mass){
	return (hbar*hbar/(2*mass))*( pow(PX,2) + pow(PY,2) + pow(PZ,2) );
}

inline __host__ __device__ double2 operator_gnd(double oper, double dt_hbar){
	double2 result;
	result.x = exp(-oper*dt_hbar);
	result.y = 0.0;
	return result;
}

inline __host__ __device__ double2 operator_ev(double oper, double dt_hbar){
	double2 result;
	result.x = cos(-oper*dt_hbar);
	result.y = sin(-oper*dt_hbar);
	return result;
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Scalar x Vector functions. Double-Double, Double-Complex, Complex-Complex, Int-Int
*/
//###########################################################################################################//

__global__ void scalVecMult_d2d(double2 *vecIn, double scalIn, double2 *vecOut){
	unsigned int i = getGid3d3d();
	vecOut[i] = realCompMult(scalIn,vecIn[i]);
}

__global__ void scalVecMult_dd(double *vecIn, double scalIn, double *vecOut){
	unsigned int i = getGid3d3d();
	vecOut[i] = scalIn*vecIn[i];
}

__global__ void scalVecMult_ii(int *vecIn, int scalIn, int *vecOut){
	unsigned int i = getGid3d3d();
	vecOut[i] = scalIn*vecIn[i];
}

__global__ void scalVecMult_d2d2(double2 *vecIn, double2 scalIn, double2 *vecOut){
	unsigned int i = getGid3d3d();
	vecOut[i] = compCompMult(scalIn, vecIn[i]);
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Vector x Vector functions. Double-Double, Double-Complex, Complex-Complex, Int-Int
*/
//###########################################################################################################//

__global__ void vecVecMult_d2d2(double2 *vec1In, double2 *vec2In, double2 *vecOut){
	unsigned int i = getGid3d3d();
	double2 result = compCompMult(vec1In[i],vec2In[i]);
	vecOut[i] = result;
}

__global__ void vecVecMult_d2d(double2 *vec1In, double *vec2In, double2 *vecOut){
	unsigned int i = getGid3d3d();
	vecOut[i] = realCompMult(vec2In[i],vec1In[i]);
}

__global__ void vecVecMult_dd(double *vec1In, double *vec2In, double *vecOut){
	unsigned int i = getGid3d3d();
	vecOut[i] = vec1In[i]*vec2In[i];
}

__global__ void vecVecMult_ii(int *vec1In, int *vec2In, int *vecOut){
	unsigned int i = getGid3d3d();
	vecOut[i] = vec1In[i]*vec1In[i];
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Matrix transpose function. Double-Double, Double-Complex, Complex-Complex, Int-Int
*/
//###########################################################################################################//

template<typename T>
__global__ void matTrans(T *vecIn, T *vecOut){

	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for(int j=0; j<blockDim.x; j+=blockDim.x){
		vecOut[ x*width + (y+j)  ] = vecIn[(y+j)*width + x];
	}

}


template<typename T>
__global__ void matTrans2(T *vecIn, T *vecOut){
	__shared__ double tile[TILE_DIM][TILE_DIM];
	int x = blockIdx.x * TILE_DIM + threadIdx.x;
	int y = blockIdx.y * TILE_DIM + threadIdx.y;
	int width = gridDim.x * TILE_DIM;

	for(int j=0; j<blockDim.x; j+=blockDim.x){
		tile[threadIdx.y + j][threadIdx.x] = vecIn[(y+j)*width + x];
	}
	__syncthreads();
	x = blockIdx.y * TILE_DIM + threadIdx.x;
	y = blockIdx.x * TILE_DIM + threadIdx.y;
	for (int j = 0; j < TILE_DIM; j += BLOCK_ROW){
		vecOut[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
	}

}
//###########################################################################################################//

//###########################################################################################################//
/*
*  Parallel summation. Double, Complex
*/
//###########################################################################################################//

//Taken from cuda slide 1.1-beta
/*
* n is the number of elements to sum by a single thread. Values of 64-2048 are best, allegedly.
*/
template <unsigned int blockSize>
__global__ void sumVector_d(double* vecIn, double* vecOut, unsigned int n){
	extern __shared__ double sdata_d[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata_d[tid]=0.0;

	while ( i < n ){
		sdata_d[tid] += vecIn[i] + vecIn[i + blockSize];
		i += gridSize;
	}
	__syncthreads();
	if(blockSize >= 1024) { if(tid < 512) { sdata_d[tid] += sdata_d[tid+512];} __syncthreads; }
	if(blockSize >= 512) { if(tid < 256) { sdata_d[tid] += sdata_d[tid+256];} __syncthreads; }
	if(blockSize >= 256) { if(tid < 128) { sdata_d[tid] += sdata_d[tid+128];} __syncthreads; }
	if(blockSize >= 128) { if(tid < 64) { sdata_d[tid] += sdata_d[tid+64];} __syncthreads; }

	if (tid < 32){
		if(blockSize >= 64) sdata_d[tid] += sdata_d[tid+32];
		if(blockSize >= 32) sdata_d[tid] += sdata_d[tid+16];
		if(blockSize >= 16) sdata_d[tid] += sdata_d[tid+8];
		if(blockSize >= 8) sdata_d[tid] += sdata_d[tid+4];
		if(blockSize >= 4) sdata_d[tid] += sdata_d[tid+2];
		if(blockSize >= 2) sdata_d[tid] += sdata_d[tid+1];
	}
	if(tid == 0) vecOut[blockIdx.x] = sdata_d[0];
}

template <unsigned int blockSize>
__global__ void sumVector_d2(double2* vecIn, double2* vecOut, unsigned int n){
	extern __shared__ double2 sdata_d2[];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata_d2[tid].x=0.0;	sdata_d2[tid].y=0.0;

	while ( i < n ){
		sdata_d2[tid].x += vecIn[i].x + vecIn[i + blockSize].x;
		sdata_d2[tid].y += vecIn[i].y + vecIn[i + blockSize].y;
		i += gridSize;
	}
	if(blockSize >= 1024) { if(tid < 512) { sdata_d2[tid].x += sdata_d2[tid+512].x; sdata_d2[tid].y += sdata_d2[tid+512].y; } __syncthreads; }
	if(blockSize >= 512)  { if(tid < 256) { sdata_d2[tid].x += sdata_d2[tid+256].x; sdata_d2[tid].y += sdata_d2[tid+256].y; } __syncthreads; }
	if(blockSize >= 256)  { if(tid < 128) { sdata_d2[tid].x += sdata_d2[tid+128].x; sdata_d2[tid].y += sdata_d2[tid+128].y; } __syncthreads; }
	if(blockSize >= 128)  { if(tid < 64)  { sdata_d2[tid].x += sdata_d2[tid+64].x;  sdata_d2[tid].y += sdata_d2[tid+64].y;  } __syncthreads; }

	if (tid < 32){
		if(blockSize >= 64){ sdata_d2[tid].x += sdata_d2[tid+32].x; sdata_d2[tid].y += sdata_d2[tid+32].y; }
		if(blockSize >= 32){ sdata_d2[tid].x += sdata_d2[tid+16].x; sdata_d2[tid].y += sdata_d2[tid+16].y; }
		if(blockSize >= 16){ sdata_d2[tid].x += sdata_d2[tid+8].x;  sdata_d2[tid].y += sdata_d2[tid+8].y; }
		if(blockSize >= 8){  sdata_d2[tid].x += sdata_d2[tid+4].x;  sdata_d2[tid].y += sdata_d2[tid+4].y; }
		if(blockSize >= 4){  sdata_d2[tid].x += sdata_d2[tid+2].x;  sdata_d2[tid].y += sdata_d2[tid+2].y; }
		if(blockSize >= 2){  sdata_d2[tid].x += sdata_d2[tid+1].x;  sdata_d2[tid].y += sdata_d2[tid+1].y; }
	}
	if(tid == 0) vecOut[blockIdx.x] = sdata_d2[0];
}

//###########################################################################################################//

//###########################################################################################################//
/*
*  Device functions for dealing with complex numbers.
*/
//###########################################################################################################//

__host__ __device__ double compMagnitude(double2 cmp1){
	return sqrt(cmp1.x*cmp1.x + cmp1.y*cmp1.y);
}

__host__ __device__ double2 realCompMult(double rl, double2 cmp){
	double2 result;
	result.x = rl*cmp.x;
	result.y = rl*cmp.y;
	return result;
}
__host__ __device__ double2 compCompMult(double2 cmp1, double2 cmp2){
	double2 result;
	result.x = (cmp1.x*cmp2.x - cmp1.y*cmp2.y);
	result.y = (cmp1.x*cmp2.y + cmp1.y*cmp2.x);
	return result;
}
__host__ __device__ double2 compCompSum(double2 cmp1, double2 cmp2){
	double2 result;
	result.x = cmp1.x + cmp2.x;
	result.y = cmp1.y + cmp2.y;
	return result;
}

//###########################################################################################################//
