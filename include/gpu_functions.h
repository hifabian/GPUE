#ifndef GPU_FUNCTIONS_H 
#define GPU_FUNCTIONS_H 
__host__ __device__ double compMagnitude(double2 cmp1);
__host__ __device__ double2 realCompMult(double rl, double2 cmp);
__host__ __device__ double2 compCompMult(double2 cmp1, double2 cmp2);
__host__ __device__ double2 compSum(double2 cmp1, double2 cmp2);
__host__ __device__ double2 conj(double2 cmp);

__device__ unsigned int getGid3d3d();
__global__ void scalVecMult_d2d(double2 *vecIn, double scalIn, double2 *vecOut);
__global__ void scalVecMult_dd(double *vecIn, double scalIn, double *vecOut);
__global__ void scalVecMult_ii(int *vecIn, int scalIn, int *vecOut);
__global__ void scalVecMult_d2d2(double2 *vecIn, double2 scalIn, double2 *vecOut);

__global__ void vecVecMult_d2d2(double2 *vec1In, double2 *vec2In, double2 *vecOut);
__global__ void vecVecMult_d2d(double2 *vec1In, double *vec2In, double2 *vecOut);
__global__ void vecVecMult_dd(double *vec1In, double *vec2In, double *vecOut);
__global__ void vecVecMult_ii(int *vec1In, int *vec2In, int *vecOut);
#endif
