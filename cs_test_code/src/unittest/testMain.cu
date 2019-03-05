#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file
#include <cuda_runtime.h>
#include <gsl/gsl_sf_hyperg.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cufft.h>
#include <algorithm>
#include <vector>
#include "catch.hpp"
#include "../easyloggingcpp/easylogging++.h"
#include "../mxutils.h"

typedef double (*ConvFunc)(const double, const double);

using namespace std;

// Initialize logging system
INITIALIZE_EASYLOGGINGPP

__global__ void findID(double *a, int n){

    // First we need to find our global threadID
    int tPosX = blockIdx.x * blockDim.x + threadIdx.x;
    // Make sure we are not out of range
    if (tPosX < n){
        a[tPosX] = tPosX;
    }
}

__device__ double dplus(double a, double b){
  return a+b;
  //return 0;
}

__device__ double dprod(double a, double b){
  return a*b;
}

__global__ void convolve(double * arr, size_t n, ConvFunc * func){
  // First we need to find our global threadID
  int tPosX = blockIdx.x * blockDim.x + threadIdx.x;
  // Make sure we are not out of range
  if (tPosX >= n)
    return;
  arr[tPosX] = (*func)((double)tPosX, (double)tPosX);
}

__global__ void copyFuncPtr(ConvFunc * d_funcptr, int n){
  switch(n){
    case 0:
      d_funcptr[0] = &dprod;
      //printf("=====%f=======\n", (*d_funcptr[0])(1,1));
      return;
    case 1:
      d_funcptr[0] = &dplus;
      //printf("=====%f=======\n", (*d_funcptr[0])(1,1));
      return;
    default:
      return;
  }
}

TEST_CASE("Device function pointer magic works","[CUDA]"){
  size_t N = 10;
  double * h_arr = new double[10];
  // Important to have it as a pointer to a function pointer
  ConvFunc * d_funcProd = NULL;
  cudaError_t err = cudaMalloc(&d_funcProd, sizeof(ConvFunc) * 1);
  REQUIRE(err == cudaSuccess);
  ConvFunc * d_funcSum = NULL;
  err = cudaMalloc(&d_funcSum, sizeof(ConvFunc) * 1);
  REQUIRE(err == cudaSuccess);
  // Copy device function pointer into a variable on device
  copyFuncPtr<<<1, 1>>>(d_funcProd, 0);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  copyFuncPtr<<<1, 1>>>(d_funcSum, 1);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  // Run the kernel with the function pointer
  double * d_arr = NULL;
  err = cudaMalloc(&d_arr, sizeof(double) * N);
  REQUIRE(err == cudaSuccess);
  SECTION("Product"){
    convolve<<<1, N>>>(d_arr, N, d_funcProd);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    err = cudaMemcpy(h_arr, d_arr, sizeof(double) * N, cudaMemcpyDeviceToHost);
    REQUIRE(err == cudaSuccess);
    for (int i = 0; i < N; i++)
      REQUIRE(h_arr[i] == i * i);
  }
  SECTION("Sum"){
    convolve<<<1, N>>>(d_arr, N, d_funcSum);
    REQUIRE(cudaGetLastError() == cudaSuccess);
    err = cudaMemcpy(h_arr, d_arr, sizeof(double) * N, cudaMemcpyDeviceToHost);
    REQUIRE(err == cudaSuccess);
    for (int i = 0; i < N; i++)
      REQUIRE(h_arr[i] == 2 * i);
  }
  delete [] h_arr;
  cudaFree(d_arr);
  cudaFree(d_funcSum);
  cudaFree(d_funcProd);
}

TEST_CASE( "Catch unit testing works", "[CATCH_UNITTEST]" ) {
    REQUIRE(true);
}

TEST_CASE("Logging system loads properly", "[LOGGING]"){
  // Load configuration from file
  el::Configurations conf("src/logConfigTEST.config");
  // Reconfigure single logger
  el::Loggers::reconfigureLogger("default", conf);
  LOG(INFO) << "=========================\n";
  LOG(INFO) << "Test Log starts\n";
  REQUIRE(true);
}

TEST_CASE("GSL library functions properly", "[GSL]"){
  gsl_sf_result result;
  REQUIRE(gsl_sf_hyperg_1F1_int_e(1,1, 1.0, &result)==0);
  REQUIRE(abs(gsl_sf_hyperg_U(0.2, 0.5, 0.0) - 1.36547) < 0.001);
  REQUIRE(gsl_sf_hyperg_U( 0, 0.5, 0) == 1.0);
  REQUIRE(gsl_sf_hyperg_U( 0, 0.5, 0.1) == 1.0);
}

TEST_CASE("CUDA dynamic query works", "[CUDA]"){
  // Get cuda properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  // Maximum threads per block on this device
  int maxThreads = prop.maxThreadsPerBlock;
  LOG(INFO) << "Maximum threads per block: " << maxThreads << "; Max shared memory per block: " << prop.sharedMemPerBlock << "; Warp size: " << prop.warpSize << endl;
  REQUIRE(maxThreads > 0);
}

TEST_CASE("Kronecker product test", "[MXUTILS:CUDA]"){
  const int nxa = 2;
  const int nya = 2;
  double h_a[nxa * nya] = {1.0, 2.0, 3.0, 4.0};
  const int nxb = 2;
  const int nyb = 2;
  double h_b[nxb * nyb] = {0.0, 5.0, 6.0, 7.0};
  const int resSzx = nxa * nxb;
  const int resSzy = nya * nyb;
  double h_resCorrect[resSzx * resSzy] = {0, 5, 0, 10, 6, 7, 12, 14, 0, 15, 0, 20, 18, 21, 24, 28};
  double * h_res = new double[resSzx * resSzy];
  // Allocating space on the GPU
  double * d_a;
  cudaError_t err = cudaMalloc(&d_a, sizeof(double)*(nxa * nya));
  REQUIRE(err == cudaSuccess);
  double * d_b;
  err = cudaMalloc(&d_b, sizeof(double)*(nxb * nyb));
  REQUIRE(err == cudaSuccess);
  double * d_res;
  err = cudaMalloc(&d_res, sizeof(double)*(resSzx * resSzy));
  REQUIRE(err == cudaSuccess);
  // Copy inputs to device
  err = cudaMemcpy(d_a, h_a, sizeof(double)*(nxa * nya), cudaMemcpyHostToDevice);
  REQUIRE(err == cudaSuccess);
  err = cudaMemcpy(d_b, h_b, sizeof(double)*(nxb * nyb), cudaMemcpyHostToDevice);
  REQUIRE(err == cudaSuccess);
  // Run the kernel
  dim3 threadsPerBlock(resSzx, resSzy);
  kronecker<<<1, threadsPerBlock>>>(d_a, nxa, nya, d_b, nxb, nyb, d_res);
  // Copy the result
  err = cudaMemcpy(h_res, d_res, sizeof(double)*(resSzx * resSzy), cudaMemcpyDeviceToHost);
  REQUIRE(err == cudaSuccess);
  // Check the result
  for (int i = 0; i < resSzx * resSzy; i++)
    REQUIRE(h_resCorrect[i] == h_res[i]);
  // Release memory
  err = cudaFree(d_a);
  REQUIRE(err == cudaSuccess);
  err = cudaFree(d_b);
  REQUIRE(err == cudaSuccess);
  delete [] h_res;
  cudaFree(d_res);
}

TEST_CASE("Simple CUDA kernel run", "[CUDA]"){
  // size of the array
  int n = 130;
  // Host array
  double *h_a;
  // Device array
  double *d_a;
  // allocating space on host and device
  h_a = (double*)malloc(sizeof(double)*n);
  // Allocating space on GPU
  cudaError_t err = cudaMalloc(&d_a, sizeof(double)*n);
  REQUIRE(err == cudaSuccess);
  // Creating blocks and grid ints
  int nThreads = 64;
  int gridSizeX = (int)ceil((float)n/nThreads);
  // Run kernel
  findID<<<gridSizeX, nThreads>>>(d_a, n);
  // Copy the result
  err = cudaMemcpy(h_a, d_a, sizeof(double)*n, cudaMemcpyDeviceToHost);
  REQUIRE(err == cudaSuccess);
  for (int i = 0; i < n; ++i){
      REQUIRE(h_a[i] == i);
  }
  // Release memory
  err = cudaFree(d_a);
  REQUIRE(err == cudaSuccess);
  free(h_a);
}


__global__ void reduceStep(long int * d_in, long int N, long int * d_out){
    extern __shared__ long int shArr[];
    int tID = blockIdx.x * blockDim.x + threadIdx.x;
    int tbid = threadIdx.x;
    shArr[tbid] = (tID < N ? d_in[tID] : 0);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tbid < s)
            shArr[tbid] += shArr[tbid + s];
        __syncthreads();
    }
    if (tbid == 0)
        d_out[blockIdx.x] = shArr[tbid];
}
