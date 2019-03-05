#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <typeinfo>
#include "catch.hpp"
#include "../easyloggingcpp/easylogging++.h"
#include "../twoParticlesInHO_Ham.h"
#include "../utils.h"
const double pi = M_PI;

TEST_CASE("Device function pointer copy works", "[HAM]"){
  resetCudaError();
  C2ParticlesHO ham(64, 0.0);
  CHamiltonian * vH = &ham;
  REQUIRE(vH->timeDepPotential() != NULL);
  REQUIRE(vH->timeDepInteraction() != NULL);
  REQUIRE(vH->timeDepNonLin() != NULL);
  REQUIRE(vH->getParams() != NULL);
}

TEST_CASE("CHamiltonian virtualization works", "[HAM]"){
  resetCudaError();
  C2ParticlesHO ham(64, 0.0);
  CHamiltonian * vH = &ham;
  REQUIRE(vH->getHostX() != NULL);
  REQUIRE(vH->getDeviceX() != NULL);
  REQUIRE(vH->getHostMom() != NULL);
  REQUIRE(vH->getDeviceMom() != NULL);
  REQUIRE(vH->getCoordStep() > 0.0);
  REQUIRE(vH->getMomStep() > 0.0);
}

TEST_CASE("Coordinate and momentum axes definition works","[HAM]"){
  resetCudaError();
  const size_t N = 8;
  double xmax = 4.0;
  double * h_x = NULL;
  double * h_px = NULL;
  double dx = 0.0;
  double dpx = 0.0;
  REQUIRE(C2ParticlesHO::fftdef(N, xmax, h_x, h_px, dx, dpx) == 0);
  REQUIRE(dx == 1.0);
  REQUIRE(dpx == pi / xmax);
  // Test values
  const double testx[N] = {1 - xmax, 2 - xmax, 3 - xmax,4 - xmax,5 - xmax, 6 - xmax, 7 - xmax, 8 - xmax};
  double pxmax = pi * N / (2.0 * xmax);
  const double testPx[N] = {4 * dpx - pxmax, 5 * dpx - pxmax, 6 * dpx - pxmax, 7 * dpx - pxmax, 8 * dpx - pxmax, dpx - pxmax, 2 * dpx - pxmax, 3 * dpx - pxmax};
  for (int i = 0; i < N; i++)
  {
    REQUIRE(h_x[i] == testx[i]);
    REQUIRE(h_px[i] == testPx[i]);
  }
  delete [] h_x;
  delete [] h_px;
  // BAD cases
  REQUIRE(C2ParticlesHO::fftdef(0, xmax, h_x, h_px, dx, dpx) == -1);
  REQUIRE(C2ParticlesHO::fftdef(N, 0, h_x, h_px, dx, dpx) == -1);
}

__global__ void makeVector(double * d_out, double * d_x, size_t sz, hamFunc * d_f, void * params){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= sz)
    return;
  //printf("%.5f ,", (*d_f)(0, 0.0, params));
  d_out[idx] = (*d_f)(d_x[idx], 0.0, params);
}

TEST_CASE("Calling of a device function through the pointer works", "[HAM]"){
  REQUIRE(resetCudaError() == 0);
  size_t N = 64;
  C2ParticlesHO ham(N, 1.0);
  CHamiltonian * vH = &ham;
  // allocate output array
  double * d_arr = NULL;
  cudaError_t err = cudaMalloc(&d_arr, sizeof(double) * N);
  REQUIRE(err == cudaSuccess);
  double * h_V = new double[N];
  // Get cuda properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  // Maximum threads per block on this device
  int maxThreads = prop.maxThreadsPerBlock;
  int blockSz = min((size_t)maxThreads, N);
  int gridSz = ceil(N / blockSz);
  //std::cout << gridSz << " " << blockSz << std::endl;
  double eps = pow(10, -5);
  SECTION("External potential"){
    // Call the kernel to generate x^2
    makeVector<<<gridSz, blockSz>>>(d_arr, vH->getDeviceX(), N, vH->timeDepPotential(), vH->getParams());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(copyArrFromGPU<double>(h_V, d_arr, N) == 0);
    REQUIRE(h_V[0] == Approx(23.4619140625).epsilon(eps));
    REQUIRE(h_V[1] == Approx(21.97265625).epsilon(eps));
    REQUIRE(h_V[N-1] == Approx(25).epsilon(eps));
  }
  SECTION("Interaction"){
    // Call the kernel to generate g*delta(x)
    makeVector<<<gridSz, blockSz>>>(d_arr, vH->getDeviceX(), N, vH->timeDepInteraction(), vH->getParams());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(copyArrFromGPU<double>(h_V, d_arr, N) == 0);
    REQUIRE(h_V[0] == 0.0);
    REQUIRE(h_V[N/2 - 1] == Approx(1.0 / ham.getCoordStep()).epsilon(eps));
    REQUIRE(h_V[N-1] == 0.0);
    REQUIRE(h_V[N-4] == 0.0);
  }
  SECTION("Non-linear"){
    // Call the kernel to get non-linear (which is zero)
    makeVector<<<gridSz, blockSz>>>(d_arr, vH->getDeviceX(), N, vH->timeDepNonLin(), vH->getParams());
    REQUIRE(cudaGetLastError() == cudaSuccess);
    REQUIRE(copyArrFromGPU<double>(h_V, d_arr, N) == 0);
    REQUIRE(h_V[0] == 0.0);
    REQUIRE(h_V[N/2 - 1] == 0);
    REQUIRE(h_V[N-1] == 0);
  }
  cudaFree(d_arr);
  delete [] h_V;
}
