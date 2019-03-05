#include "catch.hpp"
#include <cufft.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include "../easyloggingcpp/easylogging++.h"
#include "../cs_utils.h"
#include "../mxutils.h"

using namespace std::chrono;

TEST_CASE("Test complex to abs value conversion of big data piecewise", "[CSUTILS]"){
  size_t maxMem = sizeof(double) * 1024 + 6;
  size_t N = 123456;
  // Fill in the data
  fftw_complex * h_complexWF = new fftw_complex[N];
  for (size_t i = 0; i < N; i++)
  {
    h_complexWF[i][0] = i*3;
    h_complexWF[i][1] = i*4;
  }
  SECTION("All cpu version"){
    // time the all-cpu version
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    double * h_abs = toAbs_host(h_complexWF, N, 0);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    REQUIRE(h_abs != NULL);
    LOG(INFO) << "All cpu toAbs version takes " << time_span.count() << "ms\n";
    for (size_t i = 0; i < N; i++)
    {
      REQUIRE(h_abs[i] == (double)(25*i*i));
    }
    delete [] h_abs;
  }
  SECTION("piecewise gpu version"){
    // time the gpu version
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    double * h_abs = toAbs_host(h_complexWF, N, maxMem);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    REQUIRE(h_abs != NULL);
    LOG(INFO) << "piecewise GPU toAbs version takes " << time_span.count() << "ms\n";
    for (size_t i = 0; i < N; i++)
    {
      REQUIRE(h_abs[i] == (double)(25*i*i));
    }
    delete [] h_abs;
  }
  SECTION("Bad cases"){
    // piecewise gpu
    REQUIRE(toAbs_host(NULL, N, maxMem) == NULL);
    REQUIRE(toAbs_host(h_complexWF, 0, maxMem) == NULL);
    // all cpu
    REQUIRE(toAbs_host(NULL, N, 0) == NULL);
    REQUIRE(toAbs_host(h_complexWF, 0, 0) == NULL);
  }
  delete [] h_complexWF;
}

TEST_CASE("Test find support with sorting (device version)", "[CSUTILS]"){
  cufftDoubleComplex arr[2048];
  size_t N = 2048;
  for (int i = 0; i < N; i++)
  {
    arr[i].x = i % 3;
    arr[i].y = arr[i].x;
  }
  cufftDoubleComplex * d_data = allocateAndCopyToGPU<cufftDoubleComplex>(arr, N);
  REQUIRE(d_data != NULL);
  char * d_support = NULL;
  cudaError_t err = cudaMalloc(&d_support, sizeof(char) * N);
  REQUIRE(err == cudaSuccess);
  size_t k = 600;
  // Get cuda properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  // Maximum threads per block on this device
  int maxThreads = prop.maxThreadsPerBlock;
  SECTION("Good case"){
    REQUIRE(findSupport_sort_device(d_data, d_support, N, k, maxThreads) == 0);
    REQUIRE(copyArrFromGPU(arr, d_data, N) == 0);
    char * h_support = new char[N];
    REQUIRE(copyArrFromGPU(h_support, d_support, N) == 0);
    size_t cnt = 0;
    for (int i = 0; i < N; i++)
    {
      cnt++;
      if (i % 3 == 2)
        REQUIRE(h_support[i] == 1);
      else REQUIRE(h_support[i] == 2);
    }
    REQUIRE(cnt >= k);
    delete [] h_support;
  }
  SECTION("Bad cases"){
    REQUIRE(findSupport_sort_device(NULL, d_support, N, k, maxThreads) == -1);
    REQUIRE(findSupport_sort_device(d_data, NULL, N, k, maxThreads) == -1);
    REQUIRE(findSupport_sort_device(d_data, d_support, 0, k, maxThreads) == -1);
    REQUIRE(findSupport_sort_device(d_data, d_support, N, 0, maxThreads) == -1);
    REQUIRE(findSupport_sort_device(d_data, d_support, N, k, 0) == -1);
  }
  cudaFree(d_support);
  cudaFree(d_data);
}

TEST_CASE("Test find support with sorting (host version)", "[CSUTILS]"){
  size_t N = 2048;
  fftw_complex * h_complexWF = new fftw_complex[N];
  for (int i = 0; i < N; i++)
  {
    h_complexWF[i][0] = i % 3;
    h_complexWF[i][1] = h_complexWF[i][0];
  }
  size_t k = 600;
  // Get cuda properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  // Maximum threads per block on this device
  int maxThreads = prop.maxThreadsPerBlock;
  SECTION("Good case - cpu only version"){
    char * support = findSupport_sort_host(h_complexWF, N, k, 0, maxThreads);
    REQUIRE(support != NULL);
    size_t cnt = 0;
    for (int i = 0; i < N; i++)
    {
      cnt++;
      if (i % 3 == 2)
        REQUIRE(support[i] == 1);
      else REQUIRE(support[i] == 2);
    }
    REQUIRE(cnt >= k);
    delete [] support;
  }
  SECTION("Good case - piecewise gpu version"){
    char * support = findSupport_sort_host(h_complexWF, N, k, sizeof(double) * 128 + 6, maxThreads);
    REQUIRE(support != NULL);
    size_t cnt = 0;
    for (int i = 0; i < N; i++)
    {
      cnt++;
      if (i % 3 == 2)
        REQUIRE(support[i] == 1);
      else REQUIRE(support[i] == 2);
    }
    REQUIRE(cnt >= k);
    delete [] support;
  }
  SECTION("Bad cases"){
    // piecewise gpu
    REQUIRE(findSupport_sort_host(NULL, N, k, sizeof(double) * 128 + 6, maxThreads) == NULL);
    REQUIRE(findSupport_sort_host(h_complexWF, 0, k, sizeof(double) * 128 + 6, maxThreads) == NULL);
    REQUIRE(findSupport_sort_host(h_complexWF, N, 0, sizeof(double) * 128 + 6, maxThreads) == NULL);
    // CPU only
    REQUIRE(findSupport_sort_host(NULL, N, k, 0, maxThreads) == NULL);
    REQUIRE(findSupport_sort_host(h_complexWF, 0, k, 0, maxThreads) == NULL);
    REQUIRE(findSupport_sort_host(h_complexWF, N, 0, 0, maxThreads) == NULL);
  }
  delete [] h_complexWF;
}

TEST_CASE("Test upsamling", "[CSUTILS]"){
  size_t Nx = 64;
  size_t Ny = 32;
  size_t N1 = 16;
  size_t N2 = 32;
  fftw_complex * h_complexWF = new fftw_complex[N1*N2];
  for (size_t i = 0; i < N1*N2; i++)
  {
    h_complexWF[i][0] = i + 1;
    h_complexWF[i][1] = 3*i + 1;
  }
  size_t * h_rows = new size_t[N1];
  size_t * h_cols = new size_t[N2];
  for (size_t i = 0; i < N1; i++)
  {
    h_rows[i] = i * Ny / N1;
  }
  for (size_t i = 0; i < N2; i++)
  {
    h_cols[i] = i * Nx / N2;
  }
  fftw_complex * h_out = new fftw_complex[Nx*Ny];
  memset(h_out, 0, sizeof(double) * 2 * Nx * Ny);
  SECTION("Good case"){
    REQUIRE(restore_host(h_complexWF, h_rows, N1, h_cols, N2, h_out, Ny, Nx) == 0);
    for (size_t i = 0; i < Ny; i++){
      for (size_t j = 0; j < Nx; j++)
      {
        //std::cout << h_outR[i * Nx + j] << " ";
        if (i % (Ny / N1) == 0 && j % (Nx / N2) == 0)
        {
          REQUIRE(h_out[i * Nx + j][0] == i / (Ny / N1) * N2 + j / (Nx / N2) + 1);
          REQUIRE(h_out[i * Nx + j][1] == 3 * (i / (Ny / N1) * N2 + j / (Nx / N2)) + 1);
        }
        else
        {
          REQUIRE(h_out[i * Nx + j][0] == 0.0);
          REQUIRE(h_out[i * Nx + j][1] == 0.0);
        }
      }
      //std::cout << std::endl;
    }
  }
  SECTION("Bad cases"){
    REQUIRE(restore_host(NULL, h_rows, N1, h_cols, N2, h_out, Ny, Nx) == -1);
    REQUIRE(restore_host(h_complexWF, NULL, N1, h_cols, N2, h_out, Ny, Nx) == -1);
    REQUIRE(restore_host(h_complexWF, h_rows, 0, h_cols, N2, h_out, Ny, Nx) == -1);
    REQUIRE(restore_host(h_complexWF, h_rows, N1, NULL, N2, h_out, Ny, Nx) == -1);
    REQUIRE(restore_host(h_complexWF, h_rows, N1, h_cols, 0, h_out, Ny, Nx) == -1);
    REQUIRE(restore_host(h_complexWF, h_rows, N1, h_cols, N2, NULL, Ny, Nx) == -1);
    REQUIRE(restore_host(h_complexWF, h_rows, N1, h_cols, N2, h_out, 0, Nx) == -1);
    REQUIRE(restore_host(h_complexWF, h_rows, N1, h_cols, N2, h_out, Ny, 0) == -1);
  }
  // Cleanup
  delete [] h_complexWF;
  delete [] h_rows;
  delete [] h_cols;
  delete [] h_out;
}
