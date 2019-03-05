#include <gsl/gsl_sf_hyperg.h>
#include <complex.h>
#include <fftw3.h>
#include <iostream>
#include <cmath>
#include <fstream>
#include <stdio.h>
#include <cufft.h>
#include <algorithm>
#include <vector>
#include <unordered_map>
#include "catch.hpp"
#include "../easyloggingcpp/easylogging++.h"
#include "../mxutils.h"
#include "../wavefunction.h"
#include "../utils.h"

using namespace std;

TEST_CASE("Matrix save and load work correctly", "[MXUTILS]"){
  resetCudaError();
  size_t szX = 0;
  size_t szY = 0;
  SECTION("Single file ") {
    std::string fn("test.csv");
    double mx[6] = {0,1,2, 4,5, 6};
    saveMatrixToCSV(fn, mx, 3, 2);
    ifstream infile(fn);
    REQUIRE(infile.good());
    size_t szX = 0;
    size_t szY = 0;
    double * mx2 = parseMxFromCSV(fn, szX, szY);
    REQUIRE(szX == 3);
    REQUIRE(szY == 2);
    for (int i = 0; i < szX * szY; i++)
    {
      REQUIRE(mx2[i] == mx[i]);
    }
    remove(fn.c_str());
    delete [] mx2;
  }
  SECTION("Complex into two files ") {
    std::string fnRe("testRe.csv");
    std::string fnIm("testIm.csv");
    int N = 6;
    fftw_complex mx[N];
    for (int i = 0; i < N ; i++)
    {
      mx[i][0] = i;
      mx[i][1] = i+1;
    }
    saveMatrixToCSV(fnRe, fnIm, mx, 3, 2);
    ifstream infileRe(fnRe);
    ifstream infileIm(fnIm);
    REQUIRE(infileRe.good());
    REQUIRE(infileIm.good());
    double * mx2Re = parseMxFromCSV(fnRe, szX, szY);
    double * mx2Im = parseMxFromCSV(fnIm, szX, szY);
    REQUIRE(szX == 3);
    REQUIRE(szY == 2);
    for (int i = 0; i < szX * szY; i++)
    {
      REQUIRE(mx2Re[i] == mx[i][0]);
      REQUIRE(mx2Im[i] == mx[i][1]);
    }
    remove(fnRe.c_str());
    remove(fnIm.c_str());
    delete [] mx2Re;
    delete [] mx2Im;
  }
  // Empty file
  SECTION("parse Empty file") {
    std::string badFile("abdjkgkjsbgs.cdv");
    double * badMx = parseMxFromCSV(badFile, szX, szY);
    REQUIRE(badMx == NULL);
  }
}

TEST_CASE("Convertion of array the Busch state wave function to cfftDoubleComplex 2D matrix works (Imaginary part = NULL, Real part != NULL)","[MXUTILS:CUDA]"){
  resetCudaError();
  int N = 64;
  double x0 = 0.0;
  double omega = 1.0;
  double xmax = 5.0;
  double E = 1.1;
  // Generate the GS of the two interacting bosons in the trap
  double * h_wf = calcInitialPsi(N, E, x0, xmax, omega);
  REQUIRE(h_wf != NULL);
  REQUIRE(h_wf[N * N / 2 + N / 2] != 0.0);
  fftw_complex * h_complexWF = new fftw_complex[N*N];
  for (size_t i = 0; i < N*N; i++)
  {
    h_complexWF[i][0] = h_wf[i];
    h_complexWF[i][1] = 0.0;
  }
  // Allocate memory for the complex array on GPU
  size_t pitch = 0;
  cufftDoubleComplex * d_wfComplex = NULL;
  cudaError_t cudaStatus = cudaMallocPitch(&d_wfComplex, &pitch, sizeof(cufftDoubleComplex) * N, N);
  REQUIRE(cudaStatus == cudaSuccess);
  REQUIRE(copyToCudaComplex(d_wfComplex, pitch, h_complexWF, N, N) == 0);
  // Check if the copy is correct
  cufftDoubleComplex * h_wfComplex = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * pitch * N);
  REQUIRE(h_wfComplex != NULL);
  cudaStatus = cudaMemcpy(h_wfComplex, d_wfComplex, pitch * N, cudaMemcpyDeviceToHost);
  REQUIRE(cudaStatus == cudaSuccess);
  for (int i = 0 ; i < N; i++)
    for (int j = 0; j < N; j++)
    {
      cufftDoubleComplex * curRow = (cufftDoubleComplex *) &(((char *)h_wfComplex)[i * pitch]);
      double real = ((double *)&curRow[j])[0];
      double imag = ((double *)&curRow[j])[1];
      REQUIRE(imag == 0.0);
      REQUIRE(real == h_complexWF[i * N + j][0]);
    }
  free (h_wfComplex);
  delete [] h_complexWF;
  delete [] h_wf;
  cudaStatus = cudaFree(d_wfComplex);
  REQUIRE(cudaStatus == cudaSuccess);
}

TEST_CASE("Convertion of bad data to cudaComplex returns correct values","[MXUTILS:CUDA]"){
  resetCudaError();
  int N = 64;
  fftw_complex * h_complexWF = new fftw_complex[N*N];
  // Allocate memory for the complex array on GPU
  size_t pitch = 0;
  cufftDoubleComplex * d_wfComplex = NULL;
  cudaError_t cudaStatus = cudaMallocPitch(&d_wfComplex, &pitch, sizeof(cufftDoubleComplex) * N, N);
  REQUIRE(cudaStatus == cudaSuccess);
  REQUIRE(copyToCudaComplex(NULL, pitch, NULL, N, N) == -1);
  REQUIRE(copyToCudaComplex(d_wfComplex, N - 1, h_complexWF, N, N) == -1);
  REQUIRE(copyToCudaComplex(d_wfComplex, pitch, h_complexWF, 0, N) == -1);
  REQUIRE(copyToCudaComplex(d_wfComplex, pitch, h_complexWF, N, 0) == -1);
  cudaStatus = cudaFree(d_wfComplex);
  REQUIRE(cudaStatus == cudaSuccess);
  delete [] h_complexWF;
}

TEST_CASE("Convertion of array the imaginary Busch state wave function to cfftDoubleComplex 2D matrix works (Imaginary part != NULL, Real part == NULL)","[MXUTILS:CUDA]"){
  resetCudaError();
  int N = 64;
  double x0 = 0.0;
  double omega = 1.0;
  double xmax = 5.0;
  double E = 1.7;
  // Generate the GS of the two interacting bosons in the trap
  double * h_wf = calcInitialPsi(N, E, x0, xmax, omega);
  REQUIRE(h_wf != NULL);
  fftw_complex * h_complexWF = new fftw_complex[N*N];
  for (size_t i = 0; i < N*N; i++)
  {
    h_complexWF[i][1] = h_wf[i];
    h_complexWF[i][0] = 0.0;
  }
  // Allocate memory for the complex array on GPU
  size_t pitch = 0;
  cufftDoubleComplex * d_wfComplex = NULL;
  cudaError_t cudaStatus = cudaMallocPitch(&d_wfComplex, &pitch, sizeof(cufftDoubleComplex) * N, N);
  REQUIRE(cudaStatus == cudaSuccess);
  REQUIRE(copyToCudaComplex(d_wfComplex, pitch, h_complexWF, N, N) == 0);
  // Check if the copy is correct
  cufftDoubleComplex * h_wfComplex = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * pitch * N);
  REQUIRE(h_wfComplex != NULL);
  cudaStatus = cudaMemcpy(h_wfComplex, d_wfComplex, pitch * N, cudaMemcpyDeviceToHost);
  REQUIRE(cudaStatus == cudaSuccess);
  for (int i = 0 ; i < N; i++)
    for (int j = 0; j < N; j++)
    {
      cufftDoubleComplex * curRow = (cufftDoubleComplex *) &(((char *)h_wfComplex)[i * pitch]);
      double real = curRow[j].x;
      double imag = curRow[j].y;
      REQUIRE(imag == h_complexWF[i * N + j][1]);
      REQUIRE(real == 0.0);
    }
  free (h_wfComplex);
  cudaStatus = cudaFree(d_wfComplex);
  REQUIRE(cudaStatus == cudaSuccess);
  delete [] h_wf;
  delete [] h_complexWF;
}

TEST_CASE("Convertion of array a full complex state wave function to cfftDoubleComplex 2D matrix works (Imaginary part != NULL, Real part != NULL)","[MXUTILS:CUDA]"){
  resetCudaError();
  int N = 64;
  double x0 = 0.0;
  double omega = 1.0;
  double xmax = 5.0;
  double E = 1.7;
  // Generate the GS of the two interacting bosons in the trap
  double * h_wf = calcInitialPsi(N, E, x0, xmax, omega);
  REQUIRE(h_wf != NULL);
  fftw_complex * h_complexWF = new fftw_complex[N*N];
  for (size_t i = 0; i < N*N; i++)
  {
    h_complexWF[i][0] = h_wf[i];
    h_complexWF[i][1] = 1.0;
  }
  // Allocate memory for the complex array on GPU
  size_t pitch = 0;
  cufftDoubleComplex * d_wfComplex = NULL;
  cudaError_t cudaStatus = cudaMallocPitch(&d_wfComplex, &pitch, sizeof(cufftDoubleComplex) * N, N);
  REQUIRE(cudaStatus == cudaSuccess);
  REQUIRE(copyToCudaComplex(d_wfComplex, pitch, h_complexWF, N, N) == 0);
  // Check if the copy is correct
  cufftDoubleComplex * h_wfComplex = (cufftDoubleComplex *) malloc(sizeof(cufftDoubleComplex) * pitch * N);
  REQUIRE(h_wfComplex != NULL);
  cudaStatus = cudaMemcpy(h_wfComplex, d_wfComplex, pitch * N, cudaMemcpyDeviceToHost);
  REQUIRE(cudaStatus == cudaSuccess);
  for (int i = 0 ; i < N; i++)
    for (int j = 0; j < N; j++)
    {
      cufftDoubleComplex * curRow = (cufftDoubleComplex *) &(((char *)h_wfComplex)[i * pitch]);
      double real = ((double *)&curRow[j])[0];
      double imag = ((double *)&curRow[j])[1];
      REQUIRE(imag == h_complexWF[i * N + j][1]);
      REQUIRE(real == h_complexWF[i * N + j][0]);
    }
  free (h_wfComplex);
  cudaStatus = cudaFree(d_wfComplex);
  REQUIRE(cudaStatus == cudaSuccess);
  delete [] h_wf;
  delete [] h_complexWF;
}


TEST_CASE("cufftDoubleComplex Array copy from GPU to host works", "[cufftDoubleComplex][MXUTILS]"){
  resetCudaError();
  cufftDoubleComplex * h_arr = new cufftDoubleComplex[10];
  cufftDoubleComplex * d_arr = NULL;
  cudaError_t err = cudaMalloc(&d_arr, sizeof(cufftDoubleComplex) * 10);
  REQUIRE(err == cudaSuccess);
  SECTION("Good case"){
    REQUIRE(copyArrFromGPU<cufftDoubleComplex>(h_arr, d_arr, 10) == 0);
  }
  SECTION("Bad cases"){
    REQUIRE(copyArrFromGPU<cufftDoubleComplex>(NULL, d_arr, 10) == -1);
    REQUIRE(copyArrFromGPU<cufftDoubleComplex>(h_arr, NULL, 10) == -1);
    REQUIRE(copyArrFromGPU<cufftDoubleComplex>(h_arr, d_arr, 0) == -1);
  }
  delete [] h_arr;
  cudaFree(d_arr);
}

TEST_CASE("Double Array copy from GPU to host works", "[double][MXUTILS]"){
  resetCudaError();
  double * h_arr = new double[10];
  double * d_arr = NULL;
  cudaError_t err = cudaMalloc(&d_arr, sizeof(double) * 10);
  REQUIRE(err == cudaSuccess);
  SECTION("Good case"){
    REQUIRE(copyArrFromGPU<double>(h_arr, d_arr, 10) == 0);
  }
  SECTION("Bad cases"){
    REQUIRE(copyArrFromGPU<double>(NULL, d_arr, 10) == -1);
    REQUIRE(copyArrFromGPU<double>(h_arr, NULL, 10) == -1);
    REQUIRE(copyArrFromGPU<double>(h_arr, d_arr, 0) == -1);
  }
  delete [] h_arr;
  cudaFree(d_arr);
}

TEST_CASE( "allocate and copy to GPU template works", "[cufftDoubleComplex][MXUTILS]") {
  resetCudaError();
  cufftDoubleComplex * h_arr = new cufftDoubleComplex[10];
  SECTION("Good case"){
    cufftDoubleComplex * h_tmparr = new cufftDoubleComplex[10];
    cufftDoubleComplex * d_arr = allocateAndCopyToGPU<cufftDoubleComplex>(h_arr, 10);
    REQUIRE(d_arr != NULL);
    cudaError_t err = cudaMemcpy(h_tmparr, d_arr, 10 * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost);
    REQUIRE(err == cudaSuccess);
    for (int i = 0; i < 10; i++)
    {
      REQUIRE(((double *)(&h_tmparr[i]))[0] == ((double *)(&h_arr[i]))[0]);
      REQUIRE(((double *)(&h_tmparr[i]))[1] == ((double *)(&h_arr[i]))[1]);
    }
    delete [] h_tmparr;
    cudaFree(d_arr);
  }
  SECTION("Bad cases"){
    cufftDoubleComplex * d_arr = allocateAndCopyToGPU<cufftDoubleComplex>(h_arr, 0);
    REQUIRE(d_arr == NULL);
    d_arr = allocateAndCopyToGPU<cufftDoubleComplex>(NULL, 10);
    REQUIRE(d_arr == NULL);
  }
  delete [] h_arr;
}

TEST_CASE("exp(-0.5 i * dt * V) generation works", "[MXUTILS]"){
  resetCudaError();
  int N = 64;
  double x0 = 0.0;
  double omega = 1.0;
  double xmax = 5.0;
  double g = 4.4467;
  double dt = 0.001;
  double dx = 2 * xmax / (double)N;
  double h_V[N];
  double eps = pow(10, -5);
  // Harmonic pot. centered on x0, x in [-xmax, xmax]
  for (int i = 0; i < N; i++)
  {
    double x = (i + 1) * dx - xmax;
    h_V[i] = omega * (x - x0) * (x - x0);
  }
  REQUIRE(h_V[0] == Approx(23.4619140625).epsilon(eps));
  REQUIRE(h_V[1] == Approx(21.97265625).epsilon(eps));
  REQUIRE(h_V[N-1] == Approx(25).epsilon(eps));
  double * d_V = allocateAndCopyToGPU<double>(h_V, N);
  REQUIRE(d_V != NULL);
  SECTION("Good case") {
    // generate expV
    cufftDoubleComplex * d_expV = makeExpV(dt, d_V, N, g, dx);
    REQUIRE(d_expV != NULL);
    // Read reference
    size_t szX = 0;
    size_t szY = 0;
    double * refReal = parseMxFromCSV("src/unittest/expVReal.csv", szX, szY);
    //cout << szX << " " << szY << endl;
    double * refImag = parseMxFromCSV("src/unittest/expVImag.csv", szX, szY);
    //cout << szX << " " << szY << endl;
    REQUIRE(refReal != NULL);
    REQUIRE(refImag != NULL);
    // Copy expv from GPU
    cufftDoubleComplex * h_expV = new cufftDoubleComplex[szX * szY];
    REQUIRE(copyArrFromGPU(h_expV, d_expV, szX * szY) == 0);
    // Compare with the reference
    for (int i = 0; i < szX * szY; i++)
    {
      double * curVal = (double *)&h_expV[i];
      REQUIRE(curVal[0] == Approx(refReal[i]).margin(eps));
      REQUIRE(curVal[1] == Approx(refImag[i]).margin(eps));
    }
    delete [] refReal;
    delete [] refImag;
    delete [] h_expV;
    cudaFree(d_expV);
  }
  SECTION("Bad case"){
    cufftDoubleComplex * d_expV = makeExpV(dt, NULL, N, g, dx);
    REQUIRE(d_expV == NULL);
    d_expV = makeExpV(dt, d_V, 0, g, dx);
    REQUIRE(d_expV == NULL);
  }
  cudaFree(d_V);
}

TEST_CASE("FFT transform works", "[MXUTILS:CUFFT]"){
  LOG(INFO) << "========Test mxutils doFFT======";
  resetCudaError();
  std::string realFile("src/unittest/expVReal.csv");
  std::string imagFile("src/unittest/expVImag.csv");
  CWaveFunction wf(realFile, imagFile);
  REQUIRE(wf.getColsSize() > 0);
  REQUIRE(wf.getRowSize() > 0);
  REQUIRE(wf.getHostWF() != NULL);
  LOG(INFO) << wf.getHostWF()[0][0] << " " << wf.getHostWF()[10][0];
  REQUIRE(wf.getHostWF()[0][0] != 0.0);
  REQUIRE(wf.getHostWF()[0][1] != 0.0);
  REQUIRE(wf.copyToGPU() == 0);
  REQUIRE(wf.getHostWF()[0][0] != 0.0);
  REQUIRE(wf.getHostWF()[0][1] != 0.0);
  cufftHandle * plan = NULL;
  SECTION("Compare with matlab reference"){
    // Run FFT
    REQUIRE(doFFT(wf.getDeviceWF(), wf.getColsSize(), wf.getRowSize(), plan, true) == 0);
    REQUIRE(plan != NULL);
    // Copy result back
    REQUIRE(wf.copyFromGPU() == 0);
    REQUIRE(wf.getHostWF()[0][0] != 0.0);
    REQUIRE(wf.getHostWF()[0][1] != 0.0);
    // Load reference
    CWaveFunction referenceWF("src/unittest/expVFFTReal.csv","src/unittest/expVFFTImag.csv");
    REQUIRE(referenceWF.getHostWF() != NULL);
    // Compare with the reference
    REQUIRE(wf.getColsSize() == referenceWF.getColsSize());
    REQUIRE(wf.getRowSize() == referenceWF.getRowSize());
    double eps = pow(10, -3);
    for (int i = 0; i < wf.getColsSize() * wf.getRowSize(); i++){
      REQUIRE(wf.getHostWF()[i][0] == Approx(referenceWF.getHostWF()[i][0]).margin(eps));
      REQUIRE(wf.getHostWF()[i][1] == Approx(referenceWF.getHostWF()[i][1]).margin(eps));
    }
    cufftDestroy(*plan);
  }
  SECTION("IFFT(FFT) == I"){
    // Run FFT
    REQUIRE(doFFT(wf.getDeviceWF(), wf.getColsSize(), wf.getRowSize(), plan, true) == 0);
    REQUIRE(plan != NULL);
    // Run ifft
    REQUIRE(doFFT(wf.getDeviceWF(), wf.getColsSize(), wf.getRowSize(), plan, false) == 0);
    // Copy back to CPU
    REQUIRE(wf.copyFromGPU() == 0);
    // Load reference - the initial wf
    CWaveFunction referenceWF(realFile,imagFile);
    REQUIRE(referenceWF.getHostWF() != NULL);
    // Compare with the reference
    REQUIRE(wf.getColsSize() == referenceWF.getColsSize());
    REQUIRE(wf.getRowSize() == referenceWF.getRowSize());
    double eps = pow(10, -5);
    for (int i = 0; i < wf.getColsSize() * wf.getRowSize(); i++){
      REQUIRE(wf.getHostWF()[i][0]/(double)(wf.getColsSize() * wf.getRowSize()) == Approx(referenceWF.getHostWF()[i][0]).margin(eps));
      REQUIRE(wf.getHostWF()[i][1]/(double)(wf.getColsSize() * wf.getRowSize()) == Approx(referenceWF.getHostWF()[i][1]).margin(eps));
    }
    cufftDestroy(*plan);
  }
  SECTION("Bad cases"){
    REQUIRE(doFFT(NULL, wf.getColsSize(), wf.getRowSize(), plan, true) != 0);
    REQUIRE(plan == NULL);
    REQUIRE(doFFT(wf.getDeviceWF(), 0, wf.getRowSize(), plan, true) != 0);
    REQUIRE(plan == NULL);
    REQUIRE(doFFT(wf.getDeviceWF(), wf.getColsSize(), 0, plan, true) != 0);
    REQUIRE(plan == NULL);
    REQUIRE(doFFT(NULL, wf.getColsSize(), wf.getRowSize(), plan, false) != 0);
    REQUIRE(plan == NULL);
    REQUIRE(doFFT(wf.getDeviceWF(), 0, wf.getRowSize(), plan, false) != 0);
    REQUIRE(plan == NULL);
    REQUIRE(doFFT(wf.getDeviceWF(), wf.getColsSize(), 0, plan, false) != 0);
    REQUIRE(plan == NULL);
  }
}

TEST_CASE("Getting the norm works", "[MXUTILS]"){
  resetCudaError();
  std::string realFile("src/unittest/expVReal.csv");
  std::string imagFile("src/unittest/expVImag.csv");
  CWaveFunction wf(realFile, imagFile);
  REQUIRE(wf.getColsSize() > 0);
  REQUIRE(wf.getRowSize() > 0);
  REQUIRE(wf.getHostWF() != NULL);
  REQUIRE(wf.copyToGPU() == 0);
  size_t sz = wf.getColsSize();
  double norm = 0.0;
  SECTION("Good case"){
    double eps = pow(10, -5);
    REQUIRE(getNorm(wf.getDeviceWF(), sz, norm) == 0);
    REQUIRE(norm == Approx(64).margin(eps));
  }
  SECTION("Bad cases"){
    REQUIRE(getNorm(NULL, sz, norm) == -1);
    REQUIRE(getNorm(wf.getDeviceWF(), 0, norm) == -1);
  }
}

TEST_CASE("Renormalization works","[MXUTILS]"){
  resetCudaError();
  std::string realFile("src/unittest/NotNormPsiRe.csv");
  std::string imagFile("src/unittest/NotNormPsiIm.csv");
  CWaveFunction wf(realFile, imagFile);
  REQUIRE(wf.getColsSize() > 0);
  REQUIRE(wf.getRowSize() > 0);
  REQUIRE(wf.getHostWF() != NULL);
  REQUIRE(wf.copyToGPU() == 0);
  size_t sz = wf.getColsSize();
  double dx = 1/(double)sz;
  SECTION("Good case"){
    double eps = pow(10, -5);
    REQUIRE(normalize(wf.getDeviceWF(), sz, dx) == 0);
    double norm = 0.0;
    REQUIRE(getNorm(wf.getDeviceWF(), sz, norm) == 0);
    REQUIRE(norm * dx == Approx(1.0).margin(eps));
  }
  SECTION("Bad cases"){
    REQUIRE(normalize(wf.getDeviceWF(), 0, dx) == -1);
    REQUIRE(normalize(NULL, sz, dx) == -1);
    REQUIRE(normalize(wf.getDeviceWF(), sz, 0) == -1);
  }
}

TEST_CASE("Uniform random pick with replacement works", "[MXUTILS:CS]"){
  size_t N = 2048;
  size_t * h_dst = new size_t[N];
  memset(h_dst, 255, sizeof(size_t) * N);
  size_t * d_dst = NULL;
  unordered_map<size_t, bool> histo;
  cudaError_t err = cudaMalloc(&d_dst, sizeof(size_t) * N);
  REQUIRE(err == cudaSuccess);
  SECTION("Good case"){
    // Query device information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreads = prop.maxThreadsPerBlock;
    REQUIRE(generateRandomPicks(d_dst, N, maxThreads) == 0);
    REQUIRE(copyArrFromGPU<size_t>(h_dst, d_dst, N) == 0);
    for (int i = 0; i < N; i++)
    {
      REQUIRE(h_dst[i] < N);
      if (histo.count(h_dst[i]) == 0)
        histo[h_dst[i]] = true;
    }
    // Should be more than one number in there
    REQUIRE(histo.size() > 1);
  }
  SECTION("Bad cases"){
      REQUIRE(generateRandomPicks(NULL, N, 1024) == -1);
      REQUIRE(generateRandomPicks(d_dst, 0, 1024) == -1);
      REQUIRE(generateRandomPicks(d_dst, N, 0) == -1);
  }
  cudaFree(d_dst);
  delete [] h_dst;
}

TEST_CASE("Filling up the array with the range works","[MXUTILS]"){
  size_t N = 4096;
  size_t * d_arr = NULL;
  cudaError_t err = cudaMalloc(&d_arr, sizeof(size_t) * N);
  REQUIRE(err == cudaSuccess);
  size_t * h_arr = new size_t[N];
  // Query device information
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int maxThreads = prop.maxThreadsPerBlock;
  SECTION("Good case"){
    REQUIRE(fillRange(d_arr, N, maxThreads) == 0);
    REQUIRE(copyArrFromGPU(h_arr, d_arr, N) == 0);
    for (int i = 0 ; i < N ; i++)
      REQUIRE(h_arr[i] == i);
  }
  SECTION("Bad cases"){
    REQUIRE(fillRange(NULL, N, maxThreads) == -1);
    REQUIRE(fillRange(d_arr, 0, maxThreads) == -1);
  }
  delete [] h_arr;
  cudaFree(d_arr);
}

TEST_CASE("Generation of random permutations of an array works", "[MXUTILS:CS]"){
  size_t N = 4096;
  size_t * d_arr = NULL;
  cudaError_t err = cudaMalloc(&d_arr, sizeof(size_t) * N);
  REQUIRE(err == cudaSuccess);
  size_t * h_arr = new size_t[N];
  unsigned short * h_cntArr = new unsigned short[N];
  memset(h_cntArr, 0, sizeof(unsigned short) * N);
  bool permuted = false;
  // Query device information
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int maxThreads = prop.maxThreadsPerBlock;
  REQUIRE(fillRange(d_arr, N, maxThreads) == 0);
  SECTION("Good case"){
    REQUIRE(permuteElements(d_arr, N, maxThreads) == 0);
    REQUIRE(copyArrFromGPU(h_arr, d_arr, N) == 0);
    for (int i = 0; i < N; i++)
    {
      // Has a swap occured?
      permuted = permuted || (h_arr[i] != i);
      // Count an element as present
      REQUIRE(h_arr[i] < N);
      h_cntArr[h_arr[i]]++;
    }
    // At least one swap has occured
    REQUIRE(permuted);
    for (int i = 0; i < N; i++)
    {
      // All elements must be present only once
      REQUIRE(h_cntArr[i]  == 1);
    }
  }
  SECTION("Bad cases"){
    REQUIRE(permuteElements(NULL, N, maxThreads) == -1);
    REQUIRE(permuteElements(d_arr, 0, maxThreads) == -1);
  }
  cudaFree(d_arr);
  delete [] h_arr;
  delete [] h_cntArr;
}

TEST_CASE("HOST: Generation of random permutations of an array works", "[MXUTILS:CS]"){
  size_t N = 4096;
  size_t * h_arr = new size_t[N];
  unsigned short * h_cntArr = new unsigned short[N];
  memset(h_cntArr, 0, sizeof(unsigned short) * N);
  bool permuted = false;
  for (size_t i = 0; i < N; i++)
  {
    h_arr[i] = i;
  }
  SECTION("Good case"){
    REQUIRE(permuteElements_host(h_arr, N) == 0);
    for (int i = 0; i < N; i++)
    {
      // Has a swap occured?
      permuted = permuted || (h_arr[i] != i);
      // Count an element as present
      REQUIRE(h_arr[i] < N);
      h_cntArr[h_arr[i]]++;
    }
    // At least one swap has occured
    REQUIRE(permuted);
    for (int i = 0; i < N; i++)
    {
      // All elements must be present only once
      REQUIRE(h_cntArr[i]  == 1);
    }
  }
  SECTION("Bad cases"){
    REQUIRE(permuteElements_host(NULL, N) == -1);
    REQUIRE(permuteElements_host(h_arr, 0) == -1);
  }
  delete [] h_arr;
  delete [] h_cntArr;
}
