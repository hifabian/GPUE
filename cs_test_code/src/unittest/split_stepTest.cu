#include <chrono>
#include <ctime>
#include <ratio>
#include <cuda_profiler_api.h>
#include "catch.hpp"
#include "../split_step.h"
#include "../twoParticlesInHO_Ham.h"
#include "../wavefunction.h"
#include "../easyloggingcpp/easylogging++.h"
#include "../utils.h"

using namespace std::chrono;

__global__ void g_calcExpVPhi(hamFunc * df_V,  hamFunc * df_U, hamFunc * df_Nonl, void * d_par, double T, double half_i_dt, cufftDoubleComplex * d_expV_phi, double * d_x, size_t vSz);

TEST_CASE("Split step Constructor works","[SPLIT]"){
  resetCudaError();
  size_t Nx = 64;
  // dx = 2.0 * xmax / (double)Nx
  C2ParticlesHO ham(Nx, 0.0);
  CSplitStep splitStepOperator(&ham);
  REQUIRE(splitStepOperator.ham != NULL);
  CSplitStep badOper(NULL);
  REQUIRE(badOper.ham == NULL);
}

TEST_CASE("ExpV.*phi works","[SPLIT]"){
  resetCudaError();
  CWaveFunction wf("src/unittest/psiRe.csv", "src/unittest/psiIm.csv");
  REQUIRE(wf.getColsSize() > 0);
  REQUIRE(wf.getRowSize() > 0);
  REQUIRE(wf.getHostWF() != NULL);
  REQUIRE(wf.copyToGPU() == 0);
  size_t vSize = wf.getColsSize();
  double x0 = 0.0;
  double omega = 0.5;
  double xmax = 5.0;
  double g = 4.4467;
  double dt = 0.001;
  double T = 0;
  double eps = pow(10, -3);
  double half_i_dt = -0.5 * dt;
  // Harmonic pot. centered on x0, x in [-xmax, xmax]
  C2ParticlesHO ham(vSize, g, xmax, x0, omega);
  CSplitStep splitStepOperator(&ham);
  REQUIRE(splitStepOperator.ham != NULL);
  // Get cuda properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  // Maximum threads per block on this device
  int maxThreads = prop.maxThreadsPerBlock;
  // Call the kernel to generate expV
  int blockSzX = min((size_t)maxThreads, vSize);
  dim3 threadsPerBlock(blockSzX, 1);
  dim3 grid(ceil(vSize / blockSzX), vSize);
  g_calcExpVPhi<<<grid, threadsPerBlock>>>(ham.timeDepPotential(),  ham.timeDepInteraction(), ham.timeDepNonLin(), ham.getParams(), T, half_i_dt, wf.getDeviceWF(), ham.getDeviceX(), vSize);
  REQUIRE(cudaGetLastError() == cudaSuccess);
  // normalize
  REQUIRE(normalize(wf.getDeviceWF(), vSize, ham.getCoordStep()) == 0);
  // Compare with the reference
  REQUIRE(wf.copyFromGPU() == 0);
  CWaveFunction refWF("src/unittest/expVPsiReal.csv","src/unittest/expVPsiImag.csv");
  REQUIRE(refWF.copyToGPU() == 0);
  REQUIRE(normalize(refWF.getDeviceWF(), vSize, ham.getCoordStep()) == 0);
  REQUIRE(refWF.copyFromGPU() == 0);
  for (int i = 0; i < vSize * vSize; i++)
  {
    REQUIRE(wf.getHostWF()[i][0] == Approx(refWF.getHostWF()[i][0]).margin(eps));
    REQUIRE(wf.getHostWF()[i][1] == Approx(wf.getHostWF()[i][1]).margin(eps));
  }
}

TEST_CASE("Time one split step", "[SPLIT]"){
  cudaProfilerStart();
  size_t N = 512;
  double xmax = 5.0;
  double x0 = 0.0;
  double omega = 0.5;
  double dt = 0.001;
  cufftHandle * plan = NULL;
  CWaveFunction wf(N, xmax, 1.7);
  REQUIRE(wf.getColsSize() > 0);
  REQUIRE(wf.getRowSize() > 0);
  REQUIRE(wf.getRowSize() > 0);
  REQUIRE(wf.getHostWF() != NULL);
  REQUIRE(wf.copyToGPU() == 0);
  C2ParticlesHO ham(N, 0.0, xmax, x0, omega);
  CSplitStep splitStepOperator(&ham);
  int avNum = 10000;
  double sec = 0.0;
  double av = 0.0;
  for (int i = 0; i < avNum; i++)
  {
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    REQUIRE(splitStepOperator.advanceOneStep(dt, 0, wf, plan) == 0);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    sec += time_span.count();
    //std::cout <<"Time step " << i << "takes " << time_span.count() << "sec" << std::endl;
  }
  av = sec / (double)avNum;
  LOG(INFO) << "Total time for "<< avNum << " split steps : " << sec << " , average time per step: " << av << std::endl;
  cufftDestroy(*plan);
  cudaProfilerStop();
}

TEST_CASE("One time step advancement works", "[SPLIT]"){
  resetCudaError();
  std::string realFile("src/unittest/WF1024dt0_001Re.csv");
  std::string imagFile("src/unittest/WF1024dt0_001Im.csv");
  CWaveFunction wf(realFile, imagFile);
  REQUIRE(wf.getColsSize() > 0);
  REQUIRE(wf.getHostWF() != NULL);
  REQUIRE(wf.copyToGPU() == 0);
  // Split step operator
  double xmax = 5.0;
  double x0 = 0.0;
  double omega = 0.5;
  C2ParticlesHO ham(wf.getColsSize(), 0.0, xmax, x0, omega);
  CSplitStep splitStepOperator(&ham);
  REQUIRE(splitStepOperator.ham != NULL);
  // Advance one step
  double curT = 1.1;
  double dt = 0.001;
  cufftHandle * plan = NULL;
  REQUIRE(splitStepOperator.advanceOneStep(dt, curT, wf, plan) == 0);
  REQUIRE(normalize(wf.getDeviceWF(), wf.getColsSize(), ham.getCoordStep()) == 0);
  REQUIRE(wf.copyFromGPU() == 0);
  // Load reference
  CWaveFunction refWF("src/unittest/WF1024dt0_001StepRe.csv","src/unittest/WF1024dt0_001StepIm.csv");
  REQUIRE(refWF.copyToGPU() == 0);
  REQUIRE(normalize(refWF.getDeviceWF(), refWF.getColsSize(), ham.getCoordStep()) == 0);
  REQUIRE(refWF.copyFromGPU() == 0);
  // Compare with the reference
  double eps = pow(10, -3);
  REQUIRE(wf.getColsSize() == refWF.getColsSize());
  REQUIRE(wf.getRowSize() == refWF.getRowSize());
  for (int i = 0; i < wf.getColsSize() * wf.getRowSize(); i++)
  {
    REQUIRE(wf.getHostWF()[i][0] == Approx(refWF.getHostWF()[i][0]).margin(eps));
    REQUIRE(wf.getHostWF()[i][1] == Approx(refWF.getHostWF()[i][1]).margin(eps));
  }
  cufftDestroy(*plan);
}
