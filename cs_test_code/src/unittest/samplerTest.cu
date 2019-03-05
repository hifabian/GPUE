#include "catch.hpp"
#include <cufft.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <sstream>
#include "../easyloggingcpp/easylogging++.h"
#include "../sampler.h"
#include "../mxutils.h"

using namespace std::chrono;

TEST_CASE("Random wave function sampling works (Row/Col sampler)", "[ROWCOLSAMPLER]"){
  // Both files are good
  CWaveFunction wf("src/unittest/sinbunchwf_real.csv", "src/unittest/sinbunchwf_imag.csv");
  size_t Nc = wf.getColsSize() / 2;
  size_t Nr = wf.getRowSize() / 4;
  size_t range = wf.getColsSize();
  CWaveFunction wfout(Nc, Nr);
  // Query device information
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int maxThreads = prop.maxThreadsPerBlock;
  CRowColSampler sampler(Nr, Nc, range, maxThreads);
  size_t * h_rows = sampler.getRows_host();
  size_t * h_cols = sampler.getCols_host();
  SECTION("Good case"){
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    REQUIRE(sampler.sample(&wf, &wfout) == 0);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    LOG(INFO) << "Total time for random" << Nr <<  "x " << Nc << " sampling of a wavefunction of size " << range << " x " << range << "is " << time_span.count() << "\n";
    for (int i = 0; i < Nr; i++){
      for (int j = 0; j < Nc; j++){
        int curCIdx = i * Nc + j;
        int curOIdx = h_rows[i] * range + h_cols[j];
        REQUIRE(wfout.getHostWF()[curCIdx][0] == wf.getHostWF()[curOIdx][0]);
        REQUIRE(wfout.getHostWF()[curCIdx][1] == wf.getHostWF()[curOIdx][1]);
      }
    }
  }
  SECTION("Bad cases"){
    REQUIRE(sampler.sample(NULL, &wfout) == -1);
    REQUIRE(sampler.sample(&wf, NULL) == -1);
    CWaveFunction tmpwf;
    REQUIRE(sampler.sample(&wf, &tmpwf) == -1);
  }
}

TEST_CASE("Test Upsampling of the wave function (Row/Col sampler)", "[ROWCOLSAMPLER]"){
  // Both files are good
  CWaveFunction wf("src/unittest/sinbunchwf_real.csv", "src/unittest/sinbunchwf_imag.csv");
  CWaveFunction dstwf(wf.getColsSize(), wf.getRowSize());
  SECTION("Good case")
  {
    size_t Nr = wf.getRowSize() / 2;
    size_t Nc = wf.getColsSize() / 4;
    size_t range = wf.getColsSize();
    // Query device information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreads = prop.maxThreadsPerBlock;
    CWaveFunction cwf(Nc, Nr);
    CRowColSampler sampler(Nr, Nc, range, maxThreads);
    size_t * h_rows = sampler.getRows_host();
    size_t * h_cols = sampler.getCols_host();
    REQUIRE(sampler.sample(&wf, &cwf) == 0);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    REQUIRE(sampler.upsample(&cwf, &dstwf) == 0);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    LOG(INFO) << "Total time for " << Nr <<  "x " << Nc << " upsampling of a wavefunction of size " << range << " x " << range << "is " << time_span.count() << "\n";
    for (size_t i = 0; i < cwf.getRowSize(); i++)
    {
      for (size_t j = 0; j < cwf.getColsSize(); j++)
      {
        size_t curOIdx = h_rows[i] * range + h_cols[j];
        REQUIRE(wf.getHostWF()[curOIdx][0] == dstwf.getHostWF()[curOIdx][0]);
        REQUIRE(wf.getHostWF()[curOIdx][1] == dstwf.getHostWF()[curOIdx][1]);
        dstwf.getHostWF()[curOIdx][0] = 0.0;
        dstwf.getHostWF()[curOIdx][1] = 0.0;
      }
    }
    for (size_t i = 0; i < dstwf.getRowSize(); i++)
    {
      for (size_t j = 0; j < dstwf.getColsSize(); j++)
      {
        size_t curOIdx = i * dstwf.getColsSize() + j;
        REQUIRE(dstwf.getHostWF()[curOIdx][0] == 0.0);
        REQUIRE(dstwf.getHostWF()[curOIdx][1] == 0.0);
      }
    }
  }
  SECTION("No compression case")
  {
    size_t Nr = wf.getRowSize();
    size_t Nc = wf.getColsSize();
    size_t range = Nc;
    // Query device information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreads = prop.maxThreadsPerBlock;
    CWaveFunction cwf(Nc, Nr);
    CRowColSampler sampler(Nr, Nc, range, maxThreads);
    size_t * h_rows = sampler.getRows_host();
    size_t * h_cols = sampler.getCols_host();
    REQUIRE(sampler.sample(&wf, &cwf) == 0);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    REQUIRE(sampler.upsample(&cwf, &dstwf) == 0);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    LOG(INFO) << "Total time for " << Nr <<  "x " << Nc << " upsampling of a wavefunction of size " << range << " x " << range << "is " << time_span.count() << "\n";
    for (size_t i = 0; i < cwf.getRowSize(); i++)
    {
      for (size_t j = 0; j < cwf.getColsSize(); j++)
      {
        size_t curOIdx = h_rows[i] * range + h_cols[j];
        REQUIRE(wf.getHostWF()[curOIdx][0] == dstwf.getHostWF()[curOIdx][0]);
        REQUIRE(wf.getHostWF()[curOIdx][1] == dstwf.getHostWF()[curOIdx][1]);
        dstwf.getHostWF()[curOIdx][0] = 0.0;
        dstwf.getHostWF()[curOIdx][1] = 0.0;
      }
    }
    for (size_t i = 0; i < dstwf.getRowSize(); i++)
    {
      for (size_t j = 0; j < dstwf.getColsSize(); j++)
      {
        size_t curOIdx = i * dstwf.getColsSize() + j;
        REQUIRE(dstwf.getHostWF()[curOIdx][0] == 0.0);
        REQUIRE(dstwf.getHostWF()[curOIdx][1] == 0.0);
      }
    }
  }
}

TEST_CASE("Test application of the sampling operator back and forth", "[ROWCOLSAMPLER]"){
  LOG(INFO) << "======Test application of the sampling operator=====";
  CWaveFunction wf("src/unittest/sinbunchwf_real.csv", "src/unittest/sinbunchwf_imag.csv");
  REQUIRE(wf.getColsSize() != 0);
  REQUIRE(wf.getRowSize() != 0);
  REQUIRE(wf.getDomain() == true);
  CWaveFunction wf2(wf);
  SECTION("The wave function is in freq domain")
  {
    wf.switchDomain();
    REQUIRE(wf.getDomain() == false);
  }
  SECTION("The wave function is in space domain")
  {
    REQUIRE(wf.getDomain() == true);
  }
  SECTION("No compression")
  {
    size_t Nr = wf.getRowSize();
    size_t Nc = wf.getColsSize();
    size_t range = wf.getColsSize();
    // Query device information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreads = prop.maxThreadsPerBlock;
    CWaveFunction cwf(Nc, Nr);
    CWaveFunction cwf2(Nc, Nr);
    CRowColSampler sampler(Nr, Nc, range, maxThreads);
    REQUIRE(sampler.applySamplingOperator(&wf, &cwf) == 0);
    REQUIRE(wf.getDomain() == true);
    REQUIRE(cwf.getDomain() == true);
    REQUIRE(sampler.sample(&wf, &cwf2) == 0);
    REQUIRE(cwf.getDomain() == cwf2.getDomain());
    for (size_t i = 0; i < cwf.getColsSize() * cwf.getRowSize(); i++)
    {
      REQUIRE(cwf.getHostWF()[i][0] == cwf2.getHostWF()[i][0]);
      REQUIRE(cwf.getHostWF()[i][1] == cwf2.getHostWF()[i][1]);
    }
    REQUIRE(sampler.applyInverseSamplingOperator(&cwf, &wf) == 0);
    REQUIRE(wf.getDomain() == false);
    wf.switchDomain();
    REQUIRE(wf.getDomain() == true);
    for (size_t i = 0; i < wf.getColsSize() * wf.getRowSize(); i++)
    {
      REQUIRE(wf.getHostWF()[i][0] == Approx(wf2.getHostWF()[i][0]).margin(0.001));
      REQUIRE(wf.getHostWF()[i][1] == Approx(wf2.getHostWF()[i][1]).margin(0.001));
    }
    REQUIRE(sampler.upsample(&cwf, &wf2) == 0);
    for (size_t i = 0; i < wf.getColsSize() * wf.getRowSize(); i++)
    {
      REQUIRE(wf.getHostWF()[i][0] == Approx(wf2.getHostWF()[i][0]).margin(0.001));
      REQUIRE(wf.getHostWF()[i][1] == Approx(wf2.getHostWF()[i][1]).margin(0.001));
    }
  }
  SECTION("Some compression")
  {
    size_t Nr = wf.getRowSize() / 2;
    size_t Nc = wf.getColsSize();
    size_t range = wf.getColsSize();
    // Query device information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreads = prop.maxThreadsPerBlock;
    CWaveFunction cwf(Nc, Nr);
    CWaveFunction cwf2(Nc, Nr);
    CRowColSampler sampler(Nr, Nc, range, maxThreads);
    REQUIRE(sampler.applySamplingOperator(&wf, &cwf) == 0);
    REQUIRE(wf.getDomain() == true);
    REQUIRE(cwf.getDomain() == true);
    REQUIRE(sampler.sample(&wf, &cwf2) == 0);
    REQUIRE(cwf.getDomain() == cwf2.getDomain());
    for (size_t i = 0; i < cwf.getColsSize() * cwf.getRowSize(); i++)
    {
      REQUIRE(cwf.getHostWF()[i][0] == cwf2.getHostWF()[i][0]);
      REQUIRE(cwf.getHostWF()[i][1] == cwf2.getHostWF()[i][1]);
    }
    REQUIRE(sampler.applyInverseSamplingOperator(&cwf, &wf) == 0);
    REQUIRE(wf.getDomain() == false);
    wf.switchDomain();
    REQUIRE(wf.getDomain() == true);
    // Tested visually if the result seems legit. It does
    REQUIRE(sampler.upsample(&cwf, &wf2) == 0);
    for (size_t i = 0; i < wf.getColsSize() * wf.getRowSize(); i++)
    {
      REQUIRE(wf.getHostWF()[i][0] == Approx(wf2.getHostWF()[i][0]).margin(0.001));
      REQUIRE(wf.getHostWF()[i][1] == Approx(wf2.getHostWF()[i][1]).margin(0.001));
    }
  }
}

//============= CRandomSampler

TEST_CASE("Random wave function sampling works (Random sampler)", "[RANDOMSAMPLER]"){
  // Both files are good
  CWaveFunction wf("src/unittest/sinbunchwf_real.csv", "src/unittest/sinbunchwf_imag.csv");
  size_t N = wf.getColsSize() * wf.getRowSize() / 2;
  size_t range = wf.getColsSize();
  CWaveFunction wfout(N, (size_t)1);
  REQUIRE(wfout.getRowSize() == 1);
  REQUIRE(wfout.getColsSize() == N);
  // Query device information
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int maxThreads = prop.maxThreadsPerBlock;
  CRandomSampler sampler(N, range, maxThreads);
  size_t * h_indices = sampler.getIndices_host();
  SECTION("Good case"){
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    REQUIRE(sampler.sample(&wf, &wfout) == 0);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    LOG(INFO) << "Total time for random" << 1 <<  "x " << N << " sampling of a wavefunction of size " << range << " x " << range << "is " << time_span.count() << "\n";
    for (size_t j = 0; j < N; j++){
      int curCIdx = j;
      int curOIdx = h_indices[j];
      REQUIRE(wfout.getHostWF()[curCIdx][0] == wf.getHostWF()[curOIdx][0]);
      REQUIRE(wfout.getHostWF()[curCIdx][1] == wf.getHostWF()[curOIdx][1]);
    }
  }
  SECTION("Bad cases"){
    REQUIRE(sampler.sample(NULL, &wfout) == -1);
    REQUIRE(sampler.sample(&wf, NULL) == -1);
    CWaveFunction tmpwf;
    REQUIRE(sampler.sample(&wf, &tmpwf) == -1);
  }
}

TEST_CASE("Test Upsampling of the wave function (Random sampler)", "[RANDOMSAMPLER]"){
  LOG(INFO) << "============Test random upsampling======";
  // Both files are good
  CWaveFunction wf("src/unittest/sinbunchwf_real.csv", "src/unittest/sinbunchwf_imag.csv");
  CWaveFunction dstwf(wf.getColsSize(), wf.getRowSize());
  SECTION("Good case")
  {
    size_t Nr = 1;
    size_t Nc = wf.getColsSize()*wf.getRowSize() / 4;
    size_t range = wf.getColsSize();
    // Query device information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreads = prop.maxThreadsPerBlock;
    CWaveFunction cwf(Nc, Nr);
    CRandomSampler sampler(Nc, range, maxThreads);
    size_t * h_indices = sampler.getIndices_host();
    REQUIRE(sampler.sample(&wf, &cwf) == 0);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    REQUIRE(sampler.upsample(&cwf, &dstwf) == 0);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    LOG(INFO) << "Total time for " << Nr <<  "x " << Nc << " upsampling of a wavefunction of size " << range << " x " << range << "is " << time_span.count() << "\n";
    for (size_t j = 0; j < Nc; j++)
    {
      size_t curOIdx = h_indices[j];
      REQUIRE(wf.getHostWF()[curOIdx][0] == dstwf.getHostWF()[curOIdx][0]);
      REQUIRE(wf.getHostWF()[curOIdx][1] == dstwf.getHostWF()[curOIdx][1]);
      dstwf.getHostWF()[curOIdx][0] = 0.0;
      dstwf.getHostWF()[curOIdx][1] = 0.0;
    }
    for (size_t i = 0; i < dstwf.getRowSize(); i++)
    {
      for (size_t j = 0; j < dstwf.getColsSize(); j++)
      {
        size_t curOIdx = i * dstwf.getColsSize() + j;
        REQUIRE(dstwf.getHostWF()[curOIdx][0] == 0.0);
        REQUIRE(dstwf.getHostWF()[curOIdx][1] == 0.0);
      }
    }
  }
  SECTION("No compression case")
  {
    size_t Nr = 1;
    size_t Nc = wf.getColsSize() * wf.getRowSize();
    size_t range = wf.getColsSize();
    // Query device information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreads = prop.maxThreadsPerBlock;
    CWaveFunction cwf(Nc, Nr);
    CRandomSampler sampler(Nc, range, maxThreads);
    size_t * h_indices = sampler.getIndices_host();
    REQUIRE(sampler.sample(&wf, &cwf) == 0);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    REQUIRE(sampler.upsample(&cwf, &dstwf) == 0);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    LOG(INFO) << "Total time for " << Nr <<  "x " << Nc << " upsampling of a wavefunction of size " << range << " x " << range << "is " << time_span.count() << "\n";
    for (size_t j = 0; j < Nc; j++)
    {
      size_t curOIdx = h_indices[j];
      REQUIRE(wf.getHostWF()[curOIdx][0] == dstwf.getHostWF()[curOIdx][0]);
      REQUIRE(wf.getHostWF()[curOIdx][1] == dstwf.getHostWF()[curOIdx][1]);
      dstwf.getHostWF()[curOIdx][0] = 0.0;
      dstwf.getHostWF()[curOIdx][1] = 0.0;
    }
    for (size_t i = 0; i < dstwf.getRowSize(); i++)
    {
      for (size_t j = 0; j < dstwf.getColsSize(); j++)
      {
        size_t curOIdx = i * dstwf.getColsSize() + j;
        REQUIRE(dstwf.getHostWF()[curOIdx][0] == 0.0);
        REQUIRE(dstwf.getHostWF()[curOIdx][1] == 0.0);
      }
    }
  }
}

TEST_CASE("Test application of the sampling operator back and forth (Random sampler)", "[RANDOMSAMPLER]"){
  LOG(INFO) << "======Test application of the random sampling operator=====";
  CWaveFunction wf("src/unittest/sinbunchwf_real.csv", "src/unittest/sinbunchwf_imag.csv");
  REQUIRE(wf.getColsSize() != 0);
  REQUIRE(wf.getRowSize() != 0);
  REQUIRE(wf.getDomain() == true);
  CWaveFunction wf2(wf);
  SECTION("The wave function is in freq domain")
  {
    wf.switchDomain();
    REQUIRE(wf.getDomain() == false);
  }
  SECTION("The wave function is in space domain")
  {
    REQUIRE(wf.getDomain() == true);
  }
  SECTION("No compression")
  {
    size_t Nr = 1;
    size_t Nc = wf.getColsSize() * wf.getRowSize();
    size_t range = wf.getColsSize();
    // Query device information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreads = prop.maxThreadsPerBlock;
    CWaveFunction cwf(Nc, Nr);
    CWaveFunction cwf2(Nc, Nr);
    CRandomSampler sampler(Nc, range, maxThreads);
    REQUIRE(sampler.applySamplingOperator(&wf, &cwf) == 0);
    REQUIRE(wf.getDomain() == true);
    REQUIRE(cwf.getDomain() == true);
    REQUIRE(sampler.sample(&wf, &cwf2) == 0);
    REQUIRE(cwf.getDomain() == cwf2.getDomain());
    for (size_t i = 0; i < Nc; i++)
    {
      REQUIRE(cwf.getHostWF()[i][0] == cwf2.getHostWF()[i][0]);
      REQUIRE(cwf.getHostWF()[i][1] == cwf2.getHostWF()[i][1]);
    }
    REQUIRE(sampler.applyInverseSamplingOperator(&cwf, &wf) == 0);
    REQUIRE(wf.getDomain() == false);
    wf.switchDomain();
    REQUIRE(wf.getDomain() == true);
    // std::ostringstream fnameRe;
    // fnameRe << "rand_compressedRe.csv";
    // std::ostringstream fnameIm;
    // fnameIm << "rand_compressedIm.csv";
    // saveMatrixToCSV(fnameRe.str(), fnameIm.str(), wf.getHostWF(), wf.getColsSize(),wf.getRowSize());
    // Visual inspection confirmed sanity
    for (size_t i = 0; i < Nc; i++)
    {
      REQUIRE(wf.getHostWF()[i][0] == Approx(wf2.getHostWF()[i][0]).margin(0.001));
      REQUIRE(wf.getHostWF()[i][1] == Approx(wf2.getHostWF()[i][1]).margin(0.001));
    }
    REQUIRE(sampler.upsample(&cwf, &wf2) == 0);
    for (size_t i = 0; i < Nc; i++)
    {
      REQUIRE(wf.getHostWF()[i][0] == Approx(wf2.getHostWF()[i][0]).margin(0.001));
      REQUIRE(wf.getHostWF()[i][1] == Approx(wf2.getHostWF()[i][1]).margin(0.001));
    }
  }
  SECTION("Some compression")
  {
    size_t Nr = 1;
    size_t Nc = wf.getRowSize() * wf.getColsSize() / 2;
    size_t range = wf.getColsSize();
    // Query device information
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int maxThreads = prop.maxThreadsPerBlock;
    CWaveFunction cwf(Nc, Nr);
    CWaveFunction cwf2(Nc, Nr);
    CRandomSampler sampler(Nc, range, maxThreads);
    REQUIRE(sampler.applySamplingOperator(&wf, &cwf) == 0);
    REQUIRE(wf.getDomain() == true);
    REQUIRE(cwf.getDomain() == true);
    REQUIRE(sampler.sample(&wf, &cwf2) == 0);
    REQUIRE(cwf.getDomain() == cwf2.getDomain());
    for (size_t i = 0; i < Nc; i++)
    {
      REQUIRE(cwf.getHostWF()[i][0] == cwf2.getHostWF()[i][0]);
      REQUIRE(cwf.getHostWF()[i][1] == cwf2.getHostWF()[i][1]);
    }
    REQUIRE(sampler.applyInverseSamplingOperator(&cwf, &wf) == 0);
    REQUIRE(wf.getDomain() == false);
    wf.switchDomain();
    REQUIRE(wf.getDomain() == true);
    // std::ostringstream fnameRe;
    // fnameRe << "rand_compressedRe.csv";
    // std::ostringstream fnameIm;
    // fnameIm << "rand_compressedIm.csv";
    // saveMatrixToCSV(fnameRe.str(), fnameIm.str(), wf.getHostWF(), wf.getColsSize(),wf.getRowSize());
    // Visual inspection confirmed sanity
    REQUIRE(sampler.upsample(&cwf, &wf2) == 0);
    for (size_t i = 0; i < Nc; i++)
    {
      REQUIRE(wf.getHostWF()[i][0] == Approx(wf2.getHostWF()[i][0]).margin(0.001));
      REQUIRE(wf.getHostWF()[i][1] == Approx(wf2.getHostWF()[i][1]).margin(0.001));
    }
  }
}

TEST_CASE("Test forced even sampler (random sampler) on the small vortex lattice case", "RANDOMSAMPLER"){
  // Both files are good
  CWaveFunction wf("src/unittest/2d_real.csv", "src/unittest/2d_imaginary.csv");
  size_t Nr = wf.getRowSize();
  size_t Nc = wf.getColsSize();
  double eps = 0.0000000001;
  //LOG(INFO) << wf.getHostWF()[0][0] << " " << wf.getHostWF()[0][1];
  //LOG(INFO) << wf.getHostWF()[1023][0] << " " << wf.getHostWF()[1023][1];
  REQUIRE(wf.getHostWF()[(Nr/2-1) * Nc + Nc/2 - 1][0] == Approx(7174.6).margin(eps));
  REQUIRE(wf.getHostWF()[(Nr/2-1) * Nc + Nc/2 - 1][1] == Approx(-0.0000008816).margin(eps));
  REQUIRE(wf.getHostWF()[(Nr/2+99) * Nc + Nc/2 +99][0] == Approx(-5501.8).margin(eps));
  REQUIRE(wf.getHostWF()[(Nr/2+99) * Nc + Nc/2 +99][1] == Approx(-0.0001962300).margin(eps));
  // Query device information
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int maxThreads = prop.maxThreadsPerBlock;
  CWaveFunction wftest("src/unittest/VortexSampledRe.csv", "src/unittest/VortexSampledIm.csv");
  size_t N = wf.getColsSize() * wf.getRowSize() / 16;
  CWaveFunction wfout(N, (size_t)1);
  REQUIRE(wftest.getColsSize()*wftest.getRowSize() == wfout.getColsSize()*wfout.getRowSize());
  CRandomSampler sampler(N, wf.getRowSize(), maxThreads);
  size_t * inds = sampler.getIndices_host();
  size_t cnt = 0;
  //LOG(INFO) << "Forcing sampling";
  for (size_t i = 0; i < wf.getRowSize(); i++)
  {
      if (i%4 == 0)
        for (size_t j = 0; j < wf.getColsSize(); j++)
        {
            if (j % 4 == 0)
            {
              inds[cnt] = i*wf.getColsSize() + j;
              cnt++;
            }
        }
  }
  REQUIRE(cnt == N);
  //LOG(INFO) << "Sampling";
  sampler.sample(&wf, &wfout);
  //LOG(INFO) << "Check the result";
  for (size_t i = 0; i < N; i++)
  {
    //LOG(INFO) << i << " :" << wfout.getHostWF()[i][0] << " " << wftest.getHostWF()[i][0];
    REQUIRE(wfout.getHostWF()[i][0] == Approx(wftest.getHostWF()[i][0]));
    REQUIRE(wfout.getHostWF()[i][1] == Approx(wftest.getHostWF()[i][1]));
  }
}

TEST_CASE("Test forced even sampler operator (random sampler) on the small vortex lattice case", "RANDOMSAMPLER"){
  // Both files are good
  CWaveFunction wf("src/unittest/2d_real.csv", "src/unittest/2d_imaginary.csv");
  size_t Nr = wf.getRowSize();
  size_t Nc = wf.getColsSize();
  double eps = 0.00001;
  //LOG(INFO) << wf.getHostWF()[0][0] << " " << wf.getHostWF()[0][1];
  //LOG(INFO) << wf.getHostWF()[1023][0] << " " << wf.getHostWF()[1023][1];
  REQUIRE(wf.getHostWF()[(Nr/2-1) * Nc + Nc/2 - 1][0] == Approx(7174.6).margin(eps));
  REQUIRE(wf.getHostWF()[(Nr/2-1) * Nc + Nc/2 - 1][1] == Approx(-0.0000008816).margin(eps));
  REQUIRE(wf.getHostWF()[(Nr/2+99) * Nc + Nc/2 +99][0] == Approx(-5501.8).margin(eps));
  REQUIRE(wf.getHostWF()[(Nr/2+99) * Nc + Nc/2 +99][1] == Approx(-0.0001962300).margin(eps));
  // Query device information
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int maxThreads = prop.maxThreadsPerBlock;
  CWaveFunction wftest("src/unittest/VortexSampledRe.csv", "src/unittest/VortexSampledIm.csv");
  size_t N = wf.getColsSize() * wf.getRowSize() / 16;
  CWaveFunction wfout(N, (size_t)1);
  REQUIRE(wftest.getColsSize()*wftest.getRowSize() == wfout.getColsSize()*wfout.getRowSize());
  CRandomSampler sampler(N, wf.getRowSize(), maxThreads);
  size_t * inds = sampler.getIndices_host();
  size_t cnt = 0;
  //LOG(INFO) << "Forcing sampling";
  for (size_t i = 0; i < wf.getRowSize(); i++)
  {
      if (i%4 == 0)
        for (size_t j = 0; j < wf.getColsSize(); j++)
        {
            if (j % 4 == 0)
            {
              inds[cnt] = i*wf.getColsSize() + j;
              cnt++;
            }
        }
  }
  REQUIRE(cnt == N);
  wf.switchDomain();
  for (size_t i = 0; i < 10; i++)
  {
    LOG(INFO) << i << ": " << wf.getHostWF()[i][0] << " +i" << wf.getHostWF()[i][1];
  }
  //LOG(INFO) << "Sampling";
  sampler.applySamplingOperator(&wf, &wfout);
  //LOG(INFO) << "Check the result";
  for (size_t i = 0; i < N; i++)
  {
    //LOG(INFO) << i << " :" << wfout.getHostWF()[i][0] << " " << wftest.getHostWF()[i][0];
    REQUIRE(wfout.getHostWF()[i][0] == Approx(wftest.getHostWF()[i][0]).margin(eps));
    REQUIRE(wfout.getHostWF()[i][1] == Approx(wftest.getHostWF()[i][1]).margin(eps));
  }
}
