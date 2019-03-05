#include <sstream>
#include "catch.hpp"
#include "../easyloggingcpp/easylogging++.h"
#include "../compressed.h"
#include "../sampler.h"
#include "../mxutils.h"
#include "../cs_utils.h"

TEST_CASE("Test compressor constructor", "[CS]"){
  // Both files are good
  CWaveFunction wf("src/unittest/sinbunchwf_real.csv", "src/unittest/sinbunchwf_imag.csv");
  size_t N1 = wf.getRowSize() / 2;
  size_t N2 = wf.getColsSize() / 4;
  size_t range = wf.getColsSize();
  CWaveFunction wfout(N1, N2);
  // Query device information
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int maxThreads = prop.maxThreadsPerBlock;
  CRowColSampler sampler(N1, N2, range, maxThreads);
  CCompressor compressor(wfout, sampler);
  fftw_plan * plan = compressor.getPlan();
  REQUIRE(plan != NULL);
  REQUIRE(fftw_cost(*plan) >= 0);
}

TEST_CASE("thresholded Steepest descent step test: random sampler small vortex lattice", "[CS]"){
  LOG(INFO) << "=======Test restrictedSD with random sample=============";
  // Both files are good
  CWaveFunction wf("src/unittest/2d_real.csv", "src/unittest/2d_imaginary.csv");
  size_t Nr = wf.getRowSize();
  size_t Nc = wf.getColsSize();
  size_t k = 1000;
  size_t range = Nc;
  double eps = 0.000000001;
  CWaveFunction wf2(wf);
  REQUIRE(wf.getHostWF()[(Nr/2-1) * Nc + Nc/2 - 1][0] == Approx(7174.6).margin(eps));
  REQUIRE(wf.getHostWF()[(Nr/2-1) * Nc + Nc/2 - 1][1] == Approx(-0.0000008816).margin(eps));
  REQUIRE(wf.getHostWF()[(Nr/2+99) * Nc + Nc/2 +99][0] == Approx(-5501.8).margin(eps));
  REQUIRE(wf.getHostWF()[(Nr/2+99) * Nc + Nc/2 +99][1] == Approx(-0.0001962300).margin(eps));
  // Query device information
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int maxThreads = prop.maxThreadsPerBlock;
  CWaveFunction wftest("src/unittest/vortexRestrictedSDStepRe.csv", "src/unittest/vortexRestrictedSDStepIm.csv");
  // Force sampling
  size_t n = Nc * Nr / 16;
  CWaveFunction y(n, (size_t)1);
  CRandomSampler sampler(n, wf.getRowSize(), maxThreads);
  size_t * inds = sampler.getIndices_host();
  size_t cnt = 0;
  for (size_t i = 0; i < wf.getRowSize(); i++)
  {
      for (size_t j = 0; j < wf.getColsSize(); j++)
      {
        if (i%4 == 0)
          if (j % 4 == 0)
          {
            inds[cnt] = i*wf.getColsSize() + j;
            cnt++;
          }
      }
  }
  REQUIRE(cnt == n);
  LOG(INFO) << "Sampling";
  REQUIRE(sampler.sample(&wf, &y) == 0);
  LOG(INFO) << "Creating compressor";
  CCompressor compressor(y, sampler);
  wf.switchDomain();
  REQUIRE(wf.getHostWF()[(Nr/2-1) * Nc + Nc/2 - 1][0]  == Approx(0.0248643392).margin(eps));
  REQUIRE(wf.getHostWF()[(Nr/2-1) * Nc + Nc/2 - 1][1] == Approx(-0.0007876791).margin(eps));
  REQUIRE(wf.getHostWF()[(Nr/2+99) * Nc + Nc/2 +99][0] == Approx(0.0013521120).margin(eps));
  REQUIRE(wf.getHostWF()[(Nr/2+99) * Nc + Nc/2 +99][1] == Approx(-0.0006927824).margin(eps));
  wf2.switchDomain();
  char * support = findSupport_sort_host(wf2.getHostWF(), wf2.getColsSize() * wf2.getRowSize(), k, 0, maxThreads);
  REQUIRE(support != NULL);
  REQUIRE(support[5] == 1);
  REQUIRE(support[1019] == 1);
  REQUIRE(support[5120] == 1);
  REQUIRE(compressor.restrictedSD(&wf, &y, k, support) < eps);
  wf.switchDomain();
  for (size_t i = 0; i < wf.getRowSize() * wf.getColsSize(); i++)
  {
    REQUIRE(wf.getHostWF()[i][0] == Approx(wftest.getHostWF()[i][0]).margin(0.001));
    REQUIRE(wf.getHostWF()[i][1] == Approx(wftest.getHostWF()[i][1]).margin(0.001));
  }
  delete [] support;
}

TEST_CASE("Hard threshodling with steepest gradient descent: fully random sampling, non-interacting busch state", "[CS]"){
  LOG(INFO) << "=========Test Hard threshoding: random sampling, busch state g=0 ==========";
  // Both files are good
  CWaveFunction wf("src/unittest/WF1024dt0_001Re.csv", "src/unittest/WF1024dt0_001Im.csv");
  // Query device information
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int maxThreads = prop.maxThreadsPerBlock;
  double eps = 0.0001;
  SECTION("No compression")
  {
    size_t Nr = wf.getRowSize();
    size_t Nc = wf.getColsSize();
    size_t range = Nc;
    CWaveFunction wfout(wf.getRowSize(), wf.getColsSize());
    CWaveFunction y(Nc, Nr);
    CRandomSampler sampler(Nr * Nc, range, maxThreads);
    REQUIRE(sampler.getRange() > 0);
    REQUIRE(sampler.sample(&wf, &y) == 0);
    CCompressor compressor(y, sampler);
    REQUIRE(compressor.thresholdSD(&wfout, &y, Nc*Nr) == Approx(0.0).margin(eps));
    wf.switchDomain();
    for (size_t i = 0; i < wf.getRowSize() * wf.getColsSize(); i++)
    {
      REQUIRE(wfout.getHostWF()[i][0] == Approx(wf.getHostWF()[i][0]).margin(eps));
      REQUIRE(wfout.getHostWF()[i][1] == Approx(wf.getHostWF()[i][1]).margin(eps));
    }
  }
  SECTION("Progressive compression"){
    size_t Nr = wf.getRowSize();
    size_t Nc = wf.getColsSize();
    size_t k = 160000;
    REQUIRE(Nc == Nr);
    size_t range = Nc;
    CWaveFunction wfout(wf.getRowSize(), wf.getColsSize());
    LOG(INFO) << "=======Progressive compression=============";
    for (size_t n = Nc * Nr; n > k; n /= 2)
    {
      CWaveFunction y(n, (size_t)1);
      CRandomSampler sampler(n, range, maxThreads);
      REQUIRE(sampler.getRange() > 0);
      REQUIRE(sampler.sample(&wf, &y) == 0);
      CCompressor compressor(y, sampler);
      compressor.thresholdSD(&wfout, &y, k);
      std::ostringstream fnRe;
      std::ostringstream fnIm;
      fnRe << "random_Busch_reconstructedRe" << n << ".csv";
      fnIm << "random_Busch_reconstructedIm" << n << ".csv";
      wfout.switchDomain();
      wfout *= (1.0 / sqrt(wfout.norm2sqr()));
      saveMatrixToCSV(fnRe.str(), fnIm.str(), wfout.getHostWF(), wfout.getColsSize(), wfout.getRowSize());
      wfout.subtract_host(wf, false);
      LOG(INFO) << "========" << n << " : " << wfout.maxAbs(0, maxThreads) << "=========";
    }
    LOG(INFO) << "====================";
  }
}

TEST_CASE("Hard threshodling with steepest gradient descent: fully random sampling, vortex grid", "[CS]"){
  LOG(INFO) << "=========Test Hard threshoding: random sampling==========";
  // Both files are good
  CWaveFunction wf("src/unittest/2d_real.csv", "src/unittest/2d_imaginary.csv");
  // Query device information
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int maxThreads = prop.maxThreadsPerBlock;
  double eps = 0.0001;
  SECTION("No compression")
  {
    size_t Nr = wf.getRowSize();
    size_t Nc = wf.getColsSize();
    size_t range = Nc;
    CWaveFunction wfout(wf.getRowSize(), wf.getColsSize());
    CWaveFunction y(Nc, Nr);
    CRandomSampler sampler(Nr * Nc, range, maxThreads);
    REQUIRE(sampler.getRange() > 0);
    REQUIRE(sampler.sample(&wf, &y) == 0);
    CCompressor compressor(y, sampler);
    REQUIRE(compressor.thresholdSD(&wfout, &y, Nc*Nr) == Approx(0.0).margin(eps));
    wf.switchDomain();
    for (size_t i = 0; i < wf.getRowSize() * wf.getColsSize(); i++)
    {
      REQUIRE(wfout.getHostWF()[i][0] == Approx(wf.getHostWF()[i][0]).margin(eps));
      REQUIRE(wfout.getHostWF()[i][1] == Approx(wf.getHostWF()[i][1]).margin(eps));
    }
  }
  SECTION("Progressive compression"){
    size_t Nr = wf.getRowSize();
    size_t Nc = wf.getColsSize();
    size_t k = 1000;
    REQUIRE(Nc == Nr);
    size_t range = Nc;
    CWaveFunction wfout(wf.getRowSize(), wf.getColsSize());
    LOG(INFO) << "=======Progressive compression=============";
    for (size_t n = Nc * Nr; n > k; n /= 2)
    {
      CWaveFunction y(n, (size_t)1);
      CRandomSampler sampler(n, range, maxThreads);
      REQUIRE(sampler.getRange() > 0);
      REQUIRE(sampler.sample(&wf, &y) == 0);
      CCompressor compressor(y, sampler);
      compressor.thresholdSD(&wfout, &y, k);
      std::ostringstream fnRe;
      std::ostringstream fnIm;
      fnRe << "randomVortex_reconstructedRe" << n << ".csv";
      fnIm << "randomVortex_reconstructedIm" << n << ".csv";
      wfout.switchDomain();
      saveMatrixToCSV(fnRe.str(), fnIm.str(), wfout.getHostWF(), wfout.getColsSize(), wfout.getRowSize());
      wfout.subtract_host(wf, false);
      LOG(INFO) << "========" << n << " : " << wfout.maxAbs(0, maxThreads) << "=========";
    }
    LOG(INFO) << "====================";
  }
}

TEST_CASE("Hard threshodling with steepest gradient descent: fully random sampling, bunch of sinuses", "[CS]"){
  LOG(INFO) << "=========Test Hard threshoding: random sampling==========";
  // Both files are good
  CWaveFunction wf("src/unittest/sinbunchwf_real.csv", "src/unittest/sinbunchwf_imag.csv");
  // Query device information
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int maxThreads = prop.maxThreadsPerBlock;
  double eps = 0.0001;
  SECTION("No compression")
  {
    size_t Nr = wf.getRowSize();
    size_t Nc = wf.getColsSize();
    size_t range = Nc;
    CWaveFunction wfout(wf.getRowSize(), wf.getColsSize());
    CWaveFunction y(Nc, Nr);
    CRandomSampler sampler(Nr * Nc, range, maxThreads);
    REQUIRE(sampler.getRange() > 0);
    REQUIRE(sampler.sample(&wf, &y) == 0);
    CCompressor compressor(y, sampler);
    REQUIRE(compressor.thresholdSD(&wfout, &y, Nc*Nr) == Approx(0.0).margin(eps));
    wf.switchDomain();
    for (size_t i = 0; i < wf.getRowSize() * wf.getColsSize(); i++)
    {
      REQUIRE(wfout.getHostWF()[i][0] == Approx(wf.getHostWF()[i][0]).margin(eps));
      REQUIRE(wfout.getHostWF()[i][1] == Approx(wf.getHostWF()[i][1]).margin(eps));
    }
  }
  SECTION("Progressive compression"){
    size_t Nr = wf.getRowSize();
    size_t Nc = wf.getColsSize();
    size_t k = 160000;
    REQUIRE(Nc == Nr);
    size_t range = Nc;
    CWaveFunction wfout(wf.getRowSize(), wf.getColsSize());
    LOG(INFO) << "=======Progressive compression=============";
    for (size_t n = Nc * Nr; n > k; n /= 2)
    {
      CWaveFunction y(n, (size_t)1);
      CRandomSampler sampler(n, range, maxThreads);
      REQUIRE(sampler.getRange() > 0);
      REQUIRE(sampler.sample(&wf, &y) == 0);
      CCompressor compressor(y, sampler);
      compressor.thresholdSD(&wfout, &y, k);
      std::ostringstream fnRe;
      std::ostringstream fnIm;
      fnRe << "random_reconstructedRe" << n << ".csv";
      fnIm << "random_reconstructedIm" << n << ".csv";
      wfout.switchDomain();
      wfout *= (1.0 / sqrt(wfout.norm2sqr()));
      saveMatrixToCSV(fnRe.str(), fnIm.str(), wfout.getHostWF(), wfout.getColsSize(), wfout.getRowSize());
      wfout.subtract_host(wf, false);
      LOG(INFO) << "========" << n << " : " << wfout.maxAbs(0, maxThreads) << "=========";
    }
    LOG(INFO) << "====================";
  }
}

TEST_CASE("Hard threshodling with steepest gradient descent: Random RowCol sampling, bunch of sinuses", "[CS]"){
  LOG(INFO) << "=========Test Hard threshoding : rowcol sampling==========";
  // Both files are good
  CWaveFunction wf("src/unittest/sinbunchwf_real.csv", "src/unittest/sinbunchwf_imag.csv");
  // Query device information
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int maxThreads = prop.maxThreadsPerBlock;
  double eps = 0.0001;
  SECTION("No compression")
  {
    size_t Nr = wf.getRowSize();
    size_t Nc = wf.getColsSize();
    size_t range = Nc;
    CWaveFunction wfout(wf.getRowSize(), wf.getColsSize());
    CWaveFunction y(Nc, Nr);
    CRowColSampler sampler(Nr, Nc, range, maxThreads);
    REQUIRE(sampler.getRange() > 0);
    REQUIRE(sampler.sample(&wf, &y) == 0);
    CCompressor compressor(y, sampler);
    REQUIRE(compressor.thresholdSD(&wfout, &y, Nc*Nr) == Approx(0.0).margin(eps));
    wf.switchDomain();
    for (size_t i = 0; i < wf.getRowSize() * wf.getColsSize(); i++)
    {
      REQUIRE(wfout.getHostWF()[i][0] == Approx(wf.getHostWF()[i][0]).margin(eps));
      REQUIRE(wfout.getHostWF()[i][1] == Approx(wf.getHostWF()[i][1]).margin(eps));
    }
  }
  SECTION("Progressive compression"){
    size_t Nr = wf.getRowSize();
    size_t Nc = wf.getColsSize();
    size_t k = 160000;
    REQUIRE(Nc == Nr);
    size_t range = Nc;
    CWaveFunction wfout(wf.getRowSize(), wf.getColsSize());
    LOG(INFO) << "=======Progressive compression=============";
    for (size_t n = Nc; n > 400; n /= 2)
    {
      CWaveFunction y(n, n);
      CRowColSampler sampler(n, n, range, maxThreads);
      REQUIRE(sampler.getRange() > 0);
      REQUIRE(sampler.sample(&wf, &y) == 0);
      CCompressor compressor(y, sampler);
      compressor.thresholdSD(&wfout, &y, k);
      std::ostringstream fnRe;
      std::ostringstream fnIm;
      fnRe << "reconstructedRe" << n << ".csv";
      fnIm << "reconstructedIm" << n << ".csv";
      wfout.switchDomain();
      wfout *= (1.0 / sqrt(wfout.norm2sqr()));
      saveMatrixToCSV(fnRe.str(), fnIm.str(), wfout.getHostWF(), wfout.getColsSize(), wfout.getRowSize());
      wfout.subtract_host(wf, false);
      LOG(INFO) << "========" << n << " : " << wfout.maxAbs(0, maxThreads) << "=========";
    }
    LOG(INFO) << "====================";
  }
}
