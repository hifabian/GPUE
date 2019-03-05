#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include "../wavefunction.h"
#include "catch.hpp"
#include "../easyloggingcpp/easylogging++.h"
#include "../mxutils.h"


using namespace std;
using namespace std::chrono;

TEST_CASE("Initial wavefunction generation E=1.0", "[INIT]"){
  int N = 64;
  double x0 = 0.0;
  double omega = 1.0;
  double xmax = 5.0;
  double E = 1.0;
  //cout << "init" << endl;
  double * wf = calcInitialPsi(N, E, x0, xmax, omega);
  //cout << "WF is calculated" << endl;
  REQUIRE(wf != NULL);
  //string fn("gs1.csv");
  //remove(fn.c_str());
  //saveMatrixToCSV(fn, wf, N, N);
  // Compare our wf with reference wf from matlab
  double epsilon = pow(10, -5);
  std::string refFn( "src/unittest/buschStateEn1.csv");
  ifstream infile(refFn);
  REQUIRE(infile.good());
  size_t refSzx = 0;
  size_t refSzy = 0;
  //cout <<"parsing\n";
  double * refMx = parseMxFromCSV(refFn, refSzx, refSzy);
  //cout << "Parsed\n";
  REQUIRE(refSzx == N);
  REQUIRE(refSzy == N);
  //cout << "compare\n";
  for (int i = 0; i < N*N; i++)
  {
    REQUIRE(abs(wf[i] - refMx[i]) < epsilon);
  }
  delete [] wf;
  delete [] refMx;
}

TEST_CASE("Save bigger Busch state", "[INIT]"){
  int N = 512;
  double xmax = 5.0;
  double E = 1.7;
  CWaveFunction wf(N, xmax, E);
  REQUIRE(wf.getHostWF() != NULL);
  saveMatrixToCSV("buschState512En1_7Real.csv", "buschState512En1_7Imag.csv", wf.getHostWF(), N, N);
}

TEST_CASE("Initial wavefunction generation E=1.7", "[INIT]"){
  int N = 64;
  double x0 = 0.0;
  double omega = 1.0;
  double xmax = 5.0;
  double E = 1.7;
  double * wf = calcInitialPsi(N, E, x0, xmax, omega);
  REQUIRE(wf != NULL);
  // Compare our wf with reference wf from matlab
  double epsilon = pow(10, -5);
  std::string refFn = "src/unittest/buschStateEn1_7.csv";
  ifstream infile(refFn);
  REQUIRE(infile.good());
  size_t refSzx = 0;
  size_t refSzy = 0;
  double * refMx = parseMxFromCSV(refFn, refSzx, refSzy);
  REQUIRE(refSzx == N);
  REQUIRE(refSzy == N);
  for (int i = 0; i < N*N; i++)
  {
    REQUIRE(abs(wf[i] - refMx[i]) < epsilon);
  }
  delete [] refMx;
  delete [] wf;
}

TEST_CASE("Wavefunction default constructor and destructor work","[WF]"){
  int N = 64;
  // GOOD case
  CWaveFunction * wf = new CWaveFunction(N, 5.0);
  REQUIRE(wf->getColsSize() == N);
  REQUIRE(wf->getHostWF() != NULL);
  // Compare our wf with reference wf from matlab
  double epsilon = pow(10, -5);
  std::string refFn = "src/unittest/buschStateEn1_7.csv";
  ifstream infile(refFn);
  REQUIRE(infile.good());
  size_t refSzx = 0;
  size_t refSzy = 0;
  double * refMx = parseMxFromCSV(refFn, refSzx, refSzy);
  REQUIRE(refSzx == N);
  REQUIRE(refSzy == N);
  for (int i = 0; i < N*N; i++)
  {
    REQUIRE(abs(wf->getHostWF()[i][0] - refMx[i]) < epsilon);
  }
  delete [] refMx;
  // Delete and check if destructor works
  delete wf;
  wf = NULL;
  // BAD case
  CWaveFunction wfB1(0, 5.0);
  REQUIRE(wfB1.getColsSize() == 0);
  REQUIRE(wfB1.getRowSize() == 0);
  REQUIRE(wfB1.getHostWF() == NULL);
  CWaveFunction wfB2(N, -5.0);
  REQUIRE(wfB2.getColsSize() == 0);
  REQUIRE(wfB2.getHostWF() == NULL);
}

TEST_CASE("Test construction of an empty wave function of given size", "[WF]"){
  size_t Nx = 512;
  size_t Ny = 256;
  SECTION("Good case")
  {
    CWaveFunction wf(Nx, Ny);
    REQUIRE(wf.getRowSize() == Ny);
    REQUIRE(wf.getColsSize() == Nx);
    fftw_complex * h_complex = wf.getHostWF();
    REQUIRE(h_complex != NULL);
    for (size_t i = 0; i < Ny * Nx; i++)
    {
      REQUIRE(h_complex[i][0] == 0);
      REQUIRE(h_complex[i][1] == 0);
    }
  }
  SECTION("Bad cases")
  {
    CWaveFunction wf1(0, Nx);
    REQUIRE(wf1.getHostWF() == NULL);
    CWaveFunction wf2(Ny, (size_t)0);
    REQUIRE(wf2.getHostWF() == NULL);
  }
}

TEST_CASE("Wavefunction load from files works","[WF]"){
  int N = 64;
  // GOOD case
  std::string refFn = "src/unittest/expVReal.csv";
  // Both files are good
  CWaveFunction * wf = new CWaveFunction(refFn, refFn);
  REQUIRE(wf->getColsSize() == N);
  REQUIRE(wf->getRowSize() == N);
  REQUIRE(wf->getRowSize() == N);
  REQUIRE(wf->getHostWF() != NULL);
  REQUIRE(wf->getHostWF()[0][0] != 0.0);
  REQUIRE(wf->getHostWF()[0][1] != 0.0);
  // Compare our wf with reference wf from matlab
  double epsilon = pow(10, -5);
  ifstream infile(refFn);
  REQUIRE(infile.good());
  size_t refSzx = 0;
  size_t refSzy = 0;
  double * refMx = parseMxFromCSV(refFn, refSzx, refSzy);
  REQUIRE(refMx[0] != 0.0);
  REQUIRE(refSzx == N);
  REQUIRE(refSzy == N);
  for (int i = 0; i < N*N; i++)
  {
    REQUIRE(abs(wf->getHostWF()[i][0] - refMx[i]) < epsilon);
    REQUIRE(abs(wf->getHostWF()[i][1] - refMx[i]) < epsilon);
  }
  delete wf;
  // Real file is empty
  wf = new CWaveFunction("", refFn);
  REQUIRE(wf->getColsSize() == 0);
  REQUIRE(wf->getHostWF() == NULL);
  delete wf;
  // Imaginary file is nonexistent
  wf = new CWaveFunction(refFn, "absfbhsdbgshdbgj.csv");
  REQUIRE(wf->getColsSize() == 0);
  REQUIRE(wf->getRowSize() == 0);
  REQUIRE(wf->getHostWF() == NULL);
  delete wf;
  // Both are non-existent
  wf = new CWaveFunction("fskgslkg.csv", "absfbhsdbgshdbgj.csv");
  REQUIRE(wf->getColsSize() == 0);
  REQUIRE(wf->getHostWF() == NULL);
  delete wf;
  delete [] refMx;
}

TEST_CASE("Copy to GPU works","[WF]"){
  //int N = 64;
  // GOOD case
  std::string refFn = "src/unittest/buschStateEn1_7.csv";
  // Both files are good
  CWaveFunction * wf = new CWaveFunction(refFn, refFn);
  wf->copyToGPU();
  cufftDoubleComplex * d_wf = wf->getDeviceWF();
  REQUIRE(d_wf != NULL);
  REQUIRE(wf->getPitch() > 0);
  REQUIRE(d_wf == wf->getDeviceWF());
  delete wf;
  // Bad case
  wf = new CWaveFunction(0, 5.0);
  wf->copyToGPU();
  d_wf = wf->getDeviceWF();
  REQUIRE(d_wf == NULL);
  REQUIRE(wf->getPitch() == 0);
  delete wf;
}

TEST_CASE("Copy from GPU works","[WF]"){
  LOG(INFO) << "======testing copyToGPU==========";
  // GOOD case
  std::string refFn = "src/unittest/buschStateEn1_7.csv";
  // Both files are good
  CWaveFunction * wf = new CWaveFunction(refFn, refFn);
  REQUIRE(wf->getHostWF()[wf->getColsSize() * wf->getRowSize() / 2 + wf->getColsSize() / 2][0] != 0.0);
  REQUIRE(wf->getHostWF()[wf->getColsSize() * wf->getRowSize() / 2 + wf->getColsSize() / 2][1] != 0.0);
  REQUIRE(wf->copyToGPU() == 0);
  cufftDoubleComplex * d_wf = wf->getDeviceWF();
  REQUIRE(d_wf != NULL);
  // Copy real part into a buffer
  fftw_complex * h_ccopy = new fftw_complex[wf->getColsSize() * wf->getRowSize()];
  fftw_complex * h_c = wf->getHostWF();
  memcpy(h_ccopy, h_c, sizeof(fftw_complex) * wf->getColsSize() * wf->getRowSize());
  // Set both imaginary and real parts to zeros
  memset((double * )h_c, 0.0, sizeof(double)*2 * wf->getColsSize() * wf->getRowSize());
  // Copy from GPU
  REQUIRE(wf->copyFromGPU() == 0);
  REQUIRE(wf->getHostWF()[wf->getColsSize() * wf->getRowSize() / 2 + wf->getColsSize() / 2][0] != 0.0);
  REQUIRE(wf->getHostWF()[wf->getColsSize() * wf->getRowSize() / 2 + wf->getColsSize() / 2][1] != 0.0);
  // Check the copy
  for (int i = 0; i < wf->getColsSize() * wf->getRowSize(); i++)
  {
    REQUIRE(h_c[i][0] == h_ccopy[i][0]);
    REQUIRE(h_c[i][1] == h_ccopy[i][1]);
  }
  delete [] h_ccopy;
  delete wf;
  // Only real part is there
  wf = new CWaveFunction(64, 5.0);
  wf->copyToGPU();
  d_wf = wf->getDeviceWF();
  REQUIRE(d_wf != NULL);
  REQUIRE(wf->copyFromGPU() == 0);
  for (int i = 0; i < wf->getColsSize() * wf->getRowSize(); i++)
  {
    REQUIRE(wf->getHostWF()[i][1] == 0.0);
  }
  delete wf;
  // BAD case
  wf = new CWaveFunction(refFn, refFn);
  REQUIRE(wf->copyFromGPU() == -1);
  delete wf;
}

TEST_CASE("Test subtraction of two wave functions", "[WF]"){
  // Both files are good
  CWaveFunction wf1("src/unittest/sinbunchwf_real.csv", "src/unittest/sinbunchwf_imag.csv");
  CWaveFunction wf2(wf1);
  SECTION("Good case : straight")
  {
    REQUIRE(&wf2.subtract_host(wf1) == &wf2);
    for (size_t i = 0; i < wf2.getRowSize() * wf2.getColsSize(); i++)
    {
      REQUIRE(wf2.getHostWF()[i][0] == 0);
      REQUIRE(wf2.getHostWF()[i][1] == 0);
    }
  }
  SECTION("Good case : inversed")
  {
    CWaveFunction wf3(wf1.getRowSize(), wf1.getColsSize());
    REQUIRE(&wf2.subtract_host(wf3, true) == &wf2);
    for (size_t i = 0; i < wf2.getRowSize() * wf2.getColsSize(); i++)
    {
      REQUIRE(wf2.getHostWF()[i][0] == -1 * wf1.getHostWF()[i][0]);
      REQUIRE(wf2.getHostWF()[i][1] == -1 * wf1.getHostWF()[i][1]);
    }
  }
}

TEST_CASE("L2 norm squared", "[WF]"){
  size_t Nx = 1024;
  size_t Ny = 512;
  CWaveFunction wf(Ny, Nx);
  for (size_t i = 0; i < Nx * Ny; i++)
  {
    wf.getHostWF()[i][0] = 1.0;
    wf.getHostWF()[i][1] = 1.0;
  }
  REQUIRE(wf.norm2sqr() == 2 * Nx * Ny);
}

TEST_CASE("Max abs value", "[WF]"){
  size_t Nx = 64;
  size_t Ny = 32;
  CWaveFunction wf(Ny, Nx);
  REQUIRE(wf.maxAbs() == 0.0);
  for (size_t i = 0; i < Nx * Ny; i++)
  {
    wf.getHostWF()[i][0] = i;
    wf.getHostWF()[i][1] = i;
  }
  REQUIRE(wf.maxAbs() == Approx(sqrt(2.0)*(Nx * Ny - 1)));
}

TEST_CASE("Multiplication of the wave function by a scalar", "[WF]"){
  CWaveFunction wf("src/unittest/sinbunchwf_real.csv", "src/unittest/sinbunchwf_imag.csv");
  CWaveFunction wf2(wf);
  wf*=(2.0);
  for (size_t i = 0; i < wf.getRowSize() * wf.getColsSize(); i++)
  {
    REQUIRE(wf.getHostWF()[i][0] == 2.0 * wf2.getHostWF()[i][0]);
    REQUIRE(wf.getHostWF()[i][1] == 2.0 * wf2.getHostWF()[i][1]);
  }
}

TEST_CASE("TEST FFT/IFFT", "[WF]"){
  LOG(INFO) << "======Test FFT/IFFT change=====";
  double eps = 0.001;
  std::string realFile("src/unittest/expVReal.csv");
  std::string imagFile("src/unittest/expVImag.csv");
  CWaveFunction wf(realFile, imagFile);
  CWaveFunction wf2(wf);
  // Load reference
  CWaveFunction referenceWF("src/unittest/expVFFTReal.csv","src/unittest/expVFFTImag.csv");
  wf.fft();
  for (size_t i = 0; i < wf.getRowSize() * wf.getColsSize(); i++)
  {
    REQUIRE(wf.getHostWF()[i][0] * wf.getColsSize() == Approx(referenceWF.getHostWF()[i][0]).margin(eps));
    REQUIRE(wf.getHostWF()[i][1] * wf.getColsSize() == Approx(referenceWF.getHostWF()[i][1]).margin(eps));
  }
  wf.ifft();
  for (size_t i = 0; i < wf.getRowSize() * wf.getColsSize(); i++)
  {
    REQUIRE(wf.getHostWF()[i][0] == Approx(wf2.getHostWF()[i][0]).margin(eps));
    REQUIRE(wf.getHostWF()[i][1] == Approx(wf2.getHostWF()[i][1]).margin(eps));
  }
}

TEST_CASE("Test wave function domain change", "[WF]"){
  LOG(INFO) << "======Test domain change=====";
  CWaveFunction wf("src/unittest/sinbunchwf_real.csv", "src/unittest/sinbunchwf_imag.csv");
  REQUIRE(wf.getColsSize() != 0);
  REQUIRE(wf.getRowSize() != 0);
  CWaveFunction wf2(wf);
  CWaveFunction wf3(wf);
  // wf and wf2 and wf3 are in space domain
  REQUIRE(wf.getDomain());
  REQUIRE(wf2.getDomain());
  REQUIRE(wf3.getDomain());
  // switch wf and wf2 to freq domain
  wf.switchDomain();
  REQUIRE(wf.getDomain() == false);
  wf2.fft();
  REQUIRE(wf2.getDomain() == false);
  for (size_t i = 0; i < wf.getColsSize()*wf.getRowSize(); i++)
  {
    REQUIRE(wf.getHostWF()[i][0] == wf2.getHostWF()[i][0]);
    REQUIRE(wf.getHostWF()[i][1] == wf2.getHostWF()[i][1]);
  }
  // switch wfs back to space domain
  wf.switchDomain();
  REQUIRE(wf.getDomain() == true);
  wf2.ifft();
  REQUIRE(wf2.getDomain() == true);
  // Check all wfs are equal now
  for (size_t i = 0; i < wf.getColsSize()*wf.getRowSize(); i++)
  {
    REQUIRE(wf.getHostWF()[i][0] == wf2.getHostWF()[i][0]);
    REQUIRE(wf.getHostWF()[i][1] == wf2.getHostWF()[i][1]);
    REQUIRE(wf.getHostWF()[i][0] == Approx(wf3.getHostWF()[i][0]).margin(0.001));
    REQUIRE(wf.getHostWF()[i][1] == Approx(wf3.getHostWF()[i][1]).margin(0.001));
  }
}
