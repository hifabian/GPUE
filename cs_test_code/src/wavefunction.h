#ifndef DEFCWF
#define DEFCWF
#include <string>
#include <cufft.h>
#include <complex.h>
#include <fftw3.h>
#include "hamiltonian.h"
//! Contains all wavefunction handling functions, including TE
/**
* Class CWaveFunction contains tools for wavefunction generation, time evolution, saving and loading
**/
class CWaveFunction {
protected:
  //! wavefunction on the CPU
  fftw_complex * h_complexWF = NULL;
  //! wavefunction on GPU
  cufftDoubleComplex * d_complexWF = NULL;
  //! GPU memory pitch
  size_t pitch = 0;
private:
  //! plan is Initialized
  bool plan_init = false;
  //! Clean up
  void clear(void);
  //! Load wavefunction from file
  double * loadWF(std::string fn, size_t & wSzRow, size_t & wSzCol);
  //! Create empty wv
  int makeVacuum(size_t Ny, size_t Nx);
  //! Col size
  size_t szCols = 0;
  //! Row size
  size_t szRows = 0;
  //! Domain: true if space, false if frequency
  bool domain = true;
  //! FFTW plans
  fftw_plan plan;
  fftw_plan planInv;
public:
  //! Toggle domain
  void toggleDomain(void){
    if (this->domain == true)
      this->domain = false;
    else this->domain = true;
  }
  //! Set domain
  void setDomain(bool d){
    this->domain = d;
  }
  //! perform normalized fft on the wf
  void fft(void){
    fftw_execute(this->plan);
    (*this) *= 1.0 / sqrt(this->szCols * this->szRows);
    this->domain = false;
  }
  //! perform normalized ifft on the wf
  void ifft(void){
    fftw_execute(this->planInv);
    (*this) *= 1.0 / sqrt(this->szCols * this->szRows);
    this->domain = true;
  }
  //! Perform normalized fft if the domain is space, and ifft if the domain is frequency
  void switchDomain(void){
    fftw_execute((this->domain ? this->plan : this->planInv));
    (*this) *= 1.0 / sqrt(this->szCols * this->szRows);
    this->toggleDomain();
  }
  //! Get current domain (true if space, false if freq)
  bool getDomain(void){
    return this->domain;
  }
  //! Get number of rows in the wave function
  size_t getRowSize(void) const {
    return this->szRows;
  }
  //! Get number of columns in the wave function
  size_t getColsSize(void) const {
    return this->szCols;
  }
  //! Copy wave function
  int copy(CWaveFunction & orig);
  //! Get pitch
  size_t getPitch(void)
  {
    return this->pitch;
  }
  //! Get host real part
  fftw_complex * getHostWF(void)
  {
    return this->h_complexWF;
  }
  //! Get cuda wf
  cufftDoubleComplex * getDeviceWF(void)
  {
    return this->d_complexWF;
  }
  //! Default constructor
  CWaveFunction(void);
  //! This constructor creates a busch state with En=1.7
  /*! Create the wavefunction as a ground state of the model  by Busch, T., Englert, B.-G., Rzążewski, K. & Wilkens, M. Two Cold Atoms in a Harmonic Trap. Found. Phys 28, 549–559 (1998) for En = 1.7;
    \param N : the number of points for the coordinate dimension. The resolution is then dx = xmax/N
    \param xmax : The coordinate space will be simulated from -xmax to xmax
  */
  CWaveFunction(size_t N, double xmax = 5.0, double En=1.7);
  CWaveFunction(size_t Nc, size_t Nr);
  //! Constructor which reads a wf from the files
  /*!
    Read both real and imaginary parts of the wavefunction from the files
    \param fileReal : path to the real part
    \param fileImag : path to the imaginary part
  */
  CWaveFunction(std::string fileReal, std::string fileImag);

  //! Copy constructor
  CWaveFunction(CWaveFunction & orig);

  //! Destructor: cleans up all the memory on both device and host
  ~CWaveFunction(void);

  //! Copy the wave function to GPU
  /*!
    Copied the wave function to GPU using cudaMemcpy2D
    \return The device pointer to GPU data. The size is the same as the original wave function size (sx x sx), aligned to warp with the pitch which is set in the process.
  */
  int copyToGPU(void);

  //! Copy GPU wave function back into host wave function
  /*!
  */
  int copyFromGPU(void);

  //! Subtract two wave functions allocated on the host. If inversed is true, compute f2 = f1 - f2, else f2=f2-f1
  CWaveFunction & subtract_host(const CWaveFunction & right, bool inversed = false);

  //! Add two wave functions allocated on the host: f1 = f1+f2
  CWaveFunction & operator+=(const CWaveFunction & right);

  CWaveFunction & operator*=(double c);

  /*!
    L2 norm squared of the wave function.
    \param maxMem : maximum free memory on the device (in bytes). default 64 mb
    \param maxThreads : maximum threads per block on the device. default is 1024
  */
  double norm2sqr(size_t maxMem = 67108864, int maxThreads=1024);
  /*!
    Maximum absolute value
    \param maxMem : maximum free memory on the device (in bytes). default 64 mb
    \param maxThreads : maximum threads per block on the device. default is 1024
  */
  double maxAbs(size_t maxMem = 67108864, int maxThreads=1024);
};
double * calcInitialPsi(size_t N, double En, double x0, double xmax, double omega);
#endif
