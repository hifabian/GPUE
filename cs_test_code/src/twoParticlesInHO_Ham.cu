#include <stdio.h>
#include "twoParticlesInHO_Ham.h"
//! Get time dep potential at time T and coordinate d_x
__device__ double d_timeDepPotential(const double d_x, const double T, const void * d_params){
  if (d_params == NULL)
    return 0;
  double * d_dparams = (double *) d_params;
  double xmax = d_dparams[0];
  double x0 = d_dparams[1];
  double omega = d_dparams[2];
  if (d_x > xmax)
    return 0;
  return omega * (d_x - x0) * (d_x - x0);
}

//! Get interaction g* delta(d_r) at time T and |x1-x2| = d_r
__device__ double d_timeDepInteraction(const double d_r, const double T, const void * d_params){
  if (d_params == NULL)
    return 0;
  double * d_dparams = (double *) d_params;
  double gdx = d_dparams[3];
  return (d_r == 0) * gdx;
}

__global__ void dummyKernel(){
  printf("dummy\n");
}

//! Non-linear term is zero
__device__ double d_timeDepNonLin(const double absPsi_x, const double T, const void * d_params)
{
  return 0;
}


//! Fill in the device function pointers
__global__ void fillDevicePtrs(hamFunc * d_funcPot, hamFunc * d_funcInter, hamFunc * d_funcNonl){
  d_funcPot[0] = &d_timeDepPotential;
  d_funcInter[0] = &d_timeDepInteraction;
  d_funcNonl[0] = &d_timeDepNonLin;
}

//! Create momentum and coordinate vectors on both device and host
int C2ParticlesHO::initializeVectors(size_t Nx, double xmax, double x0){
  if (this->d_x != NULL)
    cudaFree(this->d_x);
  if (this->h_x != NULL)
    delete [] this->h_x;
  if (this->d_px != NULL)
    cudaFree(this->d_px);
  if (this->h_px != NULL)
    delete [] this->h_px;
  this->sz = Nx;
  if (this->sz <= 0)
  {
    LOG(ERROR) << __func__ << ": The resolution must be non-zero\n";
    return -1;
  }
  if(C2ParticlesHO::fftdef(this->sz, xmax, this->h_x, this->h_px, this->dx, this->dpx) != 0)
  {
    LOG(ERROR) << __func__ << ": Could not generate coordinate and momentum vectors\n";
    return -1;
  }
  // Copy to device
  this->d_x = allocateAndCopyToGPU<double>(this->h_x, this->sz);
  this->d_px = allocateAndCopyToGPU<double>(this->h_px, this->sz);
  if (this->d_x == NULL || this->d_px == NULL)
  {
    LOG(ERROR) << __func__ << ": Could not copy to GPU\n";
    return -1;
  }
  return 0;
}

//! Constructor
/*!
\param Nx : resolution
\param xmax : x is in (-xmax,xmax]
\param x0 : center of the harmonic oscillator
\param omega : frequency of the HO
\param g : discretized interactions strength = g
*/
C2ParticlesHO::C2ParticlesHO(size_t Nx, double g, double xmax, double x0, double omega){
  //std::cout << "Construct C2ParticlesHO\n ";
  if (this->initializeVectors(Nx, xmax, x0) != 0)
  {
    LOG(ERROR) << __func__ << ": Could not Initialize momentum and coordinate vectors\n";
    return;
  }
  double params[4] = {0,0,0,0};
  params[0] = xmax;
  params[1] = x0;
  params[2] = omega;
  params[3] = g / this->dx;
  this->d_params = allocateAndCopyToGPU<double>(params, 4);
  if( this->d_params == NULL)
  {
    LOG(ERROR) << __func__ << ": Could not allocate memory for the constants on gpu\n";
    return;
  }
  cudaError_t err = cudaMalloc(&this->d_Pot, sizeof(hamFunc) * 1);
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << ": Could not allocate memory for the device Potential functions on gpu\n";
    return;
  }
  err = cudaMalloc(&this->d_Inter, sizeof(hamFunc) * 1);
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << ": Could not allocate memory for the device interaction functions on gpu\n";
    return;
  }
  err = cudaMalloc(&this->d_Nonl, sizeof(hamFunc) * 1);
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << ": Could not allocate memory for the device non-linear functions on gpu\n";
    return;
  }
  fillDevicePtrs<<<1,1>>>(this->d_Pot, this->d_Inter, this->d_Nonl);
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << ": fillDevicePtrs Kernel execution failed : " << cudaGetErrorString(err) << "\n";
    return;
  }
  std::cout << "Initialized the function pointers\n";
}

//! destructor
C2ParticlesHO::~C2ParticlesHO(){
  //std::cout << "Destroy C2ParticlesHO\n";
  if (this->d_params != NULL)
    cudaFree(this->d_params);
  if (this->d_x != NULL)
    cudaFree(this->d_x);
  if (this->h_x != NULL)
    delete [] this->h_x;
  if (this->d_px != NULL)
    cudaFree(this->d_px);
  if (this->h_px != NULL)
    delete [] this->h_px;
  if (this->d_Pot != NULL)
    cudaFree(this->d_Pot);
  if (this->d_Inter != NULL)
    cudaFree(this->d_Inter);
  if (this->d_Nonl != NULL)
    cudaFree(this->d_Nonl);
  this->d_params = NULL;
  this->d_x = NULL;
  this->h_x = NULL;
  this->d_px = NULL;
  this->h_px = NULL;
  this->d_Pot = NULL;
  this->d_Inter = NULL;
  this->d_Nonl = NULL;
}

/*!
  Define coordinate and momentum axes, and their resolution for future FFT steps
  NOTE: clean up with delete [] h_x; delete [] h_px
*/
int C2ParticlesHO::fftdef(size_t Nx, double xmax, double * &h_x, double * &h_px, double &dx, double &dpx){
  if (Nx <= 0)
  {
    LOG(INFO) << __func__ << " : The number of discretization points must be positive\n";
    return -1;
  }
  if (xmax <= 0)
  {
    LOG(INFO) << __func__ << " : The size of the space has to be positive\n";
    return -1;
  }
  h_x = new double[Nx];
  h_px = new double[Nx];
  // max momentum
  double pxmax = M_PI * Nx / (2.0 * xmax);
  dx = 2.0 * xmax / (double)Nx;
  dpx = 2.0 * pxmax / (double)Nx;
  for (size_t i = 0; i < Nx; i++)
  {
    h_x[i] = (i + 1) * dx - xmax;
    // reordination needed for the fourier transform
    if (i <= (size_t) (Nx / 2))
      h_px[i] = ((size_t)(Nx / 2) + i) * dpx - pxmax;
    else
      h_px[i] = (i - (size_t)(Nx / 2)) * dpx - pxmax;
  }
  return 0;
}

hamFunc * C2ParticlesHO::timeDepPotential(void){
  return this->d_Pot;
}

/*!  |x1-x2|=d_r
\return pointer to a device function U_int(|x1-x2|)
*/
hamFunc * C2ParticlesHO::timeDepInteraction(void){
  return this->d_Inter;
}

/*!
Time dep Non-linear part of the Hamiltonian prop to |Psi(x)|
\return pointer to a device function V_nonlin(|Psi(x)|)
*/
hamFunc * C2ParticlesHO::timeDepNonLin(void){
  return this->d_Nonl;
}
