#define _USE_MATH_DEFINES
#include <cmath>
#include <gsl/gsl_sf_hyperg.h>
#include "wavefunction.h"
#include "mxutils.h"
#include "easyloggingcpp/easylogging++.h"

const double pi = M_PI;

double gaussianCOM(double x1, double x2, double x0, double omega)
{
  double R = sqrt(0.5)*(x1+x2);
  return pow(omega/pi, 0.25)  * exp(-omega * pow((x1 - x0), 2) / (double)2.0)* exp(-omega * pow((x2 - x0), 2) / (double)2.0);
}

//! Not normalized GS of the relative coordinates
double groundStateRel(double x1, double x2, double E, double omega)
{
  double mnu = 1.0 / 4.0 - E / 2.0;
  double r = sqrt(0.5)*(x1-x2);
  //cout << mnu << "," << r << ":" << gsl_sf_hyperg_U( mnu, 0.5, r*r)<< endl;
  return /*exp(-omega * pow(r, 2)/(double)2.0) /** gamma(mnu) * */  gsl_sf_hyperg_U( mnu, 0.5, r*r);
}
/*!This gives the analytical solution for the ground state of 2 bosons in a (normalized) harmonic trap running from -xmax to xmax, with a minimum located at x0. The wavefunction is normalized to 1. The value En gives the energy of the state.
Calculated accodring to Busch, T., Englert, B.-G., Rzążewski, K. & Wilkens, M. Two Cold Atoms in a Harmonic Trap. Found. Phys 28, 549–559 (1998)
* Input:
* int N : Number of points to discretize the coordinate space
* double En : two-particle ground state energy. En=1 is the non-interacting case, En=2 is the TG case
* double x0 : center of the trap coordinate
* double xmax : the coordinate space will be simulated from -xmax to xmax
* double omega : the frequency of the harmonic trap
* Return:
* pointer to the stored wave function
* NOTE: use delete [] arr; after you are done using it
**/
double * calcInitialPsi(size_t N, double En, double x0, double xmax, double omega)
{
  if (N <= 0)
    return NULL;
  En = En - 0.5;
  // Calculate the interaction strength
  double g = -2 * sqrt(2.0) * gamma(0.75 - En / 2) / gamma(0.25 - En / 2);
  // Wave function container on the host
  double * h_wf = new double[N*N];
  double * h_x = new double[N];
  double dx = 2.0 * xmax / (N );
  double curX = -xmax+dx;
  //cout << "Initialized" << endl;
  // Initialize the x vector
  for (int i = 0; i < N; i++)
  {
    h_x[i] = curX;
    curX += dx;
  }
  double wfInt = 0;
  for (int iX = 0; iX < N; iX++)
    for (int iY = 0; iY < N; iY++)
    {
      // Relative coordinate
      double wfRel = groundStateRel(h_x[iX], h_x[iY], En, omega)*gaussianCOM(h_x[iX], h_x[iY], x0, omega);
      wfInt += abs(wfRel)*abs(wfRel);
      h_wf[iY * N + iX] = wfRel;
    }
  for (int iX = 0; iX < N; iX++)
    for (int iY = 0; iY < N; iY++)
    {
      // Divide by the integral
      h_wf[iY * N + iX] /= (double) sqrt(wfInt * dx * dx);
    }
  //cout << "Cleaning up " << endl;
  delete [] h_x;
  return h_wf;
}

//! Default constructor creates a busch state with En=1.7
/*! Create the wavefunction as a ground state of the model  by Busch, T., Englert, B.-G., Rzążewski, K. & Wilkens, M. Two Cold Atoms in a Harmonic Trap. Found. Phys 28, 549–559 (1998) for En = 1.7;
  \param N : the number of points for the coordinate dimension. The resolution is then dx = xmax/N
  \param xmax : The coordinate space will be simulated from -xmax to xmax
  \param En : 2-particle ground state energy unitless
*/
CWaveFunction::CWaveFunction(size_t N, double xmax, double En){
  double x0 = 0.0;
  double omega = 1.0;
  if (N <= 0 )
  {
    LOG(ERROR) << __func__ << "The number of points N has to be positive integer\n";
    return;
  }
  if (xmax <= 0.0)
  {
    LOG(ERROR) << __func__ << "The space size xmax has to be positive\n";
    return;
  }
  this->h_complexWF = new fftw_complex [N * N];
  memset((double *)this->h_complexWF, 0, 2*N * N * sizeof(double));
  double * real = calcInitialPsi(N, En, x0, xmax, omega);
  if (real == NULL)
  {
    LOG(ERROR) <<__func__ << " : Couldn't generate an initial wave function as a Busch state\n";
    return;
  }
  for (size_t i = 0; i < N*N; i++)
  {
    this->h_complexWF[i][0] = real[i];
  }
  delete [] real;
  this->szCols = this->szRows = N;
  this->pitch = 0;
  this->d_complexWF = NULL;
  this->domain = true;
  this->plan = fftw_plan_dft_2d(this->szRows, this->szCols, this->h_complexWF, this->h_complexWF, FFTW_FORWARD, FFTW_MEASURE);
  this->planInv = fftw_plan_dft_2d(this->szRows, this->szCols, this->h_complexWF, this->h_complexWF, FFTW_BACKWARD, FFTW_MEASURE);
  this->plan_init = true;
  return;
}

//! Load wavefunction from file
double * CWaveFunction::loadWF(std::string fn, size_t & wSzRow, size_t & wSzCol){
  size_t szX;
  size_t szY;
  if (fn.empty())
  {
    wSzRow = wSzCol = 0;
    return NULL;
  }
  double * ptr = parseMxFromCSV(fn, szX, szY);
  if (ptr == NULL){
    LOG(INFO) <<__func__ << " : Couldn't load the wavefuction from file " << fn <<std::endl;
    wSzCol = wSzRow = 0;
    return NULL;
  }
  wSzCol = szX;
  wSzRow = szY;
  return ptr;
}

//! Constructor which reads a wf from the files
/*!
  Read both real and imaginary parts of the wavefunction from the files
  \param fileReal : path to the real part
  \param fileImag : path to the imaginary part
*/
CWaveFunction::CWaveFunction(std::string fileReal, std::string fileImag){
  size_t Nrr = 0;
  size_t Nrc = 0;
  size_t Nir = 0;
  size_t Nic = 0;
  double * real = this->loadWF(fileReal, Nrr, Nrc);
  double * imag = this->loadWF(fileImag, Nir, Nic);
  if (Nrr == 0 || Nir == 0 || Nrc == 0 || Nic == 0)
  {
    LOG(ERROR) << __func__ << " Imaginary and real parts of the wave function should have positive dimensions\n";
    this->clear();
    return;
  }
  if (Nrr != Nir || Nrc != Nic)
  {
    LOG(ERROR) << __func__ << " Imaginary and real parts of the wave function should have the same dimensions\n";
    this->clear();
    return;
  }
  if (real == NULL && imag == NULL)
  {
    LOG(ERROR) << __func__ << " Both input files seem to be invalid\n";
    this->clear();
    return;
  }
  this->h_complexWF = new fftw_complex[Nrr * Nrc];
  this->szRows = Nrr;
  this->szCols = Nrc;
  this->pitch = 0;
  this->d_complexWF = NULL;
  this->plan = fftw_plan_dft_2d(this->szRows, this->szCols, this->h_complexWF, this->h_complexWF, FFTW_FORWARD, FFTW_MEASURE);
  this->planInv = fftw_plan_dft_2d(this->szRows, this->szCols, this->h_complexWF, this->h_complexWF, FFTW_BACKWARD, FFTW_MEASURE);
  this->plan_init = true;
  for (size_t i = 0; i < Nrr * Nrc; i++){
    this->h_complexWF[i][0] = (real == NULL ? 0.0 : real[i]);
    this->h_complexWF[i][1] = (imag == NULL ? 0.0 : imag[i]);
  }
  delete [] real;
  delete [] imag;
  return;
}

//! clean up
void CWaveFunction::clear() {
  if (this->plan_init)
  {
    fftw_destroy_plan(this->plan);
    fftw_destroy_plan(this->planInv);
  }
  if (this->h_complexWF != NULL)
    delete [] this->h_complexWF;
  this->pitch = 0;
  // Free cuda memory
  cudaError_t err = cudaFree(this->d_complexWF);
  this->d_complexWF = NULL;
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << ": Could not free cuda memory : " << cudaGetErrorString(err) << "\n";
  }
}

//! Destructor: cleans up all the memory on both device and host
CWaveFunction::~CWaveFunction(){
  this->clear();
}


//! Copy the wave function to GPU
/*!
  Copied the wave function to GPU using cudaMemcpy2D
  The size is the same as the original wave function size (sx x sx), aligned to warp with the pitch which is set in the process.
  \return 0 if all goes well and -1 otherwise
  NOTE: DO NOT free the device pointer! Use wf.clear() instead to fully reset the wf
*/
int CWaveFunction::copyToGPU(){
  // Check if the host wave function is legit
  if (this->h_complexWF == NULL || this->szCols == 0 || this->szRows == 0)
  {
    LOG(ERROR) << __func__ << ": The wave function must not be empty\n";
    return -1;
  }
  // Allocate memory on GPU
  if (this->d_complexWF == NULL)
  {
    cudaError_t err = cudaMallocPitch(&this->d_complexWF, &this->pitch, sizeof(cufftDoubleComplex) * this->getColsSize(), this->getRowSize());
    if (err != cudaSuccess)
    {
      LOG(ERROR) << __func__ << ": Cannot allocate memory on the device:" << cudaGetErrorString(err) << "\n";
      return -1;
    }
  }
  // Copy the wave function
  if (copyToCudaComplex(this->d_complexWF, this->pitch, this->h_complexWF, this->getRowSize(), this->getColsSize()) != 0)
  {
    LOG(ERROR) << __func__ << ": Wave function copy failed\n";
    return -1;
  }
  // Everything went well
  return 0;
}

//! Copy GPU wave function back into host wave function
/*!
*/
int CWaveFunction::copyFromGPU(void){
  if (this->szRows == 0 || this->szCols == 0)
  {
    LOG(ERROR) << __func__ << "The dimensions of the wave function has to be non-zero\n";
    return -1;
  }
  if (this->d_complexWF == NULL || this->pitch == 0)
  {
    LOG(ERROR) << __func__ << " : GPU doesn't have the wavefunction\n";
    return -1;
  }
  if (this->h_complexWF == NULL)
    this->h_complexWF = new fftw_complex[this->getColsSize() * this->getRowSize()];
  // Copy memory while accounting for the pitch
  cudaError_t cudaStat = cudaMemcpy2D (this->h_complexWF, this->getColsSize() * sizeof(this->h_complexWF[0]), this->d_complexWF, this->pitch, sizeof(this->h_complexWF[0])*this->getColsSize(), this->getRowSize(), cudaMemcpyDeviceToHost);
  if (cudaStat != cudaSuccess)
  {
   LOG(ERROR) << __func__ << " : cudaMemcpy2D failed. CUDA: " << cudaGetErrorString(cudaStat) << "\n";
   return -1;
  }
  return 0;
}


/*!
  Allocate a vacuum wave function of size Nx x Ny
  \param Nc : number of columns
  \param Nr : number of rows
  \return 0 if success and -1 if failed; Logs errors
*/
int CWaveFunction::makeVacuum(size_t Nc, size_t Nr){
  this->clear();
  if (Nc == 0 || Nr == 0)
  {
    LOG(ERROR) << __func__ << "The dimensions have to be positive integers\n";
    return -1;
  }
  this->h_complexWF = new fftw_complex[Nc * Nr];
  this->szRows = Nr;
  this->szCols = Nc;
  this->domain = true;
  this->plan = fftw_plan_dft_2d(this->szRows, this->szCols, this->h_complexWF, this->h_complexWF, FFTW_FORWARD, FFTW_MEASURE);
  this->planInv = fftw_plan_dft_2d(this->szRows, this->szCols, this->h_complexWF, this->h_complexWF, FFTW_BACKWARD, FFTW_MEASURE);
  this->plan_init = true;
  memset((double*)this->h_complexWF, 0, sizeof(double) * Nc * Nr * 2);
  return 0;
}

CWaveFunction::CWaveFunction(size_t Nc, size_t Nr){
  this->makeVacuum(Nc, Nr);
}

/*! Default constructor*/
CWaveFunction::CWaveFunction(){
  this->szRows = 0;
  this->szCols = 0;
  this->h_complexWF = NULL;
  this->d_complexWF = NULL;
  this->pitch = 0;
  this->domain = true;
  return;
}

//! Copy the wave function
int CWaveFunction::copy(CWaveFunction & orig){
  if (this->getColsSize() != orig.getColsSize() || this->getRowSize() != orig.getRowSize())
  {
    if(this->makeVacuum(orig.getRowSize(), orig.getColsSize()) != 0)
      return -1;
  }
  if (orig.h_complexWF != NULL)
    memcpy(this->h_complexWF, orig.h_complexWF, this->getRowSize() * this->getColsSize() * sizeof(fftw_complex));
  this->domain = orig.domain;
  return 0;
}

//! Copy constructor
CWaveFunction::CWaveFunction(CWaveFunction & orig){
  if (this->copy(orig) != 0)
    LOG(ERROR) << __func__ << " : cannot dubplicate the wave function\n";
}

CWaveFunction & CWaveFunction::operator+=(const CWaveFunction & right){
  if (this->getRowSize() != right.getRowSize() || this->getColsSize() != right.getColsSize())
  {
    LOG(ERROR) << __func__ << " : The two wavefunctions must have the same dimensions.\n";
    return *this;
  }
  for (size_t i = 0; i < this->getRowSize() * this->getColsSize(); i++)
  {
    this->h_complexWF[i][0] += right.h_complexWF[i][0];
    this->h_complexWF[i][1] += right.h_complexWF[i][1];
  }
  return *this;
}

double CWaveFunction::maxAbs(size_t maxMem, int maxThreads){
  if (this->h_complexWF == NULL)
  {
    LOG(ERROR) << __func__ << " : The wave function must not be empty.\n";
    return -1.0;
  }
  double abs2 = 0.0;
  size_t N = this->getRowSize() * this->getColsSize();
  double max = 0.0;
  for (size_t i = 0; i < N; i++)
  {
    abs2 = this->h_complexWF[i][0] * this->h_complexWF[i][0] + this->h_complexWF[i][1] * this->h_complexWF[i][1];
    if (max < abs2)
    {
      max = abs2;
    }
  }
  return sqrt(max);
}

CWaveFunction & CWaveFunction::subtract_host(const CWaveFunction & right, bool inversed){
  if (this->getRowSize() != right.getRowSize() || this->getColsSize() != right.getColsSize())
  {
    LOG(ERROR) << __func__ << " : The two wavefunctions must have the same dimensions: (" << this->getRowSize() << ", " << this->getColsSize() << ") vs (" << right.getRowSize() << ", " << right.getColsSize() << ")";
    return *this;
  }
  for (size_t i = 0; i < this->getRowSize() * this->getColsSize(); i++)
  {
    if (!inversed)
    {
      this->h_complexWF[i][0] -= right.h_complexWF[i][0];
      this->h_complexWF[i][1] -= right.h_complexWF[i][1];
    }
    else
    {
      this->h_complexWF[i][0] = right.h_complexWF[i][0] - this->h_complexWF[i][0];
      this->h_complexWF[i][1] = right.h_complexWF[i][1] - this->h_complexWF[i][1];
    }
  }
  return *this;
}

CWaveFunction & CWaveFunction::operator*=(double c){
  for (size_t i = 0; i < this->getRowSize() * this->getColsSize(); i++)
  {
    this->h_complexWF[i][0] *= c;
    this->h_complexWF[i][1] *= c;
  }
  return *this;
}

/*!
  L2 norm squared of the wave function.
  \param maxMem : maximum free memory on the device (in bytes). default 64 mb
  \param maxThreads : maximum threads per block on the device. default is 1024
  TODO: Change to use thrust and split vector into chunks
*/
double CWaveFunction::norm2sqr(size_t maxMem, int maxThreads){
  if (this->h_complexWF == NULL)
  {
    LOG(ERROR) << __func__ << " : The vector must not be empty.\n";
    return -1.0;
  }
  size_t N = this->getRowSize() * this->getColsSize();
  double norm2 = 0.0;
  for (size_t i = 0; i < N; i++)
  {
    norm2 += this->h_complexWF[i][0] * this->h_complexWF[i][0] + this->h_complexWF[i][1] * this->h_complexWF[i][1];
  }
  return norm2;
}
