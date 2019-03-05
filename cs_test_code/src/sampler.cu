#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <algorithm>
#include "sampler.h"
#include "wavefunction.h"
#include "mxutils.h"
#include "cs_utils.h"
/*!
Sample the wave function wfin and save the result into  wfout
sampling method: random row {i} and column {j} numbers, sampled elements are {f_ij}
\param wfin : 2D wave function (NxN) to sample from. N has to be greater than or equal to the sampling range
\param wfout : 2D wave function (szRows x szCols) to save the result into. Must have the space allocated
\return 0 if all goes well and -1 otherwise
*/
int CRowColSampler::sample(CWaveFunction * wfin, CWaveFunction * wfout){
  if (wfin == NULL || wfout == NULL)
  {
    LOG(ERROR) << __func__ << "Input and output wave function pointers cannot be NULL\n";
    return -1;
  }
  if (this->range > wfin->getRowSize() || this->range > wfin->getColsSize())
  {
    LOG(ERROR) << __func__ << " : Both dimensions of the input wave function ("<< wfin->getRowSize() <<"," << wfin->getColsSize() <<") have to have a range (" << this->range <<") which fits into the compressed wave function. \n";
    return -1;
  }
  if (wfout->getRowSize() < this->szRows || wfout->getColsSize() < this->szCols)
  {
    LOG(ERROR) << __func__ << " : Both dimensions of the wave function ("<< wfout->getRowSize() <<"," << wfout->getColsSize() <<") have to have a range (" << this->range <<") which can fit the compressed wave function. \n";
    return -1;
  }
  if (this->h_randomRows == NULL || this->h_randomCols == NULL)
  {
    LOG(ERROR) << __func__ << "Sampling vectors cannot be NULL\n";
    return -1;
  }
  // Sample the wave function
  if (sample_host(wfin->getHostWF(), wfin->getRowSize(), wfin->getColsSize(), this->h_randomRows, this->szRows, this->h_randomCols, this->szCols, wfout->getHostWF()) != 0)
  {
    LOG(ERROR) << __func__ << "Could not sample the wave function\n";
    return -1;
  }
  wfout->setDomain(true);
  return 0;
}

/*! Upsample the compressed wave function according to
  x = Transpose(A) y
  \param wfin : pointer to the compressed wavefunction. Its dimensions (szR x szC) must be the same as the sampler dimensions
  \param wfout : pointer to the wave function where we are supposed to store the result. Its dimesions (N x N) must fit the upsampled data (N >= range)
  \return 0 if all goes well and  -1 otherwise
*/
int CRowColSampler::upsample(CWaveFunction * wfin, CWaveFunction * wfout){
  if (wfin == NULL || wfout == NULL)
  {
    LOG(ERROR) << __func__ << "Input and output wave function pointers cannot be NULL\n";
    return -1;
  }
  if (this->range > wfout->getRowSize() || this->range > wfout->getColsSize())
  {
    LOG(ERROR) << __func__ << " : Both dimensions of the output wave function ("<< wfout->getRowSize() <<"," << wfout->getColsSize() <<") have to greater or equal than the sampling range (" << this->range <<").\n";
    return -1;
  }
  if (wfin->getRowSize() != this->szRows || wfin->getColsSize() != this->szCols)
  {
    LOG(ERROR) << __func__ << " : Both dimensions of the compressed wave function ("<< wfin->getRowSize() <<"," << wfin->getColsSize() <<") have to be the same as the sampling dimensions (" << this->szRows <<"," << this->szCols << ").\n";
    return -1;
  }
  memset((double *)wfout->getHostWF(), 0, 2* sizeof(double) * wfout->getRowSize() * wfout->getColsSize());
  if (restore_host(wfin->getHostWF(), this->h_randomRows, this->szRows, this->h_randomCols, this->szCols, wfout->getHostWF(), wfout->getRowSize(), wfout->getColsSize()) != 0){
    LOG(ERROR) << __func__ << "Could not upsample the wave function\n";
    return -1;
  }
  wfout->setDomain(true);
  return 0;
}

int CRowColSampler::randomize(){
  // Permute the elements of the row vector
  if (permuteElements(this->d_randomRows, this->range, this->maxThreads) != 0)
  {
    LOG(ERROR) << __func__ << " : failed to permute the rows array. \n";
    return -1;
  }
  // Permute the elements of the column vector
  if (permuteElements(this->d_randomCols, this->range, this->maxThreads) != 0)
  {
    LOG(ERROR) << __func__ << " : failed to permute the column array. \n";
    return -1;
  }
  if (this->sortSampler() != 0)
  {
    LOG(ERROR) << __func__ << " : failed to sort the sampling vectors. \n";
    return -1;
  }
  return 0;
}
// Constructor
CRowColSampler::CRowColSampler(size_t szR, size_t szC, size_t range, int maxThreads){
  if (szR == 0 || szC == 0 || range == 0)
  {
    LOG(ERROR) <<  __func__ << " : The sampling sizes and the range have to be greater than zero\n";
    return;
  }
  this->range = range;
  this->szRows = szR;
  this->szCols = szC;
  this->maxThreads = maxThreads;
  // Allocate row indices vector
  cudaError_t err = cudaMalloc(&this->d_randomRows, sizeof(size_t) * this->range);
  if (err != cudaSuccess){
    LOG(ERROR) << __func__ << " : failed to allocate memory for the row permutations on the GPU. " << cudaGetErrorString(err) << "\n";
    return;
  }
  // Initialize row indices to [0,..,range-1]
  if (fillRange(this->d_randomRows, this->range, this->maxThreads) != 0)
  {
    LOG(ERROR) << __func__ << " : failed to fill the rows array. \n";
    return;
  }
  // Permute the elements of the row vector
  if (permuteElements(this->d_randomRows, this->range, this->maxThreads) != 0){
    LOG(ERROR) << __func__ << " : failed to permute the rows array. \n";
    return;
  }
  // Allocate column indices vector
  err = cudaMalloc(&this->d_randomCols, sizeof(size_t) * this->range);
  if (err != cudaSuccess){
    LOG(ERROR) << __func__ << " : failed to allocate memory for the row permutations on the GPU. " << cudaGetErrorString(err) << "\n";
    return;
  }
  // Initialize column indices to [0,...,range-1]
  if (fillRange(this->d_randomCols, this->range, this->maxThreads) != 0)
  {
    LOG(ERROR) << __func__ << " : failed to fill the columns array. \n";
    return;
  }
  if (permuteElements(this->d_randomCols, this->range, this->maxThreads) != 0){
    LOG(ERROR) << __func__ << " : failed to permute the columns array. \n";
    return;
  }
  // Allocate host random rows and column arrays
  this->h_randomCols = new size_t[this->range];
  this->h_randomRows = new size_t[this->range];
  if (copyArrFromGPU<size_t>(this->h_randomCols, this->d_randomCols, this->range) != 0)
  {
    LOG(ERROR) << __func__ << " : failed to fill the host columns array. \n";
    return;
  }
  if (copyArrFromGPU<size_t>(this->h_randomRows, this->d_randomRows, this->range) != 0)
  {
    LOG(ERROR) << __func__ << " : failed to fill the host rows array. \n";
    return;
  }
  this->sortSampler();
}
// Destructor
CRowColSampler::~CRowColSampler(){
  delete [] this->h_randomRows;
  delete [] this->h_randomCols;
  cudaFree(this->d_randomRows);
  cudaFree(this->d_randomCols);
}

/*! Sort first szCols (szRows) elements of the random column (row) vector
\return 0 if all goes well and -1 otherwise
*/
int CRowColSampler::sortSampler(){
  if (this->d_randomCols != NULL)
  {
    thrust::device_ptr<size_t> dthrust_data(this->d_randomCols);
    // Sort the columns vector
    thrust::sort(dthrust_data, dthrust_data + this->szCols);
    if (copyArrFromGPU<size_t>(this->h_randomCols, this->d_randomCols, this->szCols) != 0)
    {
      return -1;
    }
  }
  else return -1;
  if (this->d_randomRows != NULL)
  {
    thrust::device_ptr<size_t> dthrust_data(this->d_randomRows);
    // Sort the columns vector
    thrust::sort(dthrust_data, dthrust_data + this->szRows);
    if (copyArrFromGPU<size_t>(this->h_randomRows, this->d_randomRows, this->szRows) != 0)
    {
      return -1;
    }
  }
  else return -1;
  return 0;
}

/*!
Apply sampling operator f = Sample(IFFT(vec))
\param wfin : input vector in full frequency or space domain.
\param wfout : output vector in sampled space domain
\return 0 if all goes well and -1 otherwise.
*/
int CRowColSampler::applySamplingOperator(CWaveFunction * wfin, CWaveFunction * wfout){
  if (wfin == NULL || wfout == NULL)
  {
    LOG(INFO) << "The input and output wave functions must not be empty";
    return -1;
  }
  // IFFT(wfin)
  if (wfin->getDomain() == false)
  {
    wfin = new CWaveFunction(*wfin);
    wfin->switchDomain();
    if(this->sample(wfin, wfout) != 0)
    {
      LOG(INFO) << "Could not sample the wave function";
      return -1;
    }
    delete wfin;
  }
  else
  {
    if(this->sample(wfin, wfout) != 0)
    {
      LOG(INFO) << "Could not sample the wave function";
      return -1;
    }
  }
  return 0;
}

/*!
Apply inverse sampling operator vec = Upsample(FFT(f))
\param wfin : input vector in sampled space domain.
\param wfout : output vector in full frequency domain
\return 0 if all goes well and -1 otherwise.
*/
int CRowColSampler::applyInverseSamplingOperator(CWaveFunction * wfin, CWaveFunction * wfout){
  if (wfin == NULL || wfout == NULL)
  {
    LOG(INFO) << "The input and output wave functions must not be empty";
    return -1;
  }
  if (wfin->getDomain() != true)
  {
    LOG(INFO) << "The input wave function must be in space domain";
    return -1;
  }
  if (this->upsample(wfin, wfout) != 0)
  {
    LOG(INFO) << "Could not apply inverse sampling operator";
    return -1;
  }
  wfout->switchDomain();
  return 0;
}

//=====================
// CRandomSampler

// Constructor
CRandomSampler::CRandomSampler(size_t sN, size_t range, int maxThreads){
  if (sN == 0 || range == 0)
  {
    LOG(ERROR) <<  __func__ << " : The sampling size and the range have to be greater than zero\n";
    return;
  }
  this->range = range;
  this->sampleN = sN;
  this->maxThreads = maxThreads;
  // Allocate row indices vector
  this->h_randomIndices = new size_t[this->range * this->range];
  // Initialize row indices to [0,..,range-1]
  for (size_t i = 0; i < this->range * this->range; i++)
  {
    this->h_randomIndices[i] = i;
  }
  // Permute the elements of the indices vector
  if (this->randomize() != 0)
  {
    LOG(ERROR) << __func__ << " : failed to permute the indices array. \n";
  }
  return;
}
// Destructor
CRandomSampler::~CRandomSampler(){
  delete [] this->h_randomIndices;
}

/*! Sort first szCols (szRows) elements of the random column (row) vector
\return 0 if all goes well and -1 otherwise
*/
int CRandomSampler::sortSampler(){
  if (this->h_randomIndices != NULL)
  {
    std::sort(this->h_randomIndices, &this->h_randomIndices[this->sampleN-1]);
    return 0;
  }
  else return -1;
}

int CRandomSampler::randomize(){
  // Permute the elements of the row vector
  if (permuteElements_host(this->h_randomIndices, this->range*this->range) != 0)
  {
    LOG(ERROR) << __func__ << " : failed to permute the indices array. \n";
    return -1;
  }
  if (this->sortSampler() != 0)
  {
    LOG(ERROR) << __func__ << " : failed to sort the sampling vectors. \n";
    return -1;
  }
  return 0;
}

/*!
Sample the wave function wfin and save the result into  wfout
sampling method: sampleN random indices {i} from the range [0, range * range - 1], sampled elements are {f_i}
\param wfin : 2D wave function (NxN) to sample from. N has to be greater than or equal to the sampling range
\param wfout : 1D wave function (1 x sampleN) to save the result into. Must have the space allocated
\return 0 if all goes well and -1 otherwise
*/
int CRandomSampler::sample(CWaveFunction * wfin, CWaveFunction * wfout){
  if (wfin == NULL || wfout == NULL)
  {
    LOG(ERROR) << __func__ << "Input and output wave function pointers cannot be NULL\n";
    return -1;
  }
  if (this->range > wfin->getRowSize() || this->range > wfin->getColsSize())
  {
    LOG(ERROR) << __func__ << " : Both dimensions of the input wave function ("<< wfin->getRowSize() <<"," << wfin->getColsSize() <<") have to have a range (" << this->range <<") which fits into the compressed wave function. \n";
    return -1;
  }
  if (wfout->getRowSize() * wfout->getColsSize() < this->sampleN)
  {
    LOG(ERROR) << __func__ << " : The wave function size is too small \n";
    return -1;
  }
  if (this->h_randomIndices == NULL)
  {
    LOG(ERROR) << __func__ << "Sampling vector cannot be NULL\n";
    return -1;
  }
  // Sample the wave function
  size_t rows [] = {0};
  if (sample_host(wfin->getHostWF(), 1, wfin->getColsSize() * wfin->getRowSize(), rows, 1, this->h_randomIndices, this->sampleN, wfout->getHostWF()) != 0)
  {
    LOG(ERROR) << __func__ << "Could not sample the wave function\n";
    return -1;
  }
  wfout->setDomain(true);
  return 0;
}

/*! Upsample the compressed wave function according to
  x = Transpose(A) y
  \param wfin : pointer to the compressed wavefunction. Its dimensions (1 x sampleN) must be the same as the sampler dimensions
  \param wfout : pointer to the wave function where we are supposed to store the result. Its dimesions (N x N) must fit the upsampled data (N >= range)
  \return 0 if all goes well and  -1 otherwise
*/
int CRandomSampler::upsample(CWaveFunction * wfin, CWaveFunction * wfout){
  if (wfin == NULL || wfout == NULL)
  {
    LOG(ERROR) << __func__ << "Input and output wave function pointers cannot be NULL\n";
    return -1;
  }
  if (this->range > wfout->getRowSize() || this->range > wfout->getColsSize())
  {
    LOG(ERROR) << __func__ << " : Both dimensions of the output wave function ("<< wfout->getRowSize() <<"," << wfout->getColsSize() <<") have to greater or equal than the sampling range (" << this->range <<").\n";
    return -1;
  }
  if (wfin->getRowSize() * wfin->getColsSize() != this->sampleN)
  {
    LOG(ERROR) << __func__ << " : The dimensions of the compressed wave function have to be szCol * szRows = sampleN";
    return -1;
  }
  memset((double *)wfout->getHostWF(), 0, 2* sizeof(double) * wfout->getRowSize() * wfout->getColsSize());
  size_t rows [] = {0};
  if (restore_host(wfin->getHostWF(), rows, 1, this->h_randomIndices, this->sampleN, wfout->getHostWF(), 1, wfout->getRowSize() * wfout->getColsSize()) != 0){
    LOG(ERROR) << __func__ << "Could not upsample the wave function\n";
    return -1;
  }
  wfout->setDomain(true);
  return 0;
}

/*!
Apply sampling operator f = Sample(IFFT(vec))
\param wfin : input vector in full frequency or space domain.
\param wfout : output vector in sampled space domain
\return 0 if all goes well and -1 otherwise.
*/
int CRandomSampler::applySamplingOperator(CWaveFunction * wfin, CWaveFunction * wfout){
  if (wfin == NULL || wfout == NULL)
  {
    LOG(INFO) << "The input and output wave functions must not be empty";
    return -1;
  }
  // IFFT(wfin)
  if (wfin->getDomain() == false)
  {
    wfin = new CWaveFunction(*wfin);
    wfin->switchDomain();
    //LOG(INFO) << " Apply operator, to space :" << wfin->getHostWF()[524800][0] << "+ i" << wfin->getHostWF()[524800][1] << ", " << wfin->getHostWF()[524801][0] << " + i" << wfin->getHostWF()[524801][1];
    if(this->sample(wfin, wfout) != 0)
    {
      LOG(INFO) << "Could not sample the wave function";
      return -1;
    }
    delete wfin;
  }
  else
  {
    if(this->sample(wfin, wfout) != 0)
    {
      LOG(INFO) << "Could not sample the wave function";
      return -1;
    }
  }
  //LOG(INFO) << " Apply operator, sampling :" << wfout->getHostWF()[0][0] << "+ i" << wfout->getHostWF()[0][1] << ", " << wfout->getHostWF()[0][0] << " + i" << wfout->getHostWF()[0][1];
  return 0;
}

/*!
Apply inverse sampling operator vec = Upsample(FFT(f))
\param wfin : input vector in sampled space domain.
\param wfout : output vector in full frequency domain
\return 0 if all goes well and -1 otherwise.
*/
int CRandomSampler::applyInverseSamplingOperator(CWaveFunction * wfin, CWaveFunction * wfout){
  if (wfin == NULL || wfout == NULL)
  {
    LOG(INFO) << "The input and output wave functions must not be empty";
    return -1;
  }
  if (wfin->getDomain() != true)
  {
    LOG(INFO) << "The input wave function must be in space domain";
    return -1;
  }
  if (this->upsample(wfin, wfout) != 0)
  {
    LOG(INFO) << "Could not apply inverse sampling operator";
    return -1;
  }
  wfout->switchDomain();
  return 0;
}
