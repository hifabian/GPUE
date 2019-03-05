#ifndef SAMPLERH
#define SAMPLERH
#include "wavefunction.h"

class CSampler {
public:
  //! Sample space vector
  virtual int sample(CWaveFunction * wfin, CWaveFunction * wfout)=0;
  //! upsample to space vector
  virtual int upsample(CWaveFunction * wfin, CWaveFunction * wfout)=0;
  //! Apply full sampling operator to freq vector
  virtual int applySamplingOperator(CWaveFunction * wfin, CWaveFunction * wfout)=0;
  //! Apply full inverse sampling operator to sampled space vector
  virtual int applyInverseSamplingOperator(CWaveFunction * wfin, CWaveFunction * wfout)=0;
  virtual int randomize(void)=0;
  virtual size_t getRange(void)=0;
};

class CRowColSampler : public CSampler {
  size_t range = 0;
  size_t * d_randomRows = NULL;
  size_t * h_randomRows = NULL;
  size_t szRows = 0;
  size_t * d_randomCols = NULL;
  size_t * h_randomCols = NULL;
  size_t szCols = 0;
  int maxThreads = 1024;
  // private methods

  /*! Sort first szCols (szRows) elements of the random column (row) vector
  \return 0 if all goes well and -1 otherwise
  */
  int sortSampler(void);
public:
  size_t * getRows_device(void){
    return this->d_randomRows;
  }
  size_t * getCols_device(void){
    return this->d_randomCols;
  }
  size_t * getRows_host(void){
    return this->h_randomRows;
  }
  size_t * getCols_host(void){
    return this->h_randomCols;
  }
  //! Get size
  size_t getSize(void)
  {
    return this->szCols * this->szRows;
  }
  size_t getRowSize(void){
    return this->szRows;
  }
  size_t getColsSize(void){
    return this->szCols;
  }
  size_t getRange(void){
    return this->range;
  }
  int getMaxThreads(void){
    return this->maxThreads;
  }
  /*!
  Sample the wave function wfin and save the result into  wfout
  sampling method: random row {i} and column {j} numbers, sampled elements are {f_ij}
  \param wfin : 2D wave function (NxN) to sample from. N has to be greater than or equal to the sampling range
  \param wfout : 2D wave function (szRows x szCols) to save the result into. Must have the space allocated
  \return 0 if all goes well and -1 otherwise
  */
  int sample(CWaveFunction * wfin, CWaveFunction * wfout);
  /*!
  Apply sampling operator f = Sample(IFFT(vec))
  \param wfin : input vector in full frequency domain
  \param wfout : output vector in sampled space domain
  \return 0 if all goes well and -1 otherwise
  */
  int applySamplingOperator(CWaveFunction * wfin, CWaveFunction * wfout);
  /*! Upsample the compressed wave function according to
    x = Transpose(A) y
    \param wfin : pointer to the compressed wavefunction. Its dimensions (szR x szC) must be the same as the sampler dimensions
    \param wfout : pointer to the wave function where we are supposed to store the result. Its dimesions (N x N) must fit the upsampled data (N >= range)
    \return 0 if all goes well and  -1 otherwise
  */
  int upsample(CWaveFunction * wfin, CWaveFunction * wfout);
  /*!
  Apply inverse sampling operator vec = Upsample(FFT(f))
  \param wfin : input vector in sampled space domain
  \param wfout : output vector in full frequency domain
  \return 0 if all goes well and -1 otherwise
  */
  int applyInverseSamplingOperator(CWaveFunction * wfin, CWaveFunction * wfout);
  //! Randomize the sampler (both rows and columns). returns 0 is all is ok and -1 otherwise
  int randomize(void);
  /*! Constructor
  Initializes the sampler with random sorted arrangement of szRow row and szCols column numbers taken from the given range.
  Rows {i} and columns {j} in [0, range-1)
  \param szRows: number of rows to sample
  \param szCols : number of columns to sample
  \param range : total number of rows or columns to sample from
  \param maxThreads : maximum number of threads per block on the device
  */
  CRowColSampler(size_t szRows, size_t szCols, size_t range, int maxThreads = 1024);
  // Destructor
  ~CRowColSampler();
};

/*!
Class CRandomSampler
Represents random sampler which samples a 2D vector
*/
class CRandomSampler : public CSampler {
  size_t range = 0;
  size_t sampleN = 0;
  //size_t * d_randomIndices = NULL;
  size_t * h_randomIndices = NULL;
  int maxThreads = 1024;
  // private methods

  /*! Sort first sampleN elements of the random vector
  \return 0 if all goes well and -1 otherwise
  */
  int sortSampler(void);
public:
  //! Get device indices vector
  // size_t * getIndices_device(void){
  //   return this->d_randomIndices;
  // }
  //! Get host indices vector
  size_t * getIndices_host(void){
    return this->h_randomIndices;
  }
  //! Get size
  size_t getSize(void)
  {
    return this->sampleN;
  }
  size_t getRange(void){
    return this->range;
  }
  int getMaxThreads(void){
    return this->maxThreads;
  }
  /*!
  Sample the wave function wfin and save the result into  wfout
  sampling method: random indices {i} from the range [0, range^2 - 1], sampled elements are {f_i}
  \param wfin : 2D wave function (NxN) to sample from. N has to be greater than or equal to the sampling range
  \param wfout : 1D wave function (1 x sampleN) to save the result into. Must have the space allocated
  \return 0 if all goes well and -1 otherwise
  */
  int sample(CWaveFunction * wfin, CWaveFunction * wfout);
  /*!
  Apply sampling operator f = Sample(IFFT(vec))
  \param wfin : input vector in full frequency domain
  \param wfout : output vector in sampled space domain
  \return 0 if all goes well and -1 otherwise
  */
  int applySamplingOperator(CWaveFunction * wfin, CWaveFunction * wfout);
  /*! Upsample the compressed wave function according to
    x = Transpose(A) y
    \param wfin : pointer to the compressed wavefunction. Its dimensions (szR x szC) must be the same as the sampler dimensions
    \param wfout : pointer to the wave function where we are supposed to store the result. Its dimesions (N x N) must fit the upsampled data (N >= range)
    \return 0 if all goes well and  -1 otherwise
  */
  int upsample(CWaveFunction * wfin, CWaveFunction * wfout);
  /*!
  Apply inverse sampling operator vec = Upsample(FFT(f))
  \param wfin : input vector in sampled space domain
  \param wfout : output vector in full frequency domain
  \return 0 if all goes well and -1 otherwise
  */
  int applyInverseSamplingOperator(CWaveFunction * wfin, CWaveFunction * wfout);
  //! Randomize the sampler. returns 0 is all is ok and -1 otherwise
  int randomize(void);
  /*! Constructor
  Initializes the sampler with random sorted arrangement of sampleN indices taken from the range [0, range^2 - 1].
  \param sN: number of indices to sample
  \param range : total number of rows or columns to sample from
  \param maxThreads : maximum number of threads per block on the device
  */
  CRandomSampler(size_t sN, size_t range, int maxThreads = 1024);
  // Destructor
  ~CRandomSampler();
};
#endif
