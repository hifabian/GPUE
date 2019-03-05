#ifndef CSUTILSH
#define CSUTILSH
#include <cufft.h>
#include <vector>
#include <complex.h>
#include <fftw3.h>

/*!
  Find vector support given the number of largest elements
  \param d_data : data vector allocated on device
  \param d_support : output support data allocated on device
  \param sz : size of d_data and d_support vectors
  \param k : the number of largest elements to take into account. k-th largest elemets is going to be the support threshold
  return 0 if all goes well and -1 otherwise. d_data now contains thresholded vector, and d_support contains 1 if the element is in the support and 2 otherwise.
*/
int findSupport_sort_device(cufftDoubleComplex * d_data, char * d_support, size_t sz, size_t k, int maxThreads = 1024);

/*!
  Find vector support given the number of largest elements
  \param h_complexWF : data allocated on the host
  \param sz : size of h_complexWF vector
  \param k : the number of largest elements to take into account. k-th largest elemets is going to be the support threshold
  \param maxMem : maximum free memory on the device (in bytes). default 64 mb
  \param maxThreads : maximum threads per block on the device. default is 1024
  \return pointer to support array of size sz if all goes well and NULL otherwise. h_real h_imag now contain thresholded vectors, and support contains 1 if the element is in the support and 2 otherwise.
*/
char * findSupport_sort_host(fftw_complex * h_complexWF, size_t sz, size_t k, size_t maxMem = 67108864, int maxThreads=1024);

/*!
  Threshold the vector according to the given support
  \param h_complexWF : data allocated on the host
  \param support : vector with 1s if the element belong to support and 2s otherwise
  \param sz : size of the data
  \return 0 if all goes well and -1 otherwise
*/
int thresholdToSupport(fftw_complex * h_complexWF, char * h_support, size_t sz);

/*!
  Split big complex host data, convert it to abs value piece by piece on the GPU and return the result as an allocated array of abs values on the host.
  NOTE: delete results with delete []
  \param h_complexWF : data allocated on the host
  \param sz : size of the data
  \param maxMem : maximum free memory on the device (in bytes). default 64 mb
  \param maxThreads : maximum threads on the device. default 1024
  \return pointer to newly allocated array of abs values
*/
double * toAbs_host(fftw_complex * h_complexWF, size_t sz, size_t maxMem = 67108864, int maxThreads=1024);

__global__ void sampleKernel(cufftDoubleComplex * d_data, size_t dataRowN, size_t dataColNum, size_t * d_rows, size_t rowNum, size_t * d_cols, size_t colNum, cufftDoubleComplex * d_sampled);

/*!
  Restore full-sized vector from downsampled one according to the sampling vectors
  \param h_complexWF : the downsampled vector of size rowNum*colNum
  \param h_rows : sampling rows
  \param h_cols : sampling columns
  \param rowNum : number of rows sampled
  \param colNum : number of columns sampled
  \param h_outComplex : full vector of size dataRowN*dataColNum dataRowN*dataColNum
  \param dataRowN : number of rows
  \param dataColNum : number of columns
*/
int restore_host(fftw_complex * h_complexWF, size_t * h_rows, size_t rowNum, size_t * h_cols, size_t colNum, fftw_complex * h_outComplex, size_t dataRowN, size_t dataColNum);

/*! sample the array according to permutations of columns and rows
Faster if d_cols and d_rows are sorted
\params h_outComplex : already allocated array of size rowNum * colNum
*/
int sample_host(fftw_complex * h_complexWF, size_t dataRowN, size_t dataColNum, size_t * h_rows, size_t rowNum, size_t * h_cols, size_t colNum, fftw_complex * h_outComplex);

#endif
