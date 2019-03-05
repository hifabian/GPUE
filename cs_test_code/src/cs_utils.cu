#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <algorithm>
#include "cs_utils.h"
#include "mxutils.h"
#include "easyloggingcpp/easylogging++.h"

__device__ inline double complexAbs(cufftDoubleComplex z){
  return z.x * z.x + z.y * z.y;
}

/*! sample the array according to permutations of columns and rows
Faster if d_cols and d_rows are sorted
\params h_outComplex : already allocated array of size rowNum * colNum
*/
int sample_host(fftw_complex * h_complexWF, size_t dataRowN, size_t dataColNum, size_t * h_rows, size_t rowNum, size_t * h_cols, size_t colNum, fftw_complex * h_outComplex){
  if (h_complexWF == NULL || dataRowN == 0 || dataColNum == 0)
  {
    LOG(ERROR) << __func__ << " :the input vectors must not be empty\n";
    return -1;
  }
  if (h_rows == NULL || rowNum == 0 || h_cols == NULL || colNum == 0)
  {
    LOG(ERROR) << __func__ << " :the sampling vectors must not be empty\n";
    return -1;
  }
  if (h_outComplex == NULL)
  {
    LOG(ERROR) << __func__ << " :the output vector must not be empty\n";
    return -1;
  }
  for (size_t i = 0; i < rowNum; i++)
    for (size_t j = 0; j < colNum; j++)
    {
      size_t curDstPos = i * colNum + j;
      size_t curSrcPos = h_rows[i] * dataColNum + h_cols[j];
      h_outComplex[curDstPos][0] = h_complexWF[curSrcPos][0];
      h_outComplex[curDstPos][1] = h_complexWF[curSrcPos][1];
    }
  return 0;
}

//! Sample the array according to permutations of columns and rows
//! Faster if d_cols and d_rows are sorted
__global__ void sampleKernel(cufftDoubleComplex * d_data, size_t dataRowN, size_t dataColNum, size_t * d_rows, size_t rowNum, size_t * d_cols, size_t colNum, cufftDoubleComplex * d_sampled){
  // First we need to find our global threadID
  int tPosX = blockIdx.x * blockDim.x + threadIdx.x;
  int tPosY = blockIdx.y * blockDim.y + threadIdx.y;
  if (tPosX >= colNum || d_cols[tPosX] >= dataColNum)
    return;
  if (tPosY >= rowNum || d_rows[tPosY] >= dataRowN)
    return;
  int curDstPos = tPosY * colNum + tPosX;
  int curSrcPos = d_rows[tPosY] * dataColNum + d_cols[tPosX];
  d_sampled[curDstPos] = d_data[curSrcPos];
}

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
int restore_host(fftw_complex * h_complexWF, size_t * h_rows, size_t rowNum, size_t * h_cols, size_t colNum, fftw_complex * h_outComplex, size_t dataRowN, size_t dataColNum){
  if (h_complexWF == NULL || dataRowN == 0 || dataColNum == 0)
  {
    LOG(ERROR) << __func__ << " :the input vectors must not be empty\n";
    return -1;
  }
  if (h_rows == NULL || rowNum == 0 || h_cols == NULL || colNum == 0)
  {
    LOG(ERROR) << __func__ << " :the sampling vectors must not be empty\n";
    return -1;
  }
  if (h_outComplex == NULL)
  {
    LOG(ERROR) << __func__ << " :the output vectors must not be empty\n";
    return -1;
  }
  for (size_t i = 0; i < rowNum; i++)
    for (size_t j = 0; j < colNum; j++)
    {
      size_t curSrcPos = i * colNum + j;
      size_t curDstPos = h_rows[i] * dataColNum + h_cols[j];
      h_outComplex[curDstPos][0] = h_complexWF[curSrcPos][0];
      h_outComplex[curDstPos][1] = h_complexWF[curSrcPos][1];
    }
  return 0;
}
//! Restore the array according to permutations of columns and rows
//! Faster if d_cols and d_rows are sorted
__global__ void restoreKernel(cufftDoubleComplex * d_data, size_t dataRowN, size_t dataColNum, size_t * d_rows, size_t rowNum, size_t * d_cols, size_t colNum, cufftDoubleComplex * d_sampled){
  // First we need to find our global threadID
  int tPosX = blockIdx.x * blockDim.x + threadIdx.x;
  int tPosY = blockIdx.y * blockDim.y + threadIdx.y;
  if (tPosX >= colNum || d_cols[tPosX] >= dataColNum)
    return;
  if (tPosY >= rowNum || d_rows[tPosY] >= dataRowN)
    return;
  int curSrcPos = tPosY * colNum + tPosX;
  int curDstPos = d_rows[tPosY] * dataColNum + d_cols[tPosX];
  d_data[curDstPos] = d_sampled[curSrcPos];
}

__global__ void toAbs(cufftDoubleComplex * d_data, double * d_out, size_t sz){
  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  if (myId >= sz)
    return;
  d_out[myId] = complexAbs(d_data[myId]);
}

__global__ void thresholdAndSupport(cufftDoubleComplex * d_data, double * d_sortedData, char * d_support,  size_t sz, size_t k){
  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  if (myId >= sz)
    return;
  double b = d_sortedData[sz - k];
  if (complexAbs(d_data[myId]) < b)
  {
    d_data[myId].x = 0;
    d_data[myId].y = 0;
    d_support[myId] = 2;
  }
  else d_support[myId] = 1;
}

int findSupport_sort_device(cufftDoubleComplex * d_data, char * d_support, size_t sz, size_t k, int maxThreads){
  if (maxThreads <= 0)
  {
    LOG(ERROR) << __func__ << " : maximum thread number per block must be positive\n";
    return -1;
  }
  if (k == 0)
  {
    LOG(ERROR) << __func__ << " : number of largest elements k has to be greater than zero\n";
    return -1;
  }
  if (sz == 0 || d_data == NULL)
  {
    LOG(ERROR) << __func__ << " :the input vector must not be empty\n";
    return -1;
  }
  if (d_support == NULL)
  {
    LOG(ERROR) << __func__ << " :the support vector must not be empty\n";
    return -1;
  }
  double * d_tmpdata = NULL;
  cudaError_t err = cudaMalloc(&d_tmpdata, sizeof(double) * sz);
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << " : Could not allocate memory on the device: " << cudaGetErrorString(err) << "\n";
    return -1;
  }
  int threadN = min((size_t)maxThreads, sz);
  int gridSize = ceil(sz / (float)threadN);
  toAbs<<<gridSize, threadN>>>(d_data, d_tmpdata, sz);
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << " : toAbs kernel execution failed: " << cudaGetErrorString(err) << "\n";
    return -1;
  }
  //wrap raw pointer with a device_ptr to use with Thrust functions
  thrust::device_ptr<double> dthrust_data(d_tmpdata);
  thrust::sort(dthrust_data, dthrust_data + sz);
  thresholdAndSupport<<<gridSize, threadN>>>(d_data, d_tmpdata, d_support, sz, k);
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << " : thresholdAndSupport kernel execution failed: " << cudaGetErrorString(err) << "\n";
    return -1;
  }
  cudaFree(d_tmpdata);
  return 0;
}

/*!
  Split big complex host data, convert it to abs value squared piece by piece on the GPU and return the result as an allocated array of abs values on the host.
  NOTE: delete results with delete []
*/
double * toAbs_host(fftw_complex * h_complexWF, size_t sz, size_t maxMem, int maxThreads){
  if (sz == 0 || h_complexWF == NULL)
  {
    LOG(ERROR) << __func__ << " :the input vector must not be empty\n";
    return NULL;
  }
  if (maxThreads <= 0)
  {
    LOG(ERROR) << __func__ << " : maximum thread number per block must be positive\n";
    return NULL;
  }
  double * h_abs = new double[sz];
  for (size_t i = 0; i < sz; i++)
  {
    h_abs[i] = h_complexWF[i][0] * h_complexWF[i][0] + h_complexWF[i][1] * h_complexWF[i][1];
  }
  return h_abs;
  // // amount of leftover memory for doubles on the device
  // size_t maxMemDouble = maxMem / sizeof(double);
  // // We need to store real, imaginary and abs val on the gpu
  // size_t maxDoubleArraySz = maxMemDouble / 3;
  // // Number of GPU runs we have to do
  // size_t numRuns = sz / maxDoubleArraySz;
  // // size of the last GPU run
  // size_t lastRunSize = sz % maxDoubleArraySz;
  // // Make sure the last run has the right size
  // if (lastRunSize == 0)
  // {
  //   numRuns -= 1;
  //   lastRunSize = maxDoubleArraySz;
  // }
  // // Allocate all data on the gpu
  // cufftDoubleComplex * d_data = NULL;
  // double * d_abs = NULL;
  // cudaError_t err = cudaMalloc(&d_data, sizeof(cufftDoubleComplex) * maxDoubleArraySz);
  // if (err != cudaSuccess)
  // {
  //   LOG(ERROR) << __func__ << " : Could not allocate memory for d_data on the device: " << cudaGetErrorString(err) << "\n";
  //   delete [] h_abs;
  //   return NULL;
  // }
  // err = cudaMalloc(&d_abs, sizeof(double) * maxDoubleArraySz);
  // if (err != cudaSuccess)
  // {
  //   LOG(ERROR) << __func__ << " : Could not allocate memory for d_abs on the device: " << cudaGetErrorString(err) << "\n";
  //   delete [] h_abs;
  //   cudaFree(d_data);
  //   return NULL;
  // }
  // // Run the bulk
  // fftw_complex * curComplex = h_complexWF;
  // double * curAbs = h_abs;
  // int threadN = min((size_t)maxThreads, maxDoubleArraySz);
  // int gridSize = ceil(maxDoubleArraySz / (float)threadN);
  // for (int j = 0; j < numRuns; j++)
  // {
  //   // Copy real and imaginary parts into the cuda complex data
  //   if (copyToCudaComplex(d_data, sizeof(cufftDoubleComplex) * maxDoubleArraySz, curComplex, maxDoubleArraySz, 1) != 0)
  //   {
  //     LOG(ERROR) << __func__ <<" bulk run N" << j << " : Could not copy real and imaginary parts to gpu: \n";
  //     delete [] h_abs;
  //     cudaFree(d_data);
  //     cudaFree(d_abs);
  //     return NULL;
  //   }
  //   curComplex = &curComplex[j * maxDoubleArraySz];
  //   // Run the kernel
  //   toAbs<<<gridSize, threadN>>>(d_data, d_abs, maxDoubleArraySz);
  //   err = cudaGetLastError();
  //   if (err != cudaSuccess)
  //   {
  //     LOG(ERROR) << __func__ << " : toAbs kernel execution failed: " << cudaGetErrorString(err) << "\n";
  //     delete [] h_abs;
  //     cudaFree(d_data);
  //     cudaFree(d_abs);
  //     return NULL;
  //   }
  //   // Copy memory back to cpu
  //   err = cudaMemcpy(curAbs, d_abs, sizeof(double) * maxDoubleArraySz, cudaMemcpyDeviceToHost);
  //   if (err != cudaSuccess)
  //   {
  //     LOG(ERROR) << __func__ << " : Could not copy the result back to cpu: " << cudaGetErrorString(err) << "\n";
  //     delete [] h_abs;
  //     cudaFree(d_data);
  //     cudaFree(d_abs);
  //     return NULL;
  //   }
  //   curAbs = &h_abs[j * maxDoubleArraySz];
  // }
  // // perform the last Run
  // // Copy real and imaginary parts into the cuda complex data
  // if (convertToCudaComplex(d_data, sizeof(cufftDoubleComplex) * lastRunSize, curReal, curImag, lastRunSize, 1) != 0)
  // {
  //   LOG(ERROR) << __func__ <<" last run "<< " : Could not copy real and imaginary parts to gpu: \n";
  //   delete [] h_abs;
  //   cudaFree(d_data);
  //   cudaFree(d_abs);
  //   return NULL;
  // }
  // // Run the kernel
  // threadN = min((size_t)maxThreads, lastRunSize);
  // gridSize = ceil(lastRunSize / (float)threadN);
  // toAbs<<<gridSize, threadN>>>(d_data, d_abs, lastRunSize);
  // err = cudaGetLastError();
  // if (err != cudaSuccess)
  // {
  //   LOG(ERROR) << __func__ << " last run : toAbs kernel execution failed: " << cudaGetErrorString(err) << "\n";
  //   delete [] h_abs;
  //   cudaFree(d_data);
  //   cudaFree(d_abs);
  //   return NULL;
  // }
  // // Copy memory back to cpu
  // err = cudaMemcpy(curAbs, d_abs, sizeof(double) * lastRunSize, cudaMemcpyDeviceToHost);
  // if (err != cudaSuccess)
  // {
  //   LOG(ERROR) << __func__ << " : Could not copy the result back to cpu: " << cudaGetErrorString(err) << "\n";
  //   delete [] h_abs;
  //   cudaFree(d_data);
  //   cudaFree(d_abs);
  //   return NULL;
  // }
  // // clean up
  // cudaFree(d_data);
  // cudaFree(d_abs);
  // // return result
  // return h_abs;
}

/*!
  Find vector support given the number of largest elements
  \param h_complexWF : data allocated on the host
  \param sz : size of d_data and d_support vectors
  \param k : the number of largest elements to take into account. k-th largest elemets is going to be the support threshold
  \param maxMem : maximum fre memory on the device (in bytes). default 64 mb
  \param maxThreads : maximum threads per block on the device. default is 1024
  \return pointer to support array of size sz if all goes well and NULL otherwise. h_real h_imag now contain thresholded vectors, and support contains 1 if the element is in the support and 2 otherwise.
*/
char * findSupport_sort_host(fftw_complex * h_complexWF, size_t sz, size_t k, size_t maxMem, int maxThreads){
  if (k == 0)
  {
    LOG(ERROR) << __func__ << " : number of largest elements k has to be greater than zero\n";
    return NULL;
  }
  if (sz == 0 || h_complexWF == NULL)
  {
    LOG(ERROR) << __func__ << " :the input vector must not be empty\n";
    return NULL;
  }
  // Get |z|^2
  double * h_abs = toAbs_host(h_complexWF, sz, maxMem, maxThreads);
  if (h_abs == NULL)
  {
    LOG(ERROR) << __func__ << " :Could not generate a vector of absolute values\n";
    return NULL;
  }
  // Sort the abs values
  std::sort(h_abs, h_abs + sz);
  double threshold = h_abs[sz - k];
  delete [] h_abs;
  // threshold the data and fill in support
  char * support = new char[sz];
  for (size_t i = 0; i < sz; i++)
  {
    if (h_complexWF[i][0] * h_complexWF[i][0] + h_complexWF[i][1] * h_complexWF[i][1] < threshold)
    {
      h_complexWF[i][0] = 0;
      h_complexWF[i][1] = 0;
      support[i] = 2;
    }
    else{
      support[i] = 1;
    }
  }
  return support;
}

int thresholdToSupport(fftw_complex * h_complexWF, char * h_support, size_t sz){
  if (sz == 0 || h_complexWF == NULL || h_support == NULL)
  {
    LOG(ERROR) << __func__ << " :the input vectors must not be empty\n";
    return -1;
  }
  for (int i = 0; i < sz; i++)
  {
    double a = 2 - h_support[i];
    h_complexWF[i][0] *= a;
    h_complexWF[i][1] *= a;
  }
  return 0;
}
