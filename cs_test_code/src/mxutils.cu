/**
* Various matrix utils using cuda
**/
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include<curand.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include<curand_kernel.h>
#include "mxutils.h"
#include "utils.h"


/**
* Kronecker product of two matrices kernel
* input :
* a : first matrix
* nax, nay : matrix a dimensions
* b: second matrix
* nbx, nby : matrix b dimensions
* results : kronecker product of a and b
**/
__global__ void kronecker(double * a, int nax, int nay, double * b, int nbx, int nby, double * result){

    // First we need to find our global threadID
    int tPosX = blockIdx.x * blockDim.x + threadIdx.x;
    int tPosY = blockIdx.y * blockDim.y + threadIdx.y;
    int resSzx = nax * nbx;
    //int resSzy = nay * nby;
    int idxA = floor((tPosX) / (double)nbx);
    int idyA = floor((tPosY) / (double)nby);
    int idxB = (tPosX) % nbx;
    int idyB = (tPosY) % nby;
    // Check if the indices are within range
    if (idxA >= nax || idyA > nay || idxB > nbx || idyB > nby)
    {
      result[tPosX + tPosY * resSzx] = -1;
      return;
    }
    // Multiply appropriate elements
    result[tPosX + tPosY * resSzx] = a[idyA * nax +  idxA] * b[idyB * nbx + idxB];
}

void saveMatrixToCSV(std::string fnameRe, std::string fnameIm, fftw_complex * mx, size_t szX, size_t szY)
{
  std::ofstream outRe(fnameRe, std::ofstream::out);
  std::ofstream outIm(fnameIm, std::ofstream::out);
  for (int i = 0; i < szY; i++)
  {
    for (int j = 0; j < szX - 1; j++)
    {
      outRe << mx[szX * i + j][0] <<',';
      outIm << mx[szX * i + j][1] <<',';
    }
    outRe << mx[szX * i + szX - 1][0] << "\n";
    outIm << mx[szX * i + szX - 1][1] << "\n";
  }
}

void saveMatrixToCSV(std::string fname, double * mx, size_t szX, size_t szY)
{
  std::ofstream out(fname, std::ofstream::out);
  for (int i = 0; i < szY; i++)
  {
    for (int j = 0; j < szX - 1; j++)
    {
      out << mx[szX * i + j] <<',';
    }
    out << mx[szX * i + szX - 1] << "\n";
  }
}

/**
* Parse the csv file and return the array.
\param fn : file name
\param szX : New array size X (fastest-changing coordinate) is going to be stored in this variable
\param szY : New array size Y (slow-changing coordinate) is going to be stored in this variable
\return pointer to the array
* IMPORTANT NOTE: use delete [] arr when you are done using it
**/
double * parseMxFromCSV(std::string fn, size_t &szX, size_t &szY)
{
    std::ifstream  data(fn);
    if (!data.good())
      return NULL;
    std::string line;
    std::vector<std::vector<double> > parsedCsv;
    while(std::getline(data,line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<double> parsedRow;
        while(std::getline(lineStream,cell,','))
        {
            double x = 0;
            std::istringstream c(cell);
            c >> x;
            parsedRow.push_back(x);
        }

        parsedCsv.push_back(parsedRow);
    }
    szY = parsedCsv.size();
    szX = parsedCsv[0].size();
    //std::cout << "malloc\n";
    double * mx = new double[szX * szY];//(double *)malloc(sizeof(double) * szX * szY);
    //std::cout<<"allocated\n";
    for (int i = 0; i < szY; i++)
    {
      for (int j = 0; j < szX; j++)
      {
        mx[i * szX + j] = parsedCsv[i][j];
      }
    }
    //std::cout << "copied\n";
    return mx;
};

/**
* Convert double-valued wave function into cufftDoubleComplex format
* Input:
* cufftDoubleComplex * d_complexData : destination array (must be on the device)
* int pitch : the width in bytes of the 2D array pointed to by d_complexData, including any padding added to the end of each row. This function performs fastest when the pitch is one that has been passed back by cudaMallocPitch().
NOTE: To free the memory use cudaFree()
* double * h_data : double-valued wave function (host)
* int szX : Wf size X (fastest-changing coordinate)
* int szY : Wf size Y (slow-changing coordinate)
* Return:
* 0 of all goes well and -1 if there is an error
**/
int copyToCudaComplex(cufftDoubleComplex * d_complexData, size_t pitch, fftw_complex * h_complexWF, size_t szX, size_t szY)
{
  if (h_complexWF == NULL || szX == 0 || szY == 0)
  {
    LOG(ERROR) <<  __func__ << " : input data is either NULL or the dimensions are invalid\n";
    return -1;
  }
  if (d_complexData == NULL)
  {
    LOG(ERROR) <<  __func__ << " : output data is NULL\n";
    return -1;
  }
  if (pitch < szX)
  {
    LOG(ERROR) <<  __func__ << " : the pitch is invalid. Pitch has to be greater or equal to sizeX\n";
    return -1;
  }
  cudaError_t cudaStat = cudaMemcpy2D (d_complexData, pitch,
                         h_complexWF, sizeof(h_complexWF[0]) * szX,
                         sizeof(h_complexWF[0]) * szX, szY, cudaMemcpyHostToDevice);
  if (cudaStat != cudaSuccess)
  {
   LOG(ERROR) << __func__ << " : cudaMemcpy2D failed " << cudaGetErrorString(cudaStat) << "\n";
   return -1;
  }
  return 0;
}


__global__ void fillExpV(double g_over_dx, double half_i_dt, const double * d_V, cufftDoubleComplex * d_expV_phi, size_t vSz, bool useShared){
  // Declare dynamical shared memory
  extern __shared__ double s_V[];
  const double * ptrV = NULL;
  // First we need to find our global threadID
  int tPosX = blockIdx.x * blockDim.x + threadIdx.x;
  int tPosY = blockIdx.y * blockDim.y + threadIdx.y;
  if (tPosY >= vSz || tPosX >= vSz)
    return;
  if (d_expV_phi == NULL)
    return;
  int idx = tPosY * vSz + tPosX;
  // Copy the potential vector to shared memory if below is true:
  // If the flag is set; and the number of thread in the block is
  // greater of equal than the size of the vector;
  // and if the index of the current thread is between 0 and vSz
  int indexInBlock = blockDim.x * threadIdx.y + threadIdx.x;
  if (useShared && blockDim.x * blockDim.y >= vSz && indexInBlock < vSz)
  {
    //if(indexInBlock == 0)
    //  printf("=============\n");
    s_V[indexInBlock] = d_V[indexInBlock];
    //printf("(%1.5f, %1.5f) | ", s_V[indexInBlock], d_V[indexInBlock]);
  }
  __syncthreads();
  if (useShared && blockDim.x * blockDim.y >= vSz)
    ptrV = s_V;
  else
    ptrV = d_V;
  // Calculate the phase value
  double rho = ptrV[tPosX] + ptrV[tPosY] + (tPosX == tPosY) * g_over_dx;
  // exp(i b) = cos(b) + i * sin(b)
  double * curVal = (double *)&d_expV_phi[idx];
  curVal[0] = cos(rho * half_i_dt);
  curVal[1] = sin(rho * half_i_dt);
}

//! Make exp(-0.5i dt V) matrix
/*!
  \param dt : time step
  \param Vt : potential vector at the current time step
  \param vSize : size of the potential vector
  \param g : Interaction strength
  \param dx : Spatial resolution
  \return vSize x vSize matrix Exp(-0.5 i V dt) in the device memory
*/
cufftDoubleComplex * makeExpV(double dt, const double * d_Vt, size_t vSize, double g, double dx){
  if (d_Vt == NULL || vSize == 0)
  {
    LOG(ERROR) << __func__ << " : potential vector Vt is invalid\n";
    return NULL;
  }
  double goverdx = g / dx;
  double half_i_dt = -0.5 * dt;
  // Allocate memory for the output array
  cufftDoubleComplex * d_expV = NULL;
  cudaError_t err = cudaMalloc(&d_expV, sizeof(cufftDoubleComplex) * vSize * vSize);
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << " : Could not allocate memory for the resulting array\n";
    return NULL;
  }
  // Fill in the array
  // Get cuda properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  // Maximum threads per block on this device
  int maxThreads = prop.maxThreadsPerBlock;
  // Maximum dynamic shared memory size per block
  size_t sharedMemSz = prop.sharedMemPerBlock;
  // Call the kernel to generate expV
  int blockSzX = min((size_t)maxThreads, vSize);
  dim3 threadsPerBlock(blockSzX, 1);
  dim3 grid(ceil(vSize/blockSzX), vSize);
  fillExpV<<<grid, threadsPerBlock, min((size_t)sharedMemSz, vSize * sizeof(double))>>>(goverdx, half_i_dt, d_Vt, d_expV, vSize, sharedMemSz > vSize * sizeof(double));
  err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << " : Something went wrong during the fillExpV kernel call -> " << cudaGetErrorString(err) << "\n";
    return NULL;
  }
  return d_expV;
}

/*!
  Performs in-place FFT on device data
  \param d_data : pointer to device data
  \param szX : size of the data's fastest changing dimension
  \param szY : size of the data's slower changing dimension
  \param plan : pointer to a cufft plan structure. If plan == NULL, new cufft plan is going to be created and its pointer saved in plan variable
  \param forward : foward FFT will be performed if true, inverse otherwise
  \return 0 if succes and -1  otherwise
  NOTE: in case of IFFT, the output data is scaled. You have to divide it by szX * szY in order to get IFFT(FFT(A)) == A
*/
int doFFT(cufftDoubleComplex * d_data, size_t szX, size_t szY, cufftHandle * &plan, bool forward){
  if (d_data == NULL || szX == 0 || szY == 0)
  {
    LOG(ERROR) << __func__ << " : Input data is empty\n";
    return -1;
  }
  // szX is the fastest changing dimension
  int n[2] = {(int)szY, (int)szX};
  if (plan == NULL)
  {
    plan = new cufftHandle();
    cufftResult err = cufftPlan2d(plan, n[0], n[1], CUFFT_Z2Z);
    if (err != CUFFT_SUCCESS)
    {
      LOG(ERROR) << __func__ << " : Cannot create cufft plan: " << cufftGetErrorString(err) << "\n";
      return -1;
    }
  }
  cufftResult err =  cufftExecZ2Z(*plan, d_data, d_data, (forward ? CUFFT_FORWARD : CUFFT_INVERSE));
  if (err != CUFFT_SUCCESS)
  {
    LOG(ERROR) << __func__ << " : Cannot execute " << (forward ? "forward" : "inverse" ) << " cufft plan: " << cufftGetErrorString(err) << "\n";
    return -1;
  }
  // cudaError_t err2 = cudaDeviceSynchronize();
  // if (err2 != cudaSuccess)
  // {
  // 	LOG(ERROR) << __func__ << " : Cuda error: Failed to synchronize : " << cudaGetErrorString(err2) << "\n";
  //  	return -1;
  // }
  return 0;
}

__global__ void reduceNorm(cufftDoubleComplex * d_wf,
                       const size_t sz,
                       double * d_out)
{
  extern __shared__ double sh_out [];
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tId = threadIdx.x;
  if ( myId >= sz)
  {
    sh_out[tId] = 0.0;
  }
  else
  {
    // Fill in the shared memory
    double a = d_wf[myId].x; //real
    double b = d_wf[myId].y; //imag
    sh_out[tId] = a * a + b * b;
  }
  __syncthreads();
  for  (unsigned int s = blockDim.x /2; s > 0; s >>=1)
  {
    if (tId < s)
    {
      sh_out[tId] += sh_out[tId+s];
    }
    __syncthreads();
  }
  if (tId == 0)
  {
    d_out[blockIdx.x] = sh_out[0];
  }
}

__global__ void reduceSum(double * d_arr,
                       const size_t sz,
                       double * d_out)
{
  extern __shared__ double sh_out [];
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tId = threadIdx.x;
  if ( myId >= sz)
  {
    sh_out[tId] = 0.0;
  }
  else
  {
    // Fill in the shared memory
    sh_out[tId] = d_arr[myId];
  }
  __syncthreads();
  for  (unsigned int s = blockDim.x /2; s > 0; s >>=1)
  {
    if (tId < s)
    {
      sh_out[tId] += sh_out[tId+s];
    }
    __syncthreads();
  }
  if (tId == 0)
  {
    d_out[blockIdx.x] = sh_out[0];
  }
}



/*!
  Get the norm of the wave function |f| = sqrt(sum(|f_ij|^2)). Assume dx = 1;
  To get real norm, multiply the results by real dx
  \param d_wf : the sz x sz wave function allocated on the device
  \param sz : wave function size (dim(d_wf) = sz x sz)
  \param result : variable to save the result in
  \return 0 if success and -1 otherwise
*/
int getNorm(cufftDoubleComplex * d_wf, size_t sz, double & result){
  if (d_wf == NULL || sz == 0)
  {
    LOG(ERROR) << __func__ << " : The wave function is invalid\n";
   	return -1;
  }
  // Query device information
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int maxThreads = prop.maxThreadsPerBlock;
  // Set up block and grid size - I assume memory pitch == sz * sizeof(double)
  int blockSize = (int) min((size_t)maxThreads, (size_t)sz * sz);
  int gridSize = ceil((sz * sz) / (float)blockSize);
  int bufSize = pow(2, ceil(log(gridSize)/log(2)));
  //std::cout << "Reduce norm\n";
  //std::cout << blockSize << " " << gridSize << " " << bufSize << std::endl;
  double * d_tmp = NULL;
  cudaError_t err = cudaMalloc(&d_tmp, sizeof(double) * bufSize);
  if(err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << " : Could not allocate auxiliary device memory : " << cudaGetErrorString(err) << "\n";
    return -1;
  }
  // First step
  reduceNorm<<<gridSize, blockSize, sizeof(double) * blockSize>>>
    (d_wf, sz * sz, d_tmp);
  err = cudaGetLastError();
  if(err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << " : Primary Execution of reduceNorm kernel failed : " << cudaGetErrorString(err) << "\n";
    cudaFree(d_tmp);
    return -1;
  }
  // Secondary steps
  int iterN = 0;
  while (gridSize > 1)
  {
    blockSize = (int) min((size_t)maxThreads, (size_t)bufSize);
    gridSize = ceil((bufSize) / (float)blockSize);
    reduceSum<<<gridSize, blockSize, sizeof(double) * blockSize>>>
      (d_tmp, bufSize, d_tmp);
    err = cudaGetLastError();
    if(err != cudaSuccess)
    {
      LOG(ERROR) << __func__ <<"Iteration " << iterN << " : Secondary Execution of reduceSum kernel failed : " << cudaGetErrorString(err) << "\n";
      cudaFree(d_tmp);
      return -1;
    }
    bufSize = pow(2, ceil(log(gridSize)/log(2)));
    iterN++;
  }
  if (copyArrFromGPU<double>(&result, d_tmp, 1) != 0)
  {
    LOG(ERROR) << __func__ <<" : Result copy error \n";
    cudaFree(d_tmp);
    return -1;
  }
  result = sqrt(result);
  cudaFree(d_tmp);
  return 0;
}

__global__ void renormalize(cufftDoubleComplex * d_arr, size_t sz, double norm){
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  if (myId >= sz)
    return;
  if (norm == 0)
    return;
  //printf("(%u, Before: %.5f, ", myId, d_arr[myId].x);
  d_arr[myId].x = d_arr[myId].x / norm;
  //printf(" After: %.5f)", d_arr[myId].x);
  d_arr[myId].y = d_arr[myId].y / norm;
}

/*!
  Normalize device wave function psi = psi/(dx * Sqrt(Sum(|psi[i]|^2 ))). I assume its a 2D wf
  \param dx : step for integration > 0
  \param d_wf : pointer to the wave function in device memory
  \param sz : the dimensions if the 2D wf is sz x sz
  \return 0 if success and -1 otherwise
*/
int normalize(cufftDoubleComplex * d_wf, size_t sz, double dx){
  double norm = 0;
  if (dx <= 0)
  {
    LOG(ERROR) << __func__ <<" : dx has to be greater than 0\n";
    return -1;
  }
  if (getNorm(d_wf, sz, norm) != 0)
  {
    LOG(ERROR) << __func__ <<" : Couldn't Calculate the norm\n";
    return -1;
  }
  norm *= dx;
  //std::cout << "Norm: " << norm << std::endl;
  // Query device information
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  int maxThreads = prop.maxThreadsPerBlock;
  int blockSize = (int) min((size_t)maxThreads, (size_t)sz * sz);
  int gridSize = ceil((sz * sz) / (float)blockSize);
  renormalize<<<gridSize, blockSize>>>(d_wf, sz*sz, norm);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ <<" : Execution of renormalize kernel failed : " << cudaGetErrorString(err) << "\n";
    return -1;
  }
  return 0;
}


__global__ void randomKey(size_t N, float * d_dst, unsigned long seed){
  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  if (myId >= N)
    return;
  curandState state;
  curand_init ( seed, myId, 0, &state);
  float RANDOM = curand_uniform( &state );
  d_dst[myId] = (float)RANDOM;
}

__global__ void pick(size_t N , size_t * d_dst, unsigned long seed)
{
  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  if (myId >= N)
    return;
  curandState state;
  curand_init ( seed, myId, 0, &state);
  float RANDOM = curand_uniform( &state );
  d_dst[myId] = (size_t)(RANDOM * (N - 0.00001));
}

//! Generate uniform random pick from (0, N-1) with replacement
/*!
Suppose you have a deck of 100 cards, with the numbers 1-100 on one side. You select a card, note the number, replace the card, shuffle, and repeat.
  \param d_dst : destination array to store the result on device.
  \param N: number of elements to pick
  \param maxThreads : max thread per block
  \return 0 if all goes well and -1 otherwise
*/
int generateRandomPicks(size_t * d_dst, size_t N, unsigned int maxThreads){
  if (d_dst == NULL){
    LOG(ERROR) << __func__ << " : destination array must not be empty\n";
    return -1;
  }
  if (N == 0){
    LOG(ERROR) << __func__ << " : the number of elements to pick must be  greater than 0\n";
    return -1;
  }
  if (maxThreads == 0)
  {
    LOG(ERROR) << __func__ << " : the max number of threads per block must be  greater than 0\n";
    return -1;
  }
  int threadN = min((size_t)maxThreads, N);
  int gridSize = ceil(N / (float)threadN);
  pick<<<gridSize, threadN>>>(N, d_dst, unsigned(time(NULL)));
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ <<" : Execution of pick kernel failed : " << cudaGetErrorString(err) << "\n";
    return -1;
  }
  return 0;
}

//! Fill the array with integers in ascending order
__global__ void fillUp(size_t * d_dst, size_t N){
  int myId = blockIdx.x * blockDim.x + threadIdx.x;
  if (myId >= N)
    return;
  d_dst[myId] = myId;
}

//! Fill the array with 0,...,N-1
int fillRange(size_t * d_dst, size_t N, unsigned int maxThreads){
  // Make the element array
  if (d_dst == NULL || N == 0){
    LOG(ERROR) << __func__ <<" : Destination array must not be empty\n";
    return -1;
  }
  int threadN = min((size_t)maxThreads, N);
  int gridSize = ceil(N / (float)threadN);
  fillUp<<<gridSize, threadN>>>(d_dst, N);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ <<" : Execution of fillUp kernel failed : " << cudaGetErrorString(err) << "\n";
    return -1;
  }
  return 0;
}
/*!
  Generate random permutation on GPU
  \param d_dst : destination array to store the result on device.
  \param N: number of elements to pick
  \param maxThreads : max thread per block
  \return 0 if all goes well and -1 otherwise
*/
int permuteElements(size_t * d_dst, size_t N, unsigned int maxThreads){
  if (d_dst == NULL || N == 0){
    LOG(ERROR) << __func__ <<" : Destination array must not be empty\n";
    return -1;
  }
  // Make the key array
  float * d_keys = NULL;
  cudaError_t err = cudaMalloc(&d_keys, sizeof(float) * N);
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ <<" : Could not allocate memory for the keys array : " << cudaGetErrorString(err) << "\n";
    return -1;
  }
  int threadN = min((size_t)maxThreads, N);
  int gridSize = ceil(N / (float)threadN);
  randomKey<<<gridSize, threadN>>>(N, d_keys, unsigned(time(NULL)));
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ <<" : Execution of randomKey kernel failed : " << cudaGetErrorString(err) << "\n";
    return -1;
  }
  //wrap raw pointer with a device_ptr to use with Thrust functions
  thrust::device_ptr<size_t> dthrust_data(d_dst);
  thrust::device_ptr<float> dthrust_keys(d_keys);
  // Sort by keys
  thrust::sort_by_key(dthrust_keys, dthrust_keys + N, dthrust_data);
  cudaFree(d_keys);
  return 0;
}

/*!
  Generate random permutation on CPU
  \param h_dst : destination array to store the result on device.
  \param N: number of elements to pick
  \param maxThreads : max thread per block
  \return 0 if all goes well and -1 otherwise
*/
int permuteElements_host(size_t * h_dst, size_t N){
  if (h_dst == NULL || N == 0){
    LOG(ERROR) << __func__ <<" : Destination array must not be empty\n";
    return -1;
  }
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(h_dst, &h_dst[N-1], std::default_random_engine(seed));
  return 0;
}
