#include <cufft.h>
#include <chrono>
#include "mxutils.h"
#include "split_step.h"
#include "easyloggingcpp/easylogging++.h"

//! Constructor
CSplitStep::CSplitStep(CHamiltonian * h){
  this->ham = h;
  if (this->ham == NULL)
  {
    LOG(ERROR) << __func__ << " : The hamiltonian cannot be NULL\n";
    return;
  }
  // Get cuda properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  // Maximum threads per block on this device
  this->maxThreadsPerBlock = prop.maxThreadsPerBlock;
}

__global__ void g_calcExpKPhi(double half_i_dt, cufftDoubleComplex * d_expK_phi, double * d_px, size_t vSz){
  // First we need to find our global threadID
  int tPosX = blockIdx.x * blockDim.x + threadIdx.x;
  int tPosY = blockIdx.y * blockDim.y + threadIdx.y;
  if (tPosY >= vSz || tPosX >= vSz)
    return;
  int idx = tPosY * vSz + tPosX;
  cufftDoubleComplex curVal = d_expK_phi[idx];
  double a = curVal.x;
  double b = curVal.y;
  // Calculate the phase value
  double pxx = d_px[tPosX];
  double pxy = d_px[tPosY];
  double rho = half_i_dt * (pxx * pxx + pxy * pxy);
  double c = cos(rho);
  double s = sin(rho);
  d_expK_phi[idx].x = c * a - s * b;
  d_expK_phi[idx].y = a * s + b * c;
}

__global__ void g_calcExpVPhi(hamFunc * df_V,  hamFunc * df_U, hamFunc * df_Nonl, void * d_par, double T, double half_i_dt, cufftDoubleComplex * d_expV_phi, double * d_x, size_t vSz){
  // First we need to find our global threadID
  int tPosX = blockIdx.x * blockDim.x + threadIdx.x;
  int tPosY = blockIdx.y * blockDim.y + threadIdx.y;
  if (tPosY >= vSz || tPosX >= vSz)
    return;
  int idx = tPosY * vSz + tPosX;
  double c = d_expV_phi[idx].x;
  double d = d_expV_phi[idx].y;
  // Calculate the phase value
  //printf("(%u : %.3f)", tPosX, (*df_V)(d_x[tPosX], T, d_par));
  double rho = (*df_V)(d_x[tPosX], T, d_par) + (*df_V)(d_x[tPosY], T, d_par) + (*df_U)(abs(d_x[tPosX] - d_x[tPosY]), T, d_par) + (*df_Nonl)(sqrt(c * c + d * d), T, d_par);
  // exp(i b) = cos(b) + i * sin(b)
  double a = cos(rho * half_i_dt);
  double b = sin(rho * half_i_dt);
  // (a + i b)* (c + i d) = (ac - bd) + i (ad + bc)
  d_expV_phi[idx].x = a * c - b * d;
  d_expV_phi[idx].y = a * d + b * c;
}

//! Advance one step temp1=expV.*ifft2(expK.*fft2(expV.*temp1))
//! NOTE: The resulting wave function is not going to be normalized
int CSplitStep::advanceOneStep(double dt, double curT, CWaveFunction & wf, cufftHandle * &plan){
  cufftDoubleComplex * d_wf = wf.getDeviceWF();
  if (d_wf == NULL || wf.getColsSize() == 0 || wf.getRowSize() == 0)
  {
    LOG(ERROR) << __func__ << " : Device wave function is empty\n";
    return -1;
  }
  size_t vSize = wf.getColsSize();
  // Maximum threads per block on this device
  int maxThreads = this->maxThreadsPerBlock;
  // Call the kernel to generate expV
  int blockSzX = min((size_t)maxThreads, vSize);
  dim3 threadsPerBlock(blockSzX, 1);
  dim3 grid(ceil(vSize/blockSzX), wf.getRowSize());
  // A = expV .* phi
  g_calcExpVPhi<<<grid, threadsPerBlock>>>(this->ham->timeDepPotential(), this->ham->timeDepInteraction(), this->ham->timeDepNonLin(), this->ham->getParams(), curT+dt, -0.5 * dt, d_wf, this->ham->getDeviceX(), vSize);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << " : could not execute g_calcExpVPhi kernel (first call): " << cudaGetErrorString(err) << ", time T: " << curT << "\n";
    return -1;
  }
  // fft2(A)
  if (doFFT(d_wf, wf.getColsSize(), wf.getRowSize(), plan, true) != 0)
  {
    LOG(ERROR) << __func__ << " : could not perform forward FFT\n";
    return -1;
  }
  //B = expK .* fft2(A)
  g_calcExpKPhi<<<grid, threadsPerBlock>>>(-0.5 * dt, d_wf, this->ham->getDeviceMom(), vSize);
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << " : could not execute g_calcExpKPhi kernel : " << cudaGetErrorString(err) << ", time T: " << curT << "\n";
    return -1;
  }
  // ifft(B)
  if (doFFT(d_wf, wf.getColsSize(), wf.getRowSize(), plan, false) != 0)
  {
    LOG(ERROR) << __func__ << " : could not perform inverse FFT\n";
    return -1;
  }
  // C = expV.*ifft(B)
  g_calcExpVPhi<<<grid, threadsPerBlock>>>(this->ham->timeDepPotential(), this->ham->timeDepInteraction(), this->ham->timeDepNonLin(), this->ham->getParams(), curT+dt, -0.5 * dt, d_wf, this->ham->getDeviceX(), vSize);
  err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    LOG(ERROR) << __func__ << " : could not execute g_calcExpVPhi kernel (second call): " << cudaGetErrorString(err) << ", time T: " << curT << "\n";
    return -1;
  }
  return 0;
}

/*!
  Evolve the wave function in time using time-depended Hamiltonian specified in the operator
  \param dt : time step (dimensionless units)
  \param totalT : total time (dimensionless units)
  \param initWF : initial wave function
  \param snapshotNum : number of intermediate snapshots of the time evolution apart from the initial and the final wfs
  \return array of wavefunctions of size (snapshotNum + 2). The first element is the initial wave function, the last element is the wave function at the end of the time evolution
*/
CWaveFunction * CSplitStep::evolve(double dt, double totalT, CWaveFunction & initWF, unsigned short snapshotNum){
  if (dt <= 0 || totalT <= 0)
  {
    LOG(ERROR) << __func__ << " : dt and total time must be greater than zero\n";
    return NULL;
  }
  CWaveFunction & wf = initWF;
  if(wf.copyToGPU() != 0)
  {
    LOG(ERROR) << __func__ << " : Cannot copy wave function to GPU\n";
    return NULL;
  }
  CWaveFunction * snapshots = new CWaveFunction[2 + snapshotNum];
  initWF.copyFromGPU();
  snapshots[0].copy(initWF);
  unsigned int sI = 1;
  unsigned int timeStepN = (unsigned int) ceil(totalT / dt);
  unsigned int snapStep = timeStepN / snapshotNum;
  double curT = 0.0;
  cufftHandle * plan = NULL;
  for (unsigned int i = 0; i < timeStepN; i++)
  {
    if (this->advanceOneStep(dt, curT, wf, plan) != 0)
    {
      LOG(ERROR) << __func__ << " : time evolution step failed at step number " << i << std::endl;
      return snapshots;
    }
    normalize(wf.getDeviceWF(), wf.getColsSize(), this->ham->getCoordStep());
    if (i > 0 && i % snapStep == 0)
    {
      wf.copyFromGPU();
      snapshots[sI].copy(wf);
      sI++;
    }
  }
  normalize(wf.getDeviceWF(), wf.getColsSize(), this->ham->getCoordStep());
  wf.copyFromGPU();
  snapshots[1 + snapshotNum].copy(wf);
  return snapshots;
}
