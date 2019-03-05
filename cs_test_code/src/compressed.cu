#include <iostream>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include "compressed.h"
#include "mxutils.h"
#include "cs_utils.h"
#include "sampler.h"

#define BIGMU 10

void CCompressor::swap(CWaveFunction * in1, CWaveFunction * in2) {
  size_t N = in1->getRowSize() * in1->getColsSize();
  double tmp;
  for (size_t i = 0; i < N; i++)
  {
    tmp = in1->getHostWF()[i][0];
    in1->getHostWF()[i][0] = in2->getHostWF()[i][0];
    in2->getHostWF()[i][0] = tmp;
    tmp = in1->getHostWF()[i][1];
    in1->getHostWF()[i][1] = in2->getHostWF()[i][1];
    in2->getHostWF()[i][1] = tmp;
  }
}


/*!
  Restricted steepest descent step
  \param wf : wave function to restrict (freq or space domain)
  \param y : compressed wave function
  \param k : number of largest elements to keep
  \returns modified wf (freq domain) and the worst case change to the original wf
*/
double CCompressor::restrictedSD(CWaveFunction * wf, CWaveFunction * y, size_t k, char * support){
  if (wf == NULL || y == NULL || support == NULL)
  {
    LOG(ERROR) << __func__ << " : The input wave functions and the support cannot be NULL. \n";
    return -1.0;
  }
  if (wf->getDomain() != false)
  {
    LOG(ERROR) << __func__ << " : The input wave functions has to be in frequency domain";
    return -1.0;
  }
  if (y->getDomain() != true)
  {
    LOG(ERROR) << __func__ << " : Reference compressed vector y has to be in space domain";
    return -1.0;
  }
  double zeroCutoff = pow(10,-23);
  // The wf is in freq domain => wf = vec, our A = Sample * IFFT, A* = FFT * Upsample
  // A(vec)
  if (this->sampler->applySamplingOperator(wf, this->residue) != 0)
  {
    LOG(ERROR) << __func__ << " : Could not sample the wave function. \n";
    return -1.0;
  }
  // residue = A(vec) (space)
  // residue = y - A(vec)
  this->residue->subtract_host(*y, true);
  LOG(INFO) << " Resid :" << this->residue->getHostWF()[32768][0] << "+ i" << this->residue->getHostWF()[32768][1] << ", " << this->residue->getHostWF()[32768][0] << " + i" << this->residue->getHostWF()[32768][1];
  if(this->sampler->applyInverseSamplingOperator(this->residue, this->fullWF2) != 0)
  {
    LOG(ERROR) << __func__ << " : Could not A* the residue. \n";
    return -1.0;
  }
  LOG(INFO) << " Grad :" << this->fullWF2->getHostWF()[524288-1][0] << "+ i" << this->fullWF2->getHostWF()[524288-1][1] << ", " << this->fullWF2->getHostWF()[524288+3][0] << " + i" << this->fullWF2->getHostWF()[524288+3][1];
  // fullWF2 = grad = A*(residue)(freq)
  this->fullWF->copy(*this->fullWF2);
  // fullWF = grad (freq)
  thresholdToSupport(this->fullWF2->getHostWF(), support, this->sampler->getRange() * this->sampler->getRange());
  // fullWF2 = thresholded A* (residue) (freq)
  double mu1 = this->fullWF2->norm2sqr();
  // fullWF2 = thresholded grad, fullWF = grad
  if (this->sampler->applySamplingOperator(this->fullWF2, this->residue) != 0)
  {
    LOG(ERROR) << __func__ << " : Could not sample the wave function. \n";
    return -1.0;
  }
  // residue = A (grad_thresh)
  // residue is in space domain
  double mu2 = this->residue->norm2sqr();
  LOG(INFO) << "mu1:" << mu1 << " mu2: " << mu2;
  double mu = (mu2 < zeroCutoff ? BIGMU : mu1 / mu2);
  if (wf->getDomain() == true)
    wf->switchDomain();
  // wf = vec
  (*this->fullWF) *= mu;
  (*wf) += (*this->fullWF);
  //LOG(INFO) << " mu: " << mu;
  // wf = vec + mu * grad, fullWF = mu * grad
  return 2.0 * this->fullWF->maxAbs();
}

/*!
  Project y onto the subspace defined by k max values
  Save this new projection into wf
  \param wf : wave function to restrict
  \param y : compressed wave function
  \param k : number of largest elements to keep
  \param epsilon : Precision > 0.0
  \param maxCnt : maximum iterations
  \return obtained realtive convergence before stopping (difference between maximum abs values of the scaled gradients at last two steps)
*/
double CCompressor::rsdProjection(CWaveFunction * wf, CWaveFunction * y, size_t k, double epsilon, unsigned int maxCnt){
  if (wf == NULL || y == NULL)
  {
    LOG(ERROR) << __func__ << " : The input wave functions cannot be NULL. \n";
    return -1;
  }
  if (wf->getRowSize() < this->sampler->getRange() || wf->getColsSize() < this->sampler->getRange())
  {
    LOG(ERROR) << __func__ << " : The output wave functions cannot fit the upsampled vector. \n";
    return -1;
  }
  if (epsilon <= 0)
  {
    LOG(ERROR) << __func__ << " : Precision epsilon must be greater than zero \n";
    return -1;
  }
  LOG(INFO) << "y: " << y->getHostWF()[0][0] << " " << y->getHostWF()[0][1];
  if(this->sampler->applyInverseSamplingOperator(y, wf) != 0)
  {
    LOG(ERROR) << __func__ << " : Could not upsample y.";
    return -1;
  }
  //LOG(INFO) << "wf: " << wf->getHostWF()[0][0] << " " << wf->getHostWF()[0][1];
  // wf = A*(y) (freq)
  char * support = findSupport_sort_host(wf->getHostWF(), this->sampler->getRange() * this->sampler->getRange(), k, 0);
  // wf = thresholded A* (y) (freq)
  int cnt = 0;
  double maxChange = 2 * epsilon;
  while (cnt < maxCnt && maxChange > epsilon)
  {
    maxChange = restrictedSD(wf, y, k, support);
    support = findSupport_sort_host(wf->getHostWF(), this->sampler->getRange() * this->sampler->getRange(), k, 0);
    //thresholdToSupport(wf->getHostWF(), support, this->sampler->getRange() * this->sampler->getRange());
    cnt++;
  }
  delete [] support;
  return maxChange;
}

/*!
  Perform hard thresholding with steepest descent
  Save the result into wf in space domain
  \param wf : wave function to restrict
  \param y : compressed wave function
  \param k : number of largest elements to keep
  \param epsilon : Precision > 0.0
  \param maxCnt : maximum iterations
  \return obtained realtive convergence before stopping (difference between maximum abs values of the scaled gradients at last two steps)
*/
double CCompressor::thresholdSD(CWaveFunction * wf, CWaveFunction * y, size_t k, double epsilon, unsigned int maxCnt){
  double normEps = epsilon * y->norm2sqr() / (double)(y->getColsSize() * y->getRowSize());
  return this->rsdProjection(wf, y, k, normEps, maxCnt);
}
