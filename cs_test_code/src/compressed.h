#ifndef COMPRESSED_H
#define COMPRESSED_H
#include <complex.h>
#include <fftw3.h>
#include "wavefunction.h"
#include "sampler.h"

class CCompressor {
  CWaveFunction * residue = NULL;
  CWaveFunction * fullWF2 = NULL;
  CWaveFunction * fullWF = NULL;
  CSampler * sampler = NULL;
  fftw_plan plan;
  fftw_plan planInv;
  void swap(CWaveFunction * in1, CWaveFunction * in2);
public:
  fftw_plan * getPlan(void){
    return &this->plan;
  }
  //! Constructor
  CCompressor(CWaveFunction & y, CSampler & s){
    this->residue = new CWaveFunction(y.getColsSize(), y.getRowSize());
    this->fullWF2 = new CWaveFunction(s.getRange(), s.getRange());
    this->fullWF = new CWaveFunction(s.getRange(), s.getRange());
    this->plan = fftw_plan_dft_2d(s.getRange(), s.getRange(), this->fullWF2->getHostWF(), this->fullWF2->getHostWF(), FFTW_FORWARD, FFTW_MEASURE);
    this->planInv = fftw_plan_dft_2d(s.getRange(), s.getRange(), this->fullWF2->getHostWF(), this->fullWF2->getHostWF(), FFTW_BACKWARD, FFTW_MEASURE);
    this->sampler = &s;
  }
  //! Desctructor
  ~CCompressor(void){
    delete this->residue;
    delete this->fullWF2;
    delete this->fullWF;
    fftw_destroy_plan(this->plan);
    fftw_destroy_plan(this->planInv);
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
  double thresholdSD(CWaveFunction * wf, CWaveFunction * y, size_t k, double epsilon = 0.01, unsigned int maxCnt = 100);
  /*!
    Restricted steepest descent step
    \param wf : wave function to restrict
    \param y : compressed wave function
    \param k : number of largest elements to keep
    \returns modified wf and the worst case change to the original wf
  */
  double restrictedSD(CWaveFunction * wf, CWaveFunction * y, size_t k, char * support);
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
  double rsdProjection(CWaveFunction * wf, CWaveFunction * y, size_t k, double epsilon, unsigned int maxCnt);
};
#endif
