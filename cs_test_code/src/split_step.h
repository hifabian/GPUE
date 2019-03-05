#ifndef DEFCSPLITSTEP
#define DEFCSPLITSTEP
#include "wavefunction.h"
#include "hamiltonian.h"

/*!
Class CSplitStep operator
*/
class CSplitStep {
  int maxThreadsPerBlock = 0;
public:
  CHamiltonian * ham = NULL;

  //! Constructor
  CSplitStep(CHamiltonian * h);
  //! Advance one step
  int advanceOneStep(double dt, double curT, CWaveFunction & wf, cufftHandle * &plan);
  //! Time evolve
  CWaveFunction * evolve(double dt, double totalT, CWaveFunction & initWF, unsigned short snapshotNum = 0);
};

#endif
