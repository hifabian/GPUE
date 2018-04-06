/*
* state.hpp - GPUE2: GPU Split Operator solver for Nonlinear
* Schrodinger Equation, Copyright (C) 2018, Lee J. O'Riordan, James Schloss
*/

#ifndef STATE_H
#define STATE_H

//###########################################################################################################//
/*
*  These structs define the addresses to the grid arrays and operators respectively.
*/
//###########################################################################################################//
class State{
    private:
        define struct addr_grid {
            unsigned int dim;
            unsigned int *gridSize;
            unsigned int gridMax;
            double *qMax, *pMax, *dq, *dp;//These guys should depend on the size of dim
            double **Q, **P; //First index represents the respective cartesian dimension space. The outer represents the associated value at that point.
        } addr_grid;

        define struct addr_op {
            double *V, *K, *XPy, *YPx;
        } addr_op;

        define struct addr_Uop {
            double2 *wfc, *opV, *opK, *opXPy, *opYPx, *buffer;
            double *phase;
        } addr_Uop;

    public:
        addr_grid* addr_grid();
        addr_op* addr_op();
        addr_Uop* addr_Uop();
        double* phase();
        double* V();
        double* K();
        double** Q();
        double** P();
        double* qMax();
        double* pMax();
        double* dq();
        double* dp();
        unsigned int gridMax();
        unsigned int* gridSize();
        unsigned int dim();

};

#endif
