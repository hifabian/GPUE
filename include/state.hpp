/*
* state.hpp - GPUE2: Split Operator based GPU solver for Nonlinear
* Schrodinger Equation, Copyright (C) 2014, Lee J. O'Riordan.

* This library is free software; you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as
* published by the Free Software Foundation; either version 2.1 of the
* License, or (at your option) any later version. This library is
* distributed in the hope that it will be useful, but WITHOUT ANY
* WARRANTY; without even the implied warranty of MERCHANTABILITY or
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
* License for more details. You should have received a copy of the GNU
* Lesser General Public License along with this library; if not, write
* to the Free Software Foundation, Inc., 59 Temple Place, Suite 330,
* Boston, MA 02111-1307 USA
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
