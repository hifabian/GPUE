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

//###########################################################################################################//
/*
*  These structs define the addresses to the grid arrays and operators respectively.
*/
//###########################################################################################################//
class State{
    public:
        struct addr_grid {
            unsigned int dim;
            unsigned int *gridSize;
            unsigned int gridMax;
            double *qMax, *pMax, *dq, *dp;//These guys should depend on the size of dim
            double **Q, **P; //First index represents the respective cartesian dimension space. The outer represents the associated value at that point.
        };

        struct addr_op {
            double *V, *K, *XPy, *YPx;
        };

        struct addr_Uop {
            double2 *wfc, *opV, *opK, *opXPy, *opYPx, *buffer;
        };
};
