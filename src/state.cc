/*
* state.cc - GPUE2: Split Operator based GPU solver for Nonlinear
* Schrodinger Equation, Copyright (C) 2015, Lee J. O'Riordan.

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

#include "../include/state.hpp"

    addr_grid* State::addr_grid(){
        return &State::addr_grid;
    }

    addr_op* State::addr_op(){
        return &State::addr_op;
    }

    addr_Uop* State::addr_Uop(){
        return &State::addr_Uop;
    }

    double* State::phase(){
        return State::addr_op()->phase;
    }

    double* State::V(){
        return State::addr_op()->V;
    }

    double* K(){
        return State::addr_op()->K;
    }

    double** Q(){
        return State::addr_grid()->Q;
    }

    double** P(){
        return State::addr_grid()->P;
    }

    double* qMax(){
        return State::addr_grid()->qMax;
    }

    double* pMax(){
        return State::addr_grid()->pMax;
    }

    double* dq(){
        return State::addr_grid()->dq;
    }

    double* dp(){
        return State::addr_grid()->dp;
    }

    unsigned int gridMax(){
        return State::addr_grid()->gridMax;
    }

    unsigned int* gridSize(){
        return State::addr_grid()->gridSize;
    }

    unsigned int dim(){
        return State::addr_grid()->dim;
    }
