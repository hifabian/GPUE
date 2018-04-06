/*
* state.cc - GPUE2: Split Operator based GPU solver for Nonlinear
* Schrodinger Equation, Copyright (C) 2018, Lee J. O'Riordan, James Schloss
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
