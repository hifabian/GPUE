/*
* operators.h - GPUE2: GPU Split Operator solver for Nonlinear
* Schrodinger Equation, Copyright (C) 2018, Lee J. O'Riordan, James Schloss
*/

//###########################################################################################################//

#ifndef OPERATORS_H
#define OPERATORS_H

double* generate_V(struct addr_grid *grid, double mass, double[] omega);
double* generate_K(struct addr_grid *grid, double mass);
double2* generate_gndOperator(double *operator, struct addr_grid *grid, double dt_hbar);
double2* generate_evOperator(double *operator, struct addr_grid *grid, double dt_hbar);

#endif

//###########################################################################################################//
