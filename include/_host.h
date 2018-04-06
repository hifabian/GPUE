/*
* host.h - GPUE2: Split Operator based GPU solver for Nonlinear 
* Schrodinger Equation, Copyright (C) 2018, Lee J. O'Riordan, James Schloss
*/

//###########################################################################################################//

#ifndef HOST_H 
#define HOST_H 

void allocateMemoryDevice( struct addr_grid *grid, struct addr_Uop *U_op );
void freeMemoryDevice( struct addr_Uop *U_op);
void allocateMemoryHost( unsigned int selection, struct addr_grid *grid, struct addr_op *op, struct addr_Uop *U_op );
void freeMemoryHost( struct addr_op *op, struct addr_Uop *U_ops );
void initHamiltonianGnd( struct addr_grid *grid, struct addr_op *op, struct addr_Uop *U_op );
void initHamiltonianEv( struct addr_grid *grid, struct addr_op *op, struct addr_Uop *Uop );
void defineGrid( struct addr_grid *grid );
void parseArgs( int argc, char **argv );
void splitOp( unsigned int steps, double dt, struct addr_grid grid*, struct addr_op *op, struct addr_Uop *Uop);

#endif

//###########################################################################################################//
