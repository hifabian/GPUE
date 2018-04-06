# Makefile - GPUE 2e Split Operator solver for Nonlinear
# Schrodinger Equation, Copyright (C) 2018, Lee J. O'Riordan, James Schloss

OS:=	$(shell uname)
ifeq ($(OS),Darwin)
CCGPU		= nvcc
CUDA_LIB	= /usr/local/cuda/lib
CUDA_HEADER	= /usr/local/cuda/include
CFLAGSGPU	= -g -O3 -ccbin /usr/bin/clang --ptxas-options=-v#-save-temps
CFLAGSHOST	= -g -G -O3 -march=native -lcufft -lcudart
GPU_ARCH	= sm_30
else
CCGPU		= nvcc --ptxas-options=-v --compiler-options -Wall#-save-temps
CUDA_LIB	= /usr/local/cuda-5.5/lib64
CUDA_HEADER	= /usr/local/cuda-5.5/include
CFLAGSGPU	= -g -O3 #-malign-double
CFLAGSHOST	= -g -G -O3 -march=native -fopenmp -lcufft -lcudart
GPU_ARCH	= sm_20
endif
CCMPI		= mpic++
CCHOST		= g++
RM		= /bin/rm
INCFLAGS	= -I$(CUDA_HEADER)
LDFLAGS		= -L$(CUDA_LIB)
EXECS		= GPUE

gpu_functions.o: ./src/colonel/gpu_functions.cu ./include/gpu_functions.h
	$(CCGPU) -c ./src/colonel/gpu_functions.cu -o $@ $(INCFLAGS) $(CFLAGS) $(LDFLAGS) $(CHOSTFLAGS) -march=native

host.o: ./src/host.cu ./include/gpu_functions.h ./include/host.h
	$(CCGPU) -c ./src/host.cu -o $@ $(INCFLAGS) $(CFLAGS) $(LDFLAGS) $(CHOSTFLAGS) -march=native -Xcompiler "-fopenmp" -arch=$(GPU_ARCH)

host_minions.o: ./src/host_minions.cc
	$(CCGPU) -c ./src/split_op.cu -o $@ $(INCFLAGS) $(CFLAGS) $(LDFLAGS) $(CHOSTFLAGS) -march=native

main.o: ./include/* gpu_functions.o host.o host_minions.o operators.o state.o
	$(CCHOST) *.o $(LDFLAGS) $(CFLAGSHOST) -o $(EXECS)

operators.o: ./src/operators.cu ./include/operators.h
	$(CCGPU) -c ./src/operators.cu -o $@ $(INCFLAGS) $(CFLAGS) $(LDFLAGS) $(CHOSTFLAGS) -march=native

state.o: ./src/state.cc ./include/state.hpp
	$(CCHOST) -c ./src/state.cc -o $@ $(INCFLAGS) $(CFLAGS) $(LDFLAGS) $(CHOSTFLAGS) -march=native

default:	main
all:		main test

.c.o:
	$(CCGPU) $(INCFLAGS) $(CFLAGS) -c $<

clean:
	@-$(RM) -f r_0 Phi_0 E* px_* py_0* xPy* xpy* ypx* x_* y_* yPx* p0* p1* p2* EKp* EVr* gpot wfc* Tpot 0* V_* K_* Vi_* Ki_* 0i* k s_* si_* *.o *~ PI* $(EXECS) $(OTHER_EXECS) *.dat *.png *.eps *.ii *.i *cudafe* *fatbin* *hash* *module* *ptx test* vort* v_opt*;
