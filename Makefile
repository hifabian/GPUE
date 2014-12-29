OS:=	$(shell uname)
ifeq ($(OS),Darwin)
CCGPU		= nvcc
CCHOST		= g++
CUDA_LIB	= /usr/local/cuda/lib
CUDA_HEADER	= /usr/local/cuda/include
CFLAGSGPU	= -g -ccbin /usr/bin/clang --ptxas-options=-v#-save-temps
CFLAGSHOST	= -g -G -march=native -lcufft -lcudart
GPU_ARCH	= sm_30
else
CCGPU		= nvcc --ptxas-options=-v --compiler-options -Wall#-save-temps
CCHOST		= g++
CUDA_LIB	= /usr/local/cuda-5.5/lib64
CUDA_HEADER	= /usr/local/cuda-5.5/include
CFLAGSGPU	= -g  #-malign-double
CFLAGSHOST	= -g -G -march=native -fopenmp -lcufft -lcudart
GPU_ARCH	= sm_20
endif

CLINKER		= $(CCGPU)
RM		= /bin/rm
INCFLAGS	= -I$(CUDA_HEADER)
LDFLAGS		= -L$(CUDA_LIB)
EXECS		= gpue # BINARY NAME HERE

gpue: fileIO.o kernels.o split_op.o tracker.o minions.o ds.o
	$(CCHOST) *.o $(LDFLAGS) $(CFLAGSHOST) -o gpue
	#rm -rf ./*.o

split_op.o: ./src/split_op.cu ./include/split_op.h ./include/kernels.h ./include/constants.h ./include/fileIO.h ./include/minions.h Makefile
	$(CCGPU) -c ./src/split_op.cu -o $@ $(INCFLAGS) $(CFLAGS) -Xcompiler "-fopenmp" -arch=$(GPU_ARCH)

kernels.o: ./include/split_op.h Makefile ./include/constants.h ./include/kernels.h ./src/kernels.cu
	$(CCGPU) -c ./src/kernels.cu -o $@ $(INCFLAGS) $(CFLAGSGPU) -arch=$(GPU_ARCH)

fileIO.o: ./include/fileIO.h ./src/fileIO.cc Makefile
	$(CCHOST) -c ./src/fileIO.cc -o $@ $(INCFLAGS) $(CFLAGS) $(LDFLAGS) -Ofast -march=native

tracker.o: ./src/tracker.cc ./include/tracker.h ./include/fileIO.h
	$(CCHOST) -c ./src/tracker.cc -o $@ $(INCFLAGS) $(CFLAGS) $(LDFLAGS) $(CHOSTFLAGS) -Ofast -march=native

default:	gpue
all:		gpue test

.c.o:
	$(CCGPU) $(INCFLAGS) $(CFLAGS) -c $<

clean:
	@-$(RM) -f r_0 Phi_0 E* px_* py_0* xPy* xpy* ypx* x_* y_* yPx* p0* p1* p2* EKp* EVr* gpot wfc* Tpot 0* V_* K_* Vi_* Ki_* 0i* k s_* si_* *.o *~ PI* $(EXECS) $(OTHER_EXECS) *.dat *.png *.eps *.ii *.i *cudafe* *fatbin* *hash* *module* *ptx test* vort* v_opt*;
