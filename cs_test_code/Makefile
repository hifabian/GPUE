CC=nvcc
FLAGS= -std=c++11 -I/home/t/thearokke/myapps/include -Wno-deprecated-gpu-targets -L/home/t/thearokke/myapps/lib -lgsl -lgslcblas -lcufft -lfftw3 -lm -O3
DEPLOGH = src/easyloggingcpp/easylogging++.h src/logConfig.config src/logConfigTEST.config
SRCMAIN = src/main.cpp
TESTWFS = src/unittest/buschStateEn1_7.csv src/unittest/buschStateEn1.csv src/unittest/expVImag.csv src/unittest/expVFFTImag.csv src/unittest/expVFFTReal.csv src/unittest/expVReal.csv src/unittest/expV1stepReal.csv src/unittest/expV1stepImag.csv
DEPSRC = src/easyloggingcpp/easylogging++.cc src/mxutils.cu src/wavefunction.cu src/split_step.cu src/twoParticlesInHO_Ham.cu src/compressed.cu src/cs_utils.cu src/sampler.cu
DEPH = src/mxutils.h src/wavefunction.h src/hamiltonian.h src/twoParticlesInHO_Ham.h src/split_step.h src/compressed.h src/cs_utils.h src/sampler.h
TESTSRC = src/unittest/testMain.cu src/unittest/mxutilsTest.cu src/unittest/wavefunctionTest.cu src/unittest/hamiltonianTest.cu src/unittest/split_stepTest.cu src/unittest/csutilsTest.cu src/unittest/samplerTest.cu src/unittest/compressorTest.cu
TESTH = src/unittest/catch.hpp
OBJ = tevol.out
TESTOBJ = unitTest.out
LOGDIR = "logs/"
mainobj = easylogging++.o mxutils.o wavefunction.o split_step.o twoParticlesInHO_Ham.o compressed.o cs_utils.o sampler.o
utestobj = testMain.o mxutilsTest.o wavefunctionTest.o compressorTest.o hamiltonianTest.o split_stepTest.o csutilsTest.o samplerTest.o

.PHONY : cleanbuild pullbuild build cleanlogs o testlink test

test: $(TESTSRC) $(TESTH) $(DEPLOGH) $(DEPSRC) $(SRCMAIN) $(TESTWFS) $(DEPH)
		rm -f *.o $(TESTOBJ)
		$(CC) -x cu -I. -dc $(FLAGS) $(TESTSRC) $(DEPSRC)
		$(CC) $(FLAGS) $(mainobj) $(utestobj) -o $(TESTOBJ)
		./$(TESTOBJ)

build : $(DEPSRC) $(DEPLOGH) $(SRCMAIN) $(DEPH)
	mkdir -p $(LOGDIR)
	$(CC) $(FLAGS)  $(DEPSRC) $(SRCMAIN) -o $(OBJ)

cleanlogs :
	rm -f $(LOGDIR)/*.log

cleanbuild :
	rm -f *.out *.o

pullbuild :
	git pull
	make cleanbuild
	make build
