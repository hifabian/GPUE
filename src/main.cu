#include "init.h"
#include "dynamic.h"
#include "split_op.h"

int main(int argc, char **argv){

    Grid par = parseArgs(argc,argv);

    int device = par.ival("device");
    int dimnum = par.ival("dimnum");
    int wfc_num = par.ival("wfc_num");
    cudaHandleError(cudaSetDevice(device));

    time_t start,fin;
    time(&start);
    printf("Start: %s\n", ctime(&start));

    //************************************************************//
    /*
    * Initialise the Params data structure to track params and variables
    */
    //************************************************************//

    // If we want to read in a wfc, we may also need to imprint a phase. This
    // will be done in the init_2d and init_3d functions
    // We need a number of parameters for now
    int xDim = par.ival("xDim");
    int yDim = par.ival("yDim");
    int zDim = par.ival("zDim");
    if(par.bval("read_wfc") == true){

        // Initializing the wfc
        int gSize = xDim * yDim * zDim;
        std::vector<double2 *> wfc_array(wfc_num);

        std::string infile = par.sval("infile");
        std::string infilei = par.sval("infilei");
        printf("Loading wavefunction...");
        wfc_array[0]=FileIO::readIn(infile,infilei,gSize);
        par.store("wfc_array",wfc_array);
        printf("Wavefunction loaded.\n");
        //std::string data_dir = par.sval("data_dir");
        //FileIO::writeOut(data_dir + "WFC_CHECK",wfc_array,gSize,step_offset);
    }

    init(par);

    int gsteps = par.ival("gsteps");
    int esteps = par.ival("esteps");
    std::string data_dir = par.sval("data_dir");
    std::cout << "variables re-established" << '\n';

    if (par.bval("write_file")){
        FileIO::writeOutParam(par, data_dir + "Params.dat");
    }

    if(gsteps > 0){
        std::cout << "Imaginary-time evolution started..." << '\n';
        set_variables(par, 0);

        evolve(par, gsteps, 0);
    }

    if(esteps > 0){
        std::cout << "real-time evolution started..." << '\n';
        set_variables(par, 1);
        evolve(par, esteps, 1);
    }

    // Close the output file
    FileIO::destroy();

    std::cout << "done evolving" << '\n';
    time(&fin);
    printf("Finish: %s\n", ctime(&fin));
    printf("Total time: %ld seconds\n ",(long)fin-start);
    std::cout << '\n';
    return 0;
}