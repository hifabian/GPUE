
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "H5Cpp.h"
using namespace H5;

#include "../include/fileIO.h"

namespace FileIO{


    void init(Grid &par) {
        // Load file
        H5File output(par.sval("data_dir") + "output.h5", H5F_ACC_TRUNC );
    }

    /*
     * Reads datafile into memory.
     */
    double2* readIn(std::string fileR, std::string fileI,
                        int gSize){
        FILE *f;
        f = fopen(fileR.c_str(),"r");
        int i = 0;
        double2 *arr = (double2*) malloc(sizeof(double2)*gSize);
        double line;
        while(fscanf(f,"%lE",&line) > 0){
            arr[i].x = line;
            ++i;
        }
        fclose(f);
        f = fopen(fileI.c_str(),"r");
        i = 0;
        while(fscanf(f,"%lE",&line) > 0){
            arr[i].y = line;
            ++i;
        }
        fclose(f);
        return arr;
    }

    /*
     * Writes out the parameter file.
     */
    void writeOutParam(Grid &par, std::string file){
        par.write(file);
    }

    /*
     * Writes out double2 complex data files.
     */
    void writeOut(std::string file, double2 *data, int length, int step){
        std::ofstream output;
        output.open(file + "_" + std::to_string(step));
        for (int i = 0; i < length; ++i){
            output << data[i].x << '\n';
        }

        output.close();

        output.open(file + "i_" + std::to_string(step));
        for (int i = 0; i < length; ++i){
            output << data[i].y << '\n';
        }

        output.close();

    }

    /*
     * Writes out double type data files.
     */
    void writeOutDouble(std::string file, double *data, int length, int step){
        std::ofstream output;
        output.open(file + "_" + std::to_string(step));
        for (int i = 0; i < length; ++i){
            output << data[i] << '\n';
        }

        output.close();
    }

    /*
     * Writes out bool type data files.
     */
    void writeOutBool(std::string file, bool *data,int length, int step){
        std::ofstream output;
        output.open(file + "_" + std::to_string(step));
        for (int i = 0; i < length; ++i){
            output << data[i] << '\n';
        }

        output.close();
    }

    /*
     * Writes out int type data files.
     */
    void writeOutInt(std::string file, int *data, int length, int step){
        std::ofstream output;
        output.open(file + "_" + std::to_string(step));
        for (int i = 0; i < length; ++i){
            output << data[i] << '\n';
        }

        output.close();
    }

    /*
     * Writes out int2 data type.
     */
    void writeOutInt2(std::string file, int2 *data, int length, int step){
        std::ofstream output;
        output.open(file + "_" + std::to_string(step));
        for (int i = 0; i < length; ++i){
            output << data[i].x << "," << data[i].y  << '\n';
        }

        output.close();
    }

    /*
     * Writes out tracked vortex data.
     */
    void writeOutVortex(std::string file,
                        std::vector<std::shared_ptr<Vtx::Vortex>> &data,
                        int step){
        std::ofstream output;
        output.open(file + "_" + std::to_string(step));
        for (int i = 0; i < data.size(); ++i){
            output << data[i]->getCoords().x << "," 
                   << data[i]->getCoordsD().x << ","
                   << data[i]->getCoords().y << ","
                   << data[i]->getCoordsD().y << ","
                   << data[i]->getWinding() << '\n';
        }

        output.close();

    }

    /*
     * Opens and closes file. Nothing more. Nothing less.
     */
    int readState(std::string name){
        FILE *f;
        f = fopen(name.c_str(),"r");
        fclose(f);
        return 0;
    }

    /*
     * Outputs the adjacency matrix to a file
     */
    void writeOutAdjMat(std::string file, int *mat, unsigned int *uids,
                        int dim, int step){
        std::ofstream output;
        output.open(file + "_" + std::to_string(step));
        output << "(*";
        for (int i = 0; i < dim; ++i){
            output << uids[i] << ",";
        }

        output << "*)\n";
        output << "{\n";

        for(int i = 0; i < dim; ++i){
            output << "{";
            for(int j = 0; j < dim; ++j){
                output << mat[i*dim + j];
                if(j<dim-1)
                    output << ",";
                else
                    output << "}";
            }
            if(i<dim-1)
                output << ",";
            output << "\n";
        }
        output << "}\n";

        output.close();
    }
    void writeOutAdjMat(std::string file, double *mat, unsigned int *uids,
                        int dim, int step){
        std::ofstream output;
        output.open(file + "_" + std::to_string(step));
        output << "(*";
        for (int i = 0; i < dim; ++i){
            output << uids[i] << ",";
        }

        output << "*)\n";
        output << "{\n";

        for(int i = 0; i < dim; ++i){
            output << "{";
            for(int j = 0; j < dim; ++j){
                output << mat[i*dim + j];
                if(j<dim-1)
                    output << ",";
                else
                    output << "}";
            }
            if(i<dim-1)
                output << ",";
            output << "\n";
        }
        output << "}\n";

        output.close();
    }
}
