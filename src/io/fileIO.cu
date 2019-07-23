
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "H5Cpp.h"
using namespace H5;

/* HDF5 structure:
 * # data_dir/output.h5
 * /
 * /WFC
 * /WFC/CONST
 * /WFC/CONST/i
 * /WFC/EV
 * /WFC/EV/i
 * /V
 * /V/i
 * /K
 * /K/i
 * ...
 * Where "/" is the root,
 * "i" is the dataset for the i-th iteration,
 * and all other "directories" are groups.
 *
*/

#include "fileIO.h"

namespace FileIO{

    H5File *output;

    Group *wfc;
    Group *wfc_const;
    Group *wfc_ev;
    Group *v;
    Group *k;

    CompType *hdf_double2;
    DataType *hdf_double;

    DataSpace *wfc_space;
    DataSpace *v_space;
    DataSpace *k_space;

    void init(Grid &par) {
        int xDim = par.ival("xDim");
        int yDim = par.ival("yDim");
        int zDim = par.ival("zDim");
        int dimnum = par.ival("dimnum");
        int wfc_num = par.ival("wfc_num");

        // In case `init` gets called multiple times
        if (FileIO::output == NULL) {
            std::cout << "Created File!\n\n";
            // Open file
            FileIO::output = new H5File(par.sval("data_dir") + "output.h5", H5F_ACC_TRUNC);

            // Create groups
            FileIO::wfc = new Group(FileIO::output->createGroup("/WFC"));
            FileIO::wfc_const = new Group(FileIO::output->createGroup("/WFC/CONST"));
            FileIO::wfc_ev = new Group(FileIO::output->createGroup("/WFC/EV"));

            FileIO::v = new Group(FileIO::output->createGroup("/V"));
            FileIO::k = new Group(FileIO::output->createGroup("/K"));

            // Initialize composite data type
            FileIO::hdf_double2 = new CompType(2 * sizeof(double));
            FileIO::hdf_double2->insertMember("re", HOFFSET(double2, x), PredType::NATIVE_DOUBLE);
            FileIO::hdf_double2->insertMember("im", HOFFSET(double2, y), PredType::NATIVE_DOUBLE);

            FileIO::hdf_double = new DataType(PredType::NATIVE_DOUBLE);

            // Create DataSpaces
            int rank = 1 + dimnum; // number of components x spatial dimensions
            hsize_t *dims = (hsize_t *)malloc(rank * sizeof(hsize_t));
            dims[0] = wfc_num;
            if (rank > 1) {
              dims[1] = xDim;
            }
            if (rank > 2) {
              dims[2] = yDim;
            }
            if (rank > 3) {
              dims[3] = zDim;
            }

            FileIO::wfc_space = new DataSpace(rank, dims);
            FileIO::v_space = new DataSpace(rank, dims);
            FileIO::k_space = new DataSpace(rank, dims);
        }
    }

    template<typename T>
    void writeNd(Grid &par, std::string dataset_name, DataType *hdf_type, DataSpace *hdf_space, T **data) {
        if (FileIO::output == NULL) {
            return;
        }

        int n = par.ival("wfc_num");
        int gsize = par.ival("xDim") * par.ival("yDim") * par.ival("zDim");

        std::cout << "Writing dataset " << dataset_name << std::endl;
        DataSet dataset = FileIO::output->createDataSet(dataset_name, *hdf_type, *hdf_space);

        T *tmp = (T *)malloc(n * gsize * sizeof(T));
        if (tmp == NULL) {
            std::cout << "ERROR: could not allocate buffer for writing dataset " << dataset_name << std::endl;
            return;
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < gsize; j++) {
                tmp[i * gsize + j] = data[i][j];
            }
        }

        dataset.write(tmp, *hdf_type);
        free(tmp);
    }

    void writeOutWfc(Grid &par, std::vector<double2 *> wfc, int i, bool gstate) {
        std::string dataset_name = (gstate ? "/WFC/EV/" : "/WFC/CONST/") + std::to_string(i);

        FileIO::writeNd(par, dataset_name, FileIO::hdf_double2, FileIO::wfc_space, wfc.data());
    }

    void writeOutV(Grid &par, std::vector<double *> v, int i) {
        std::string dataset_name = "/V/" + std::to_string(i);

        FileIO::writeNd(par, dataset_name, FileIO::hdf_double, FileIO::v_space, v.data());
    }

    void writeOutK(Grid &par, std::vector<double *> k, int i) {
        std::string dataset_name = "/K/" + std::to_string(i);

        FileIO::writeNd(par, dataset_name, FileIO::hdf_double, FileIO::k_space, k.data());
    }

    void destroy() {
      delete FileIO::output;
      FileIO::output = NULL;

      delete FileIO::wfc;
      FileIO::wfc = NULL;
      delete FileIO::wfc_const;
      FileIO::wfc_const = NULL;
      delete FileIO::wfc_ev;
      FileIO::wfc_ev = NULL;
      delete FileIO::v;
      FileIO::v = NULL;
      delete FileIO::k;
      FileIO::k = NULL;

      delete FileIO::hdf_double2;
      FileIO::hdf_double2 = NULL;

      delete FileIO::wfc_space;
      FileIO::wfc_space = NULL;
      delete FileIO::v_space;
      FileIO::v_space = NULL;
      delete FileIO::k_space;
      FileIO::k_space = NULL;
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
