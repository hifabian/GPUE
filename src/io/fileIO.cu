
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
 * /VORTEX/EDGES/i
 * /A
 * /A/AX
 * /A/AY
 * /A/AZ
 * /DOMAIN
 * /DOMAIN/X
 * /DOMAIN/Y
 * /DOMAIN/Z
 *
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
    Group *vortex;
    Group *vortex_edges;
    Group *a;
    Group *ax;
    Group *ay;
    Group *az;
    Group *domain;
    Group *x;
    Group *y;
    Group *z;

    CompType *hdf_double2;
    DataType *hdf_double;
    DataType *hdf_int;

    DataSpace *wfc_space;
    DataSpace *v_space;
    DataSpace *k_space;
    DataSpace *edge_space;
    DataSpace *a_space;
    DataSpace *x_space;
    DataSpace *y_space;
    DataSpace *z_space;
    DataSpace *attr_space;

    std::unordered_map<std::string, DataSet> datasets = {};

    void init(Grid &par) {
        int xDim = par.ival("xDim");
        int yDim = par.ival("yDim");
        int zDim = par.ival("zDim");
        int dimnum = par.ival("dimnum");
        int wfc_num = par.ival("wfc_num");

        // In case `init` gets called multiple times
        if (FileIO::output != NULL) {
            std::cout << "Output file initialized while open!" << std::endl;
            return;
        }

        // Open file
        FileIO::output = new H5File(par.sval("data_dir") + "output.h5", H5F_ACC_TRUNC);

        // Create groups
        FileIO::wfc = new Group(FileIO::output->createGroup("/WFC"));
        FileIO::wfc_const = new Group(FileIO::output->createGroup("/WFC/CONST"));
        FileIO::wfc_ev = new Group(FileIO::output->createGroup("/WFC/EV"));

        FileIO::v = new Group(FileIO::output->createGroup("/V"));
        FileIO::k = new Group(FileIO::output->createGroup("/K"));

        FileIO::vortex = new Group(FileIO::output->createGroup("/VORTEX"));
        FileIO::vortex_edges = new Group(FileIO::output->createGroup("/VORTEX/EDGES"));

        FileIO::a = new Group(FileIO::output->createGroup("/A"));
        FileIO::ax = new Group(FileIO::output->createGroup("/A/AX"));
        FileIO::ay = new Group(FileIO::output->createGroup("/A/AY"));
        FileIO::az = new Group(FileIO::output->createGroup("/A/AZ"));

        FileIO::domain = new Group(FileIO::output->createGroup("/DOMAIN"));
        FileIO::x = new Group(FileIO::output->createGroup("/DOMAIN/X"));
        FileIO::y = new Group(FileIO::output->createGroup("/DOMAIN/Y"));
        FileIO::z = new Group(FileIO::output->createGroup("/DOMAIN/Z"));

        // Initialize composite data type
        FileIO::hdf_double2 = new CompType(2 * sizeof(double));
        FileIO::hdf_double2->insertMember("re", HOFFSET(double2, x), PredType::NATIVE_DOUBLE);
        FileIO::hdf_double2->insertMember("im", HOFFSET(double2, y), PredType::NATIVE_DOUBLE);

        FileIO::hdf_double = new DataType(PredType::NATIVE_DOUBLE);
        FileIO::hdf_int = new DataType(PredType::NATIVE_INT);

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
        FileIO::edge_space = new DataSpace(rank, dims);
        FileIO::a_space = new DataSpace(rank, dims);

        hsize_t xSize[1] = { (hsize_t)xDim };
        hsize_t ySize[1] = { (hsize_t)yDim };
        hsize_t zSize[1] = { (hsize_t)zDim };
        FileIO::x_space = new DataSpace(1, xSize);
        FileIO::y_space = new DataSpace(1, ySize);
        FileIO::z_space = new DataSpace(1, zSize);

        hsize_t one[1] = { (hsize_t)1 };
        FileIO::attr_space = new DataSpace(1, one);

        free(dims);
    }

    template<typename T>
    void writeAttribute(std::string attribute_name, DataType *hdf_type, T value, H5Object *target) {
        if (FileIO::output == NULL) {
            std::cout << "Cannot write attribute " << attribute_name << " to closed file!" << std::endl;
            return;
        }

        if (target->attrExists(attribute_name)) {
            target->removeAttr(attribute_name);
        }

        Attribute attr = target->createAttribute(attribute_name, *hdf_type, *FileIO::attr_space);

        attr.write(*hdf_type, &value);
    }

    template<typename T>
    void write1d(std::string dataset_name, DataType *hdf_type, DataSpace *hdf_space, T *data) {
        if (FileIO::output == NULL) {
            std::cout << "Output file is not open!" << std::endl;  
            return;
        }

        DataSet dataset;

        if (FileIO::output->exists(dataset_name)) {
            std::cout << "Overwriting dataset " << dataset_name << std::endl;
            dataset = FileIO::datasets[dataset_name];
        } else {
            std::cout << "Writing dataset " << dataset_name << std::endl;
            DSetCreatPropList props;
            int rank = hdf_space->getSimpleExtentNdims();
            auto dims = new hsize_t[rank];
            hdf_space->getSimpleExtentDims(dims, NULL);
            for (int i = 0; i < rank; i++) {
                dims[i] = (hsize_t)floor(sqrt((double)dims[i]));
            }
            props.setChunk(rank, dims);
            props.setDeflate(6);
            dataset = FileIO::output->createDataSet(dataset_name, *hdf_type, *hdf_space, props);
        }


        dataset.write(data, *hdf_type);
        datasets[dataset_name] = dataset;
    }

    template<typename T>
    void writeNd(Grid &par, std::string dataset_name, DataType *hdf_type, DataSpace *hdf_space, T **data) {
        int n = par.ival("wfc_num");
        int gsize = par.ival("xDim") * par.ival("yDim") * par.ival("zDim");

        if (n > 1) {
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

            FileIO::write1d(dataset_name, hdf_type, hdf_space, tmp);

            free(tmp);
        } else {
            FileIO::write1d(dataset_name, hdf_type, hdf_space, data[0]);
        }
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

    void writeOutEdges(Grid &par, std::vector<double *> edges, int i) {
        std::string dataset_name = "/VORTEX/EDGES/" + std::to_string(i);

        FileIO::writeNd(par, dataset_name, FileIO::hdf_double, FileIO::edge_space, edges.data());
    }

    void writeOutAx(Grid &par, std::vector<double *> ax, int i) {
        std::string dataset_name = "/A/AX/" + std::to_string(i);

        FileIO::writeNd(par, dataset_name, FileIO::hdf_double, FileIO::a_space, ax.data());
    }

    void writeOutAy(Grid &par, std::vector<double *> ay, int i) {
        std::string dataset_name = "/A/AY/" + std::to_string(i);

        FileIO::writeNd(par, dataset_name, FileIO::hdf_double, FileIO::a_space, ay.data());
    }

    void writeOutAz(Grid &par, std::vector<double *> az, int i) {
        std::string dataset_name = "/A/AZ/" + std::to_string(i);
      
        FileIO::writeNd(par, dataset_name, FileIO::hdf_double, FileIO::a_space, az.data());
    }

    void writeOutX(double *x, int i) {
        std::string dataset_name = "/DOMAIN/X/" + std::to_string(i);

        FileIO::write1d(dataset_name, FileIO::hdf_double, FileIO::x_space, x);
    }

    void writeOutY(double *y, int i) {
        std::string dataset_name = "/DOMAIN/Y/" + std::to_string(i);

        FileIO::write1d(dataset_name, FileIO::hdf_double, FileIO::y_space, y);
    }

    void writeOutZ(double *z, int i) {
        std::string dataset_name = "/DOMAIN/Z/" + std::to_string(i);

        FileIO::write1d(dataset_name, FileIO::hdf_double, FileIO::z_space, z);
    }

    void writeOutParams(Grid &par){
        std::cout << "Writing out params" << std::endl;
        for (auto item : par.getDoubleMap()) {
            FileIO::writeAttribute(item.first, FileIO::hdf_double, item.second, FileIO::output);
        }

        for (auto item : par.getIntMap()) {
          FileIO::writeAttribute(item.first, FileIO::hdf_int, item.second, FileIO::output);
      }
    }

    void destroy() {
        delete FileIO::output;

        delete FileIO::wfc;
        delete FileIO::wfc_const;
        delete FileIO::wfc_ev;

        delete FileIO::v;
        delete FileIO::k;

        delete FileIO::vortex;
        delete FileIO::vortex_edges;

        delete FileIO::hdf_double2;

        delete FileIO::wfc_space;
        delete FileIO::v_space;
        delete FileIO::k_space;
        delete FileIO::edge_space;
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
