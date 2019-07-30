
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fileIO.h"
#include "H5Cpp.h"
using namespace H5;

/* HDF5 structure:
 * # data_dir/output.h5
 * /ENERGY
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
 * Uppercase entries are groups,
 * Lowercase entries are datasets,
 * and "i" means the dataset at the i-th iteration
 *
*/

#include "fileIO.h"

namespace FileIO{

    // Output file
    H5File *output;

    // Groups
    Group *energy;
    Group *energy_const;
    Group *energy_ev;
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

    // Data types, stored here for easy reference
    CompType *hdf_double2;
    DataType *hdf_double;
    DataType *hdf_int;
    DataType *hdf_bool;
    StrType *hdf_str;

    // DataSpaces, formats for storage
    DataSpace *wfc_space;
    DataSpace *v_space;
    DataSpace *k_space;
    DataSpace *edge_space;
    DataSpace *a_space;
    DataSpace *x_space;
    DataSpace *y_space;
    DataSpace *z_space;
    DataSpace *attr_space;
    DataSpace *energy_space;

    // DataSet map, for storing and accessing references to datasets without reopening
    std::unordered_map<std::string, DataSet> datasets = {};

    // Initialize the types
    void createTypes() {
        FileIO::hdf_double2 = new CompType(2 * sizeof(double));
        FileIO::hdf_double2->insertMember("re", HOFFSET(double2, x), PredType::NATIVE_DOUBLE);
        FileIO::hdf_double2->insertMember("im", HOFFSET(double2, y), PredType::NATIVE_DOUBLE);

        FileIO::hdf_double = new DataType(PredType::NATIVE_DOUBLE);
        FileIO::hdf_int = new DataType(PredType::NATIVE_INT);
        FileIO::hdf_bool = new DataType(PredType::NATIVE_HBOOL);
        FileIO::hdf_str = new StrType(PredType::C_S1, 256);
    }

    // Initialize the data spaces needed for IO
    void createDataSpaces(int xDim, int yDim, int zDim, int dimnum, int wfc_num) {
        int rank = 1 + dimnum; // number of components x spatial dimensions
        // Create an array of length `rank`, contaniing the dimensionality of the wfc
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

        // Build DataSpaces for each storage object
        FileIO::wfc_space = new DataSpace(rank, dims);
        FileIO::v_space = new DataSpace(rank, dims);
        FileIO::k_space = new DataSpace(rank, dims);
        FileIO::edge_space = new DataSpace(rank, dims);
        FileIO::a_space = new DataSpace(rank, dims);

        // Dimension range shapes are 1 x *Dim
        hsize_t xSize[1] = { (hsize_t)xDim };
        hsize_t ySize[1] = { (hsize_t)yDim };
        hsize_t zSize[1] = { (hsize_t)zDim };
        FileIO::x_space = new DataSpace(1, xSize);
        FileIO::y_space = new DataSpace(1, ySize);
        FileIO::z_space = new DataSpace(1, zSize);

        // Attributes need only a single value
        hsize_t one[1] = { (hsize_t)1 };
        FileIO::attr_space = new DataSpace(1, one);

        hsize_t many[1] = { (hsize_t)wfc_num };
        FileIO::energy_space = new DataSpace(1, many);

        free(dims);
    }

    // Initialize the file output
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
        FileIO::energy = new Group(FileIO::output->createGroup("/ENERGY"));
        FileIO::energy_const = new Group(FileIO::output->createGroup("/ENERGY/CONST"));
        FileIO::energy_ev = new Group(FileIO::output->createGroup("/ENERGY/EV"));

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

        // Initialize types
        FileIO::createTypes();

        // Create DataSpaces
        FileIO::createDataSpaces(xDim, yDim, zDim, dimnum, wfc_num);
    }

    // Iterator for loading a single attribute from the file
    void loadAttr(H5Object &obj, const std::string attr_name, void *op_data) {
        // Unpack the grid from the optional data pass through the attribute iterator
        Grid &par = *((Grid *)op_data);

        // Create the attribute
        Attribute attr = obj.openAttribute(attr_name);
        auto type = attr.getDataType();

        // Unpack and store the value based on its data type
        if (type == *FileIO::hdf_double) {
            double output = 0.0;
            attr.read(type, &output);
            par.store_new(attr_name, output);
        } else if (type == *FileIO::hdf_int) {
            // Make sure that if `*steps` was set, it doesn't get overwritten
            int output = 0;
            attr.read(type, &output);
            std::cout << "Loading attribute " << attr_name << " with value " << output << std::endl;
            par.store_new(attr_name, output);
        } else if (type == *FileIO::hdf_bool) {
            bool output = false;
            attr.read(type, &output);
            par.store_new(attr_name, output);
        } else if (type == *FileIO::hdf_str) {
            std::string output("");
            attr.read(type, output);
            par.store_new(attr_name, output);
        } else {
            std::cout << "ERROR: Attribute " << attr_name << " has invalid DataType!\n";
        }
    }

    // Iterate over the attributes and load them into `par`
    void loadParams(Grid &par) {
        FileIO::output->iterateAttrs(FileIO::loadAttr, NULL, &par);
    }

    // Load the most recent wfc
    void loadWfc(Grid &par) {
        bool gstate = par.bval("gstate");
        int i = par.ival(gstate ? "g_i" : "e_i");
        int xDim = par.ival("xDim");
        int yDim = par.ival("yDim");
        int zDim = par.ival("zDim");
        int gSize = xDim * yDim * zDim;
        int wfc_num = par.ival("wfc_num");

        // Since we don't store booleans in the output we don't have gstate,
        // So we use gsteps, which is set to 0 when groundstate simulation is done
        std::string dataset_name = (gstate ? "/WFC/CONST/" : "/WFC/EV/") + std::to_string(i);

        DataSet latest_wfc = FileIO::output->openDataSet(dataset_name);

        // Load the buffer as contiguous memory and reshape into std::vector<double2 *>
        double2 *wfc_buffer = (double2 *)malloc(wfc_num * gSize * sizeof(double2));

        latest_wfc.read(wfc_buffer, *FileIO::hdf_double2);

        std::vector<double2 *> wfc_array(wfc_num);
        for (int w = 0; w < wfc_num; w++) {
            wfc_array[w] = wfc_buffer + (w * gSize);
        }
        par.store("wfc_array", wfc_array);
    }

    // Load Ax, Ay, and Az from file
    void loadA(Grid &par) {
        bool gstate = par.bval("gstate");
        int i = par.ival(gstate ? "g_i" : "e_i");;
        int gSize = par.ival("gSize");
        int wfc_num = par.ival("wfc_num");
        int dimnum = par.ival("dimnum");

        std::vector<double *> Ax(wfc_num), Ay(wfc_num), Az(wfc_num),
                          Ax_gpu(wfc_num), Ay_gpu(wfc_num), Az_gpu(wfc_num);

        std::string ax_name = "/A/AX/" + std::to_string(par.bval("Ax_time") ? i : 0);
        std::string ay_name = "/A/AY/" + std::to_string(par.bval("Ay_time") ? i : 0);
        std::string az_name = "/A/AZ/" + std::to_string(par.bval("Az_time") ? i : 0);

        // Allocate memory for three buffers, regardless of if they are used
        double *ax_buffer = (double *)malloc(wfc_num * gSize * sizeof(double));
        double *ay_buffer = (double *)malloc(wfc_num * gSize * sizeof(double));
        double *az_buffer = (double *)malloc(wfc_num * gSize * sizeof(double));

        // Clear the buffers in case they are not read in
        for (int j = 0; j < wfc_num * gSize; j++) {
          ax_buffer[j] = 0;
          ay_buffer[j] = 0;
          az_buffer[j] = 0;
        }

        // Read the datasets
        DataSet latest_ax = FileIO::output->openDataSet(ax_name);
        latest_ax.read(ax_buffer, *FileIO::hdf_double);

        if (dimnum > 1) {
            DataSet latest_ay = FileIO::output->openDataSet(ay_name);
            latest_ay.read(ay_buffer, *FileIO::hdf_double);
        }

        if (dimnum > 2) {
            DataSet latest_az = FileIO::output->openDataSet(az_name);
            latest_az.read(az_buffer, *FileIO::hdf_double);
        }

        for (int w = 0; w < wfc_num; w++) {
            // Allocate gpu memory for and copy over the data
            Ax[w] = ax_buffer + (w * gSize);
            cudaHandleError(cudaMalloc((void**) &Ax_gpu[w], sizeof(double)*gSize));
            cudaHandleError(cudaMemcpy(Ax_gpu[w], Ax[w], sizeof(double)*gSize,
                            cudaMemcpyHostToDevice));

            Ay[w] = ay_buffer + (w * gSize);
            cudaHandleError(cudaMalloc((void**) &Ay_gpu[w], sizeof(double)*gSize));
            cudaHandleError(cudaMemcpy(Ay_gpu[w],Ay[w],sizeof(double)*gSize,
                            cudaMemcpyHostToDevice));

            Az[w] = az_buffer + (w * gSize);
            cudaHandleError(cudaMalloc((void**) &Az_gpu[w], sizeof(double)*gSize));
            cudaHandleError(cudaMemcpy(Az_gpu[w],Az[w],sizeof(double)*gSize,
                            cudaMemcpyHostToDevice));
        }

        par.store("Ax", Ax);
        par.store("Ay", Ay);
        par.store("Az", Az);

        par.store("Ax_gpu", Ax_gpu);
        par.store("Ay_gpu", Ay_gpu);
        par.store("Az_gpu", Az_gpu);
    }

    // Open the file and load any immediately-required data
    void load(Grid &par) {
        if (FileIO::output != NULL) {
            std::cout << "Input file cannot be loaded while open!" << std::endl;
            return;
        }

        // Open file
        FileIO::output = new H5File(par.sval("infile"), H5F_ACC_RDWR);

        // Load groups
        FileIO::energy = new Group(FileIO::output->openGroup("/ENERGY"));
        FileIO::energy_const = new Group(FileIO::output->openGroup("/ENERGY/CONST"));
        FileIO::energy_ev = new Group(FileIO::output->openGroup("/ENERGY/EV"));

        FileIO::wfc = new Group(FileIO::output->openGroup("/WFC"));
        FileIO::wfc_const = new Group(FileIO::output->openGroup("/WFC/CONST"));
        FileIO::wfc_ev = new Group(FileIO::output->openGroup("/WFC/EV"));

        FileIO::v = new Group(FileIO::output->openGroup("/V"));
        FileIO::k = new Group(FileIO::output->openGroup("/K"));

        FileIO::vortex = new Group(FileIO::output->openGroup("/VORTEX"));
        FileIO::vortex_edges = new Group(FileIO::output->openGroup("/VORTEX/EDGES"));

        FileIO::a = new Group(FileIO::output->openGroup("/A"));
        FileIO::ax = new Group(FileIO::output->openGroup("/A/AX"));
        FileIO::ay = new Group(FileIO::output->openGroup("/A/AY"));
        FileIO::az = new Group(FileIO::output->openGroup("/A/AZ"));

        FileIO::domain = new Group(FileIO::output->openGroup("/DOMAIN"));
        FileIO::x = new Group(FileIO::output->openGroup("/DOMAIN/X"));
        FileIO::y = new Group(FileIO::output->openGroup("/DOMAIN/Y"));
        FileIO::z = new Group(FileIO::output->openGroup("/DOMAIN/Z"));

        // Create types
        FileIO::createTypes();

        // Load the params
        FileIO::loadParams(par);

        // Load the latest wfc
        FileIO::loadWfc(par);

        FileIO::createDataSpaces(par.ival("xDim"), par.ival("yDim"), par.ival("zDim"), par.ival("dimnum"), par.ival("wfc_num"));

        // Overwrite the read_wfc check so it doesn't get incorrectly loaded from file
        par.store("read_wfc", true);
    }

    // Write an arbitrary attribute to file
    template<typename T>
    void writeAttribute(std::string attribute_name, DataType *hdf_type, T *value, H5Object *target, DataSpace *space=FileIO::attr_space) {
        if (FileIO::output == NULL) {
            std::cout << "Cannot write attribute " << attribute_name << " to closed file!" << std::endl;
            return;
        }

        if (target->attrExists(attribute_name)) {
            target->removeAttr(attribute_name);
        }

        Attribute attr = target->createAttribute(attribute_name, *hdf_type, *space);

        attr.write(*hdf_type, value);
    }

    // Write a contiguous block of data into a dataset
    template<typename T>
    void write1d(std::string dataset_name, DataType *hdf_type, DataSpace *hdf_space, T *data) {
        if (FileIO::output == NULL) {
            std::cout << "Output file is not open!" << std::endl;  
            return;
        }

        DataSet dataset;

        // Check if dataset already exists
        if (FileIO::output->exists(dataset_name)) {
            std::cout << "Overwriting dataset " << dataset_name << std::endl;

            auto item = FileIO::datasets.find(dataset_name);
            // When loading from file, the dataset does not exist in the map
            if (item == FileIO::datasets.end()) {
                dataset = FileIO::output->openDataSet(dataset_name);
            } else {
                dataset = item->second;
            }
        } else {
            std::cout << "Writing dataset " << dataset_name << std::endl;
            // Create a new dataset
            DSetCreatPropList props;
            int rank = hdf_space->getSimpleExtentNdims();
            auto dims = new hsize_t[rank];
            hdf_space->getSimpleExtentDims(dims, NULL);

            // Automatically chunk the dataset into ~sqrt(N) chunks
            for (int i = 0; i < rank; i++) {
                dims[i] = (hsize_t)floor(sqrt((double)dims[i]));
            }
            props.setChunk(rank, dims);

            // Compression factor on the interval [0,9]
            props.setDeflate(6);

            dataset = FileIO::output->createDataSet(dataset_name, *hdf_type, *hdf_space, props);
        }

        dataset.write(data, *hdf_type);
        datasets[dataset_name] = dataset;
    }

    // Wrapper for writing higher-dimensional data, represented as T**
    // where the first array has wfc_num elements, and the second is contiguous with gSize elements
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
            // Escape the extra allocation when wfc_num is 1
            FileIO::write1d(dataset_name, hdf_type, hdf_space, data[0]);
        }
    }

    // Outward-facing write wrappers for each piece of data

    void writeOutEnergy(Grid &par, std::vector<double> energy, int i) {
        bool gstate = par.bval("gstate");
    
        writeAttribute(std::to_string(i), FileIO::hdf_double, energy.data(), gstate ? FileIO::energy_const : FileIO::energy_ev, FileIO::energy_space);
    }

    void writeOutWfc(Grid &par, std::vector<double2 *> wfc, int i) {
        std::string dataset_name = (par.bval("gstate") ? "/WFC/CONST/" : "/WFC/EV/") + std::to_string(i);

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

    // Write the double, int, bool, and string params to file
    void writeOutParams(Grid &par){
        std::cout << "Writing out params" << std::endl;
        for (auto item : par.getDoubleMap()) {
            FileIO::writeAttribute(item.first, FileIO::hdf_double, &item.second, FileIO::output);
        }

        for (auto item : par.getIntMap()) {
            FileIO::writeAttribute(item.first, FileIO::hdf_int, &item.second, FileIO::output);
        }

        for (auto item : par.getBoolMap()) {
            FileIO::writeAttribute(item.first, FileIO::hdf_bool, &item.second, FileIO::output);
        }

        for (auto item : par.getStringMap()) {
            FileIO::writeAttribute(item.first, FileIO::hdf_str, &item.second, FileIO::output);
        }
    }

    // Clean up allocated memory and open files
    void destroy() {
        delete FileIO::output;

        delete FileIO::energy;
        delete FileIO::energy_const;
        delete FileIO::energy_ev;
        delete FileIO::wfc;
        delete FileIO::wfc_const;
        delete FileIO::wfc_ev;
        delete FileIO::v;
        delete FileIO::k;
        delete FileIO::vortex;
        delete FileIO::vortex_edges;
        delete FileIO::a;
        delete FileIO::ax;
        delete FileIO::ay;
        delete FileIO::az;
        delete FileIO::domain;
        delete FileIO::x;
        delete FileIO::y;
        delete FileIO::z;
    
        delete FileIO::hdf_double2;
        delete FileIO::hdf_double;
        delete FileIO::hdf_int;
        delete FileIO::hdf_bool;
        delete FileIO::hdf_str;
    
        delete FileIO::wfc_space;
        delete FileIO::v_space;
        delete FileIO::k_space;
        delete FileIO::edge_space;
        delete FileIO::a_space;
        delete FileIO::x_space;
        delete FileIO::y_space;
        delete FileIO::z_space;
        delete FileIO::attr_space;
        delete FileIO::energy_space;
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
     * Outputs the adjacency matrix to a file
     */
    template<typename T>
    void writeOutAdjMat(std::string file, T *mat, unsigned int *uids,
                        int dim, int step) {
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

    
    void writeOutAdjMat(std::string file, int *mat, unsigned int *uids,
                        int dim, int step){
        writeOutAdjMat(file, mat, uids, dim, step);
    }

    void writeOutAdjMat(std::string file, double *mat, unsigned int *uids,
                        int dim, int step) {
        writeOutAdjMat(file, mat, uids, dim, step);
    }
}
