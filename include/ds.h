///@endcond
//##############################################################################
/**
 *  @file    ds.h
 *  @author  James R. Schloss (leios) and Lee J. O'Riordan (mlxd)
 *  @date    12/11/2015
 *  @version 0.1
 *
 *  @brief Dastructure for simulation runtime parameters
 *
 *  @section DESCRIPTION
 *      This file holds necessary classes and structs for all GPUE simulations.
 *      EqnNode and EqnNode_gpu are for dynamic parsing and Grid is for 
 *      general use.
 */
 //#############################################################################

#ifndef DS_H
#define DS_H
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <cuda.h>
#include <cufft.h>
#include <typeinfo>
#include <cassert>
#include <iostream>

/*----------------------------------------------------------------------------//
* CLASSES
*-----------------------------------------------------------------------------*/

/**
 * @brief       Struct for an x, y, z position
 * @ingroup     data
 */
struct pos{
    double x, y, z;
};

/**
 * @brief       function pointer type
 * @ingroup     data
 */
typedef double (*fnPtr) (double, double);

/**
 * @brief       Struct to hold the node information for the AST on the CPU
 * @ingroup     data
 */
struct EqnNode{
    double val = 0;
    bool is_dynamic = false;
    char var = '0';

    EqnNode *left, *right;

    int op_num;
    bool has_op = false;
};

/**
 * @brief       Struct to hold the node information for the AST on the GPU
 * @ingroup     data
 */
struct EqnNode_gpu{
    double val = 0;
    bool is_dynamic = false;
    char var = '0';

    int left = -1;
    int right = -1;

    int op_num;
};

/**
 * @brief       Class to hold the variable map and grid information
 * @ingroup     data
 */
class Grid{
    // Here we keep our variable map (unordered for performance)
    // and also grid information. Note that dx dy, and dz are in param_double
    private:
        typedef void (*functionPtrA)(double*, double*, double*, 
                                     double,  double,  double, 
                                     double,  double,  double, 
                                     double, double, double*);
        typedef void (*functionPtrV)(double*, double*, double*, double*,
                                     double*, double*, double*, double*);
        typedef void (*functionPtrwfc)(double*, double*, double*, 
                                       double*, double, double*, double2*);
        std::unordered_map<std::string, int> param_int;
        std::unordered_map<std::string, double> param_double;
        std::unordered_map<std::string, double*> param_dstar;
        std::unordered_map<std::string, std::vector<double>> param_dvec;
        std::unordered_map<std::string, std::vector<double*>> param_dsvec;
        std::unordered_map<std::string, bool> param_bool;
        std::unordered_map<std::string, cufftDoubleComplex*> sobel;
        std::unordered_map<std::string, std::vector<double2*>> param_d2svec;
        std::unordered_map<std::string, std::string> param_string;
        std::unordered_map<std::string, EqnNode_gpu*> param_ast;
        std::unordered_map<std::string, EqnNode> param_ast_cpu;

        // Default values, accessed when the getter for one of these values finds no result
        std::unordered_map<std::string, bool> default_bool = {
            {"read_file", false},
            {"read_wfc", false},
            {"read_a", false},
            {"corotating", false},
            {"gpe", false},
            {"write_it", false},
            {"graph", false},
            {"unit_test", false},
            {"ramp", false},
            {"write_file", true},
            {"found_sobel", false},
            {"energy_calc", false},
            {"use_param_file", false},
            {"cyl_coord", false},
            {"flip", false},
            {"gstate": false}
        };
        std::unordered_map<std::string, int> default_int = {
            {"xDim", 256},
            {"yDim", 256},
            {"zDim", 256},
            {"gsteps", 0},
            {"esteps", 0},
            {"device", 0},
            {"atoms", 1},
            {"printSteps", 100},
            {"kick_it", 0},
            {"ramp_type", 1},
            {"dimnum", 2},
            {"kill_idx", -1},
            {"energy_calc_steps", 0},
            {"wfc_num", 1},
            {"step_offset", 0},
            {"charge", 0},
            {"g_i", 0},
            {"e_i", 0}
        };
        std::unordered_map<std::string, double> default_double = {
            {"gammaY", 1.0},
            {"gdt", 1e-4},
            {"dt", 1e-4},
            {"winding", 0.0},
            {"interaction", 1.0},
            {"laser_power", 0.0},
            {"angle_sweep", 0.0},
            {"x0_shift", 0.0},
            {"y0_shift", 0.0},
            {"z0_shift", 0.0},
            {"sepMinEpsilon", 0.0},
            {"omega", 0.0},
            {"omegaX", 6.283},
            {"omegaY", 6.283},
            {"omegaZ", 6.283},
            {"fudge", 0.0},
            {"mask_2d", 0.0},
            {"box_size", -0.01},
            {"energy_calc_threshold", -1.0},
            {"thresh_const", 1.0}
        };
        std::unordered_map<std::string, std::string> default_string = {
            {"data_dir", "data/"},
            {"param_file", "param.cfg"},
            {"conv_type", "FFT"}
        };

        // List of all strings for parsing into the appropriate param map
        // 1 -> int, 2 -> double, 3 -> double*
        std::unordered_map<std::string, int> id_list;

    // Here we keep the functions to store variables and access grid data
    public:
        dim3 grid, threads;

        // Map for function pointers and keys K and V
        functionPtrV V_fn;
        functionPtrA Ax_fn, Ay_fn, Az_fn;
        functionPtrwfc wfc_fn;

        // placing grid parameters in public for now
        double *x, *y, *z, *xp, *yp, *zp;

        // Function to store sobel_fft operators into the sobel map
        void store(std::string id, cufftDoubleComplex* d2param);

        // Function to store integer into param_int
        void store(std::string id, int iparam);

        // Function to store double into param_double
        void store(std::string id, double dparam);

        // Function to store double* into param_dstar
        void store(std::string id, double *dsparam);

        // Function to store bool into param_bool
        void store(std::string id, bool bparam);

        // Function to store string into data_dir
        void store(std::string id, std::string sparam);

        // Function to store asts into data_dir
        void store(std::string id, EqnNode_gpu *ensparam);

        // Function to store asts into data_dir
        void store(std::string id, EqnNode astparam);

        // Function to store std::vector<double> values
        void store(std::string id, std::vector<double> dvec);

        // Function to store std::vector<double *> values
        void store(std::string id, std::vector<double *> dsvecparam);

        // Function to store std::vector<double2 *> values
        void store(std::string id, std::vector<double2 *> d2svecparam);

        // Function to store int into param_int iff it isn't already there
        void store_new(std::string id, int iparam);

        // Function to store double into param_double iff it isn't already there
        void store_new(std::string id, double dparam);

        // Function to store bool into param_bool iff it isn't already there
        void store_new(std::string id, bool bparam);

        // Function to store string into param_string iff it isn't already there
        void store_new(std::string id, std::string sparam);

        // Function to retrieve default integer value from default_int
        int ival_default(std::string id);

        // Function to retrieve default double value from default_double
        double dval_default(std::string id);

        // Function to retrieve default bool value from default_bool
        bool bval_default(std::string id);

        // Fucntion to retrieve default string value from default_string
        std::string sval_default(std::string id);

        // Function to retrieve integer value from param_int
        int ival(std::string id);

        // Function to retrieve double value from param_double
        double dval(std::string id);

        // Function to retrieve double star values from param_dstar
        double *dsval(std::string id);

        // Function to retrieve std::vector<double> values from param_
        std::vector<double> dvecval(std::string id);

        // Function to retrieve double star values from param_dstar
        std::vector<double *> dsvecval(std::string id);

        // Function to retrieve bool from param_bool
        bool bval(std::string id);

        // Function to retrieve string from param_string
        std::string sval(std::string id);

        // Function to call back the sobel operators
        cufftDoubleComplex *cufftDoubleComplexval(std::string id);

        // Function to retrieve std::vector<double2 *> values from param_d2svec
        std::vector<double2 *> d2svecval(std::string id);

        // Function to call back ast
        EqnNode_gpu *astval(std::string id);

        // Function to call back ast
        EqnNode ast_cpuval(std::string id);

        // Two boolean functions to check whether a string exists in 
        // param_double or param_dstar
        bool is_double(std::string id);
        bool is_dstar(std::string id);
        bool is_ast_gpu(std::string id);
        bool is_ast_cpu(std::string id);

        // Function to print all available variables
        void print_map();

        // function to set A functions
        void set_A_fn(std::string id);

        // function to set V functions
        void set_V_fn(std::string id);

        // function to set V functions
        void set_wfc_fn(std::string id);

        // Key values for operators
        // Note that Vector potential only have a single string for x, y, z
        std::string Kfn, Vfn, Afn, Axfile, Ayfile, Azfile, Wfcfn;

        // Getter for the double map
        std::unordered_map<std::string, double> getDoubleMap();

        // Getter for the integer map
        std::unordered_map<std::string, int> getIntMap();

        // Getter for the boolean map
        std::unordered_map<std::string, bool> getBoolMap();

        // Getter for the string map
        std::unordered_map<std::string, std::string> getStringMap();
};
typedef class Grid Grid;

/**
* @brief        Generates CUFFT plan for 2D simulations
* @ingroup      gpu
*/
void generate_plan_other2d(cufftHandle *plan_fft1d, Grid &par);

/**
* @brief        Generates CUFFT plan for 3D simulations
* @ingroup      gpu
*/
void generate_plan_other3d(cufftHandle *plan_fft1d, Grid &par, int axis);

/**
* @brief        Sets default functions for all fields (A, K, V)
* @ingroup      data
*/
void set_fns(Grid &par);

#endif
