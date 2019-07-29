///@endcond
//##############################################################################
/**
 *  @file    fileIO.h
 *  @author  Lee J. O'Riordan (mlxd)
 *  @date    12/11/2015
 *  @version 0.1
 *
 *  @brief Routines for input and output of simulation data.
 *
 *  @section DESCRIPTION
 *  The functions herein are used to write the simulation data to text-based
 *  files (HDF was planned, but for simplicity I removed it). Data from previous
 *  simulations can also be read into memory.
 */
 //##############################################################################

#ifndef FILEIO_H
#define FILEIO_H
#include "../include/ds.h"
#include "../include/tracker.h"
#include "../include/split_op.h"
#include <vector>
#include <string>

/** Check source file for further information on functions **/
namespace FileIO {

    /**
     * @brief Initialize the file output
     * @ingroup helper
     * 
     * @param	Grid class
     */
    void init(Grid &par);

    /**
     * @brief Load data from file and continue simulation
     * @ingroup helper
     * 
     * @param	Grid class
     */
    void load(Grid &par);

    /**
     * @brief Load Ax, Ay, and Az from file
     * @ingroup helper
     * 
     * @param	Grid class
     */
    void loadA(Grid &par);

    /**
     * @brief Write the wfc to file
     * @ingroup helper
     * 
     * @param	Grid class
     * @param Data to write
     * @param iteration number
     */
    void writeOutWfc(Grid &par, std::vector<double2 *> wfc, int i);

    /**
     * @brief Write V to file
     * @ingroup helper
     * 
     * @param	Grid class
     * @param Data to write
     * @param iteration number
     */
    void writeOutV(Grid &par, std::vector<double *> v, int i);

    /**
     * @brief Write K to file
     * @ingroup helper
     * 
     * @param	Grid class
     * @param Data to write
     * @param iteration number
     */
    void writeOutK(Grid &par, std::vector<double *> k, int i);

    /**
     * @brief Write the Edges found to file
     * @ingroup helper
     * 
     * @param	Grid class
     * @param Data to write
     * @param iteration number
     */
    void writeOutEdges(Grid &par, std::vector<double *> edges, int i);

    /**
     * @brief Write Ax to file
     * @ingroup helper
     * 
     * @param	Grid class
     * @param Data to write
     * @param iteration number
     */
    void writeOutAx(Grid &par, std::vector<double *> ax, int i);

    /**
     * @brief Write Ay to file
     * @ingroup helper
     * 
     * @param	Grid class
     * @param Data to write
     * @param iteration number
     */
    void writeOutAy(Grid &par, std::vector<double *> ay, int i);

    /**
     * @brief Write Az to file
     * @ingroup helper
     * 
     * @param	Grid class
     * @param Data to write
     * @param iteration number
     */
    void writeOutAz(Grid &par, std::vector<double *> az, int i);

    /**
     * @brief Write the scale in the x dimension to file
     * @ingroup helper
     * 
     * @param Data to write
     * @param iteration number
     */
    void writeOutX(double *x, int i);

    /**
     * @brief Write the scale in the y dimension to file 
     * @ingroup helper
     * 
     * @param	Data to write
     * @param iteration number
     */
    void writeOutY(double *y, int i);

    /**
     * @brief Write the scale in the z dimension to file
     * @ingroup helper
     * 
     * @param Data to write
     * @param iteration number
     */
    void writeOutZ(double *z, int i);

    /**
     * @brief Write the double, int, bool, and string params to file
     * @ingroup helper
     * 
     * @param	Grid class
     */
    void writeOutParams(Grid &par);

    /**
     * @brief Delete created objects, deallocate memory, and close open files
     * @ingroup helper
     */
    void destroy();

	/**
    * @brief	Writes the specified int array to a text file
    * @ingroup	helper
    *
    * @param	*file Name of data file name for saving to
    * @param	*data int array to be written out
    * @param	length Overall length of the file to write out
    * @param	step Index for the filename. file_step
    */
    void writeOutInt(std::string file, int *data, int length, int step);

	/**
    * @brief	Writes the specified Vtx::Vortex array to a text file
    * @ingroup	helper
    *
    * @param	*file Name of data file name for saving to
    * @param	*data Vtx::Vortex array to be written out
    * @param	step Index for the filename. file_step
    */
    void writeOutVortex(std::string file,
                        std::vector<std::shared_ptr<Vtx::Vortex>> &data,
                        int step);

	/**
    * @brief	Write adjacency matrix to a file in Mathematica readable format
    * @ingroup	graph
    *
    * @param	*file Name of data file name for saving to
	* @param	*mat Int Array holding the parameter values to be written out
	* @param	*uids UID array
	* @param	dim Dimension/length of the grid (xDim*yDim)
	* @param	step Index for the filename.
    */
    void writeOutAdjMat(std::string file, int *mat, unsigned int *uids,
                        int dim, int step);

	/**
    * @brief	Write adjacency matrix of doubles to a file in Mathematica readable format
    * @ingroup	graph
    *
    * @param	*buffer Char buffer for use by function internals. char[100] usually
    * @param	*file Name of data file name for saving to
	* @param	*mat double Array holding the parameter values to be written out
	* @param	*uids UID array
	* @param	dim Dimension/length of the grid (xDim*yDim)
	* @param	step Index for the filename.
    */
    void writeOutAdjMat(std::string file, double *mat, unsigned int *uids,
                        int dim, int step);
}
#endif
