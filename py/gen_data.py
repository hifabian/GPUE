#-------------gen_data.py------------------------------------------------------##
# Purpose: This file will take the data from GPUE and turn it into a bvox file
#          for visualization with blender or a vtk file for visualization with
#          paraview, example usage can be found in the documentation at:
#          https://gpue-group.github.io/data_analysis/python/
#
#------------------------------------------------------------------------------#

import numpy as np
import math

from load_data import getVar, getWfc

# Function to create plot with vtk
def to_vtk(item, xDim, yDim, zDim, output_file):
    outfile = open(output_file, "w")
    outfile.write("# vtk DataFile Version 3.0\n")
    outfile.write("vtkfile\n")
    outfile.write("ASCII\n")
    outfile.write("DATASET STRUCTURED_POINTS\n")
    outfile.write("DIMENSIONS "+str(xDim)+" "+str(yDim)+" "+str(zDim)+"\n")
    outfile.write("ORIGIN 0 0 0\n")
    outfile.write("SPACING 1 1 1\n")
    outfile.write("POINT_DATA " + str(xDim*yDim*zDim) + "\n")
    outfile.write("SCALARS scalars float 1\n")
    outfile.write("LOOKUP_TABLE default\n")
    for i in range(xDim):
        for j in range(yDim):
            for k in range(zDim):
                outfile.write(str(item[i][j][k]) + " ")
            outfile.write('\n')
        outfile.write('\n')
    outfile.close()

# Function to plot wfc with gstate as a variable to modify the type of plot
def wfc_density(filename="../data/data.h5", gstate=True, i=0, comp=0):
    print(i)
    wfc = getWfc(gstate, comp, i, filename)
    wfc = abs(wfc)
    wfc = wfc * wfc
    wfc /= np.max(wfc)
    return wfc

def wfc_phase(filename="../data/data.h5", gstate=True, i=0, comp=0):
    print(i)
    wfc = getWfc(gstate, comp, i, filename)
    wfc = np.angle(wfc)
    minimum = np.min(wfc)
    maximum = np.max(wfc)
    wfc = (wfc - minimum) / (maximum - minimum)
    return wfc

def proj_phase_2d(filename="../data/data.h5", gstate=True, i=0, comp=0):
    print(i)
    wfc = getWfc(gstate, comp, i, filename)
    wfc = np.angle(wfc)
    file = open("./wfc_ph_{}_{}".format(comp, i),'w')
    for k in range(0,wfc.shape[0]):
        for j in range(0,wfc.shape[1]):
            file.write(str(wfc[j][k][wfc.shape[2]//2]) + '\n')
    file.close()

def proj_2d(filename="../data/data.h5", gstate=True, i=0, comp=0):
    print(i)
    wfc = getWfc(gstate, comp, i, filename)
    wfc = abs(wfc)
    wfc = wfc * wfc
    file = open("./Pwfc_{}_{}".format(comp, i),'w')
    for k in range(0,wfc.shape[0]):
        for j in range(0,wfc.shape[1]):
            file.write(str(wfc[k][j][wfc.shape[2]//2]) + '\n')
    file.close()

def proj_k2d(filename="../data/data.h5", gstate=True, i=0, comp=0):
    filename = data_dir + "/wfc_1"
    print(i)
    wfc = getWfc(gstate, comp, i, filename)
    wfc = np.fft.fftshift(np.fft.fftn(wfc))
    wfc = abs(wfc)
    wfc = wfc * wfc
    file = open("./wfc_1",'w')
    for k in range(0,wfc.shape[0]):
        for j in range(0,wfc.shape[1]):
            file.write(str(wfc[j][k][wfc.shape[2]//2]) + '\n')
    file.close()

def var(path, filename="../data/data.h5"):
    data = getVar(path, filename)

    maximum = np.max(data)
    minimum = np.min(data)

    return (data - minimum) / (maximum - minimum)

def var_r2(path1, path2, filename="../data/data.h5"):
    ax = getVar(path1, filename)
    ay = getVar(path2, filename)

    out = np.sqrt(ax**2 + ay**2)

    maximum = np.max(out)
    minimum = np.min(out)

    return (out - minimum) / (maximum - minimum)
    
def proj_var2d(path, outfilename="../data/var", filename="../data/data.h5"):
    data = getVar(path, filename)

    file = open(outfilename, "w")
    for k in range(data.shape[0]):
        for j in range(data.shape[1]):
            file.write(str(data[k][j])+'\n')
    file.close()

def proj_var1d(path, outfilename="../data/var", filename="../data/data.h5"):
    data = getVar(path, filename)

    file = open(outfilename, "w")
    for k in range(data.shape[0]):
            file.write(str(data[k])+'\n')
    file.close()

# function to output the bvox bin data of a matrix
def to_bvox(item, xDim, yDim, zDim, nframes, filename):
    header = np.array([xDim, yDim, zDim, nframes])
    binfile = open(filename, "wb")
    header.astype('<i4').tofile(binfile)
    item.astype('<f4').tofile(binfile)
    binfile.close()

# find Center of Mass of toroidal condensate
def wfc_com(filename="../data/data.h5", gstate=True, i=0, comp=0):
    print(i)
    wfc = getWfc(gstate, comp, i, filename)
    wfc = abs(wfc)
    wfc = wfc * wfc

    # Here we are finding the CoM
    comx = 0
    comy = 0
    sum = 0
    for i in range(wfc.shape[1]//2,wfc.shape[1]):
        for j in range(0,wfc.shape[0]):
            comx += wfc[j][i][wfc.shape[2]//2]*i
            comy += wfc[j][i][wfc.shape[2]//2]*j
            sum += wfc[j][i][wfc.shape[2]//2]

    comx /= sum
    comy /= sum

    return comx, comy

def wfc_com_2d(filename="../data/data.h5", gstate=True, i=0, comp=0):
    print(i)

    wfc = getWfc(gstate, comp, i, filename)
    wfc = abs(wfc)
    wfc = wfc * wfc

    sum = 0
    com_x = 0
    com_y = 0
    for i in range(wfc.shape[1]//2, wfc.shape[1]):
        for j in range(wfc.shape[0]):
            sum += wfc_2d[j,i]
            com_y += j*wfc_2d[j,i]
            com_x += i*wfc_2d[j,i]

    com_y /= sum
    com_x /= sum

    return (com_x, com_y)


def find_angle_2d(filename="../data/data.h5", gstate=True, i=0, comp=0):
    print(i)

    wfc = getWfc(gstate, comp, i, filename)
    wfc = abs(wfc)
    wfc = wfc * wfc

    # find CoM and angle for each y element
    angle = 0
    count = 0
    for j in range(yDim):
        sum = 0
        com = 0
        for i in range(xDim/2, xDim):
            com += i*wfc_2d[j,i]
            sum += wfc_2d[j,i]

        if (sum >= thresh):
            com /= sum

            angle += math.atan2((yDim/2) - j,(com)-(xDim*0.25))
            count += 1

    angle /= count
    return angle

