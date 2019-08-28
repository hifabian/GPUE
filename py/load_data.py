import h5py
import numpy as np

# Get a numpy array for an arbitrary dataset in the h5file
def getVar(path, filename="../data/data.h5", idx=0):
    f = h5py.File(filename, "r")
    dset = f[path]

    if (dset.dtype.name == "void128"):
        return (dset["re"] + (dset["im"] * 1j))[idx]
    else:
        return dset.value[idx]


# Get a numpy array for the wfc_idx-th wave function at iteration i
def getWfc(gstate, wfc_idx, i, filename="../data/data.h5"):
    f = h5py.File(filename, "r")
    # Access the dataset "/WFC/CONST/i" or "/WFC/EV/i"
    dset = f["/WFC/{}/{}".format("CONST" if gstate else "EV", i)]

    # Convert to complex by indexing on real and imaginary partitions
    # And access [wfc_idx] to get the correct shape
    return (dset["re"] + (dset["im"]) * 1j)[wfc_idx]


# Get the collection of attributes
def getParams(filename="../data/data.h5"):
    f = h5py.File(filename, "r")
    return f.attrs

