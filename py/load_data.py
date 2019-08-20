import h5py
import numpy as np


# Get an numpy array for the wfc_idx-th wave function at iteration i
def getWfc(gstate, wfc_idx, i, filename="../data/output.h5"):
    f = h5py.File(filename, "r")
    # Access the dataset "/WFC/CONST/i" or "/WFC/EV/i"
    dset = f["/WFC/{}/{}".format("CONST" if gstate else "EV", i)]

    # Convert to complex by indexing on real and imaginary partitions
    # And access [wfc_idx] to get the correct shape
    return (dset["re"] + (dset["im"]) * 1j)[wfc_idx]


# Get the collection of attributes
def getParams(filename="../data/output.h5"):
    f = h5py.File(filename, "r")
    return f.attrs

