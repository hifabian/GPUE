import h5py
import numpy as np

f = h5py.File("../data/output.h5", "r")


# Get an numpy array for the wfc_idx-th wave function at iteration i
def getWfc(gstate, wfc_idx, i):
    # Access the dataset "/WFC/CONST/i" or "/WFC/EV/i"
    dset = f["/WFC/{}/{}".format("CONST" if gstate else "EV", i)]

    # Convert to complex by indexing on real and imaginary partitions
    # And access [wfc_idx] to get the correct shape
    return (dset["re"] + (dset["im"]) * 1j)[wfc_idx]
