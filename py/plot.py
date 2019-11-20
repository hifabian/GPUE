import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from argparse import ArgumentParser

from load_data import getWfc

parser = ArgumentParser(description="Plot GPUE output data")

parser.add_argument("-f", default="../data/data.h5",
                    help="Location of the data file, relative to the py/ folder")

parser.add_argument("-r", nargs=3, type=int, default=[
                    0, 1, 1], help="Range to operate on, with arguments `start`, `end`, `step` respectively")

parser.add_argument("-c", default=0, type=int,
                    help="Index (from 0) of component to plot")

parser.add_argument("-i", default="wfc",
                    help="Item to plot: can be one of: \
                    {wfc, wfc_ev, wfc_k, wfc_k_ev, wfc_phase, wfc_phase_ev}")

args = parser.parse_args()

# Plot an arbitrary variable, only once.
def plot_var(path):
    print(0)
    
    data = getVar(path, filename=argc.f)
    data = np.abs(data)

    plt.imshow(data, cmap=cm.jet)
    plt.colorbar()
    plt.show()


# Plot an arbitrary transformation of the wave function
def plot(gstate, f):
    for i in range(*args.r):
        print(i)

        # Get wfc and apply given transformation
        wfc = getWfc(gstate, args.c, i, filename=args.f)
        wfc = f(wfc)

        # Plot
        dims = (1, wfc.shape[0], 1, wfc.shape[1])
        plt.imshow(wfc, extent=dims, cmap=cm.jet)
        plt.colorbar()
        plt.show()


# Absolute value transformation
def norm(wfc):
    return np.abs(wfc) ** 2


# Momentum space transformation
def k(wfc):
    wfc = np.fft.fft2(wfc)
    wfc = np.fft.fftshift(wfc)
    return norm(wfc)


# Phase transformation
def phase(wfc):
    return np.angle(wfc)


# wfc, wfc_ev, wfc_k, wfc_k_ev, wfc_phase, wfc_phase_ev
opts = {
    "wfc": (True, norm),
    "wfc_ev": (False, norm),
    "wfc_k": (True, k),
    "wfc_k_ev": (False, k),
    "wfc_phase": (True, phase),
    "wfc_phase_ev": (False, phase)
}

plot(*opts[args.i])
