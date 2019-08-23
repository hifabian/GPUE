import h5py
import numpy as np

def write_wfc(data_im, data_real, xDim, yDim, zDim, gstate,
              wfc_idx, i, output_filename="../data/input.h5"):
    f = h5py.File(output_filename, "w")
    lines_real = np.loadtxt(data_real)
    lines_im = np.loadtxt(data_im)
    print(len(lines_real))
    wfc_real = np.reshape(lines_real, (xDim,yDim,zDim));
    wfc_im = np.reshape(lines_im, (xDim,yDim, zDim));
    wfc = wfc_real + 1j * wfc_im

    f["/WFC/{}/{}".format("CONST" if gstate else "EV", i)] = wfc

def write_gauge(input_ax, input_ay, input_az, xDim, yDim, zDim, gstate,
                wfc_idx, i, output_filename="../data/input.h5"):
    f = h5py.File(output_filename, "w")
    lines_ax = np.loadtxt(input_ax)
    lines_ay = np.loadtxt(input_ay)
    lines_az = np.loadtxt(input_az)
    gauge_ax = np.reshape(lines_ax, (xDim,yDim,zDim));
    gauge_ay = np.reshape(lines_ay, (xDim,yDim,zDim));
    gauge_az = np.reshape(lines_az, (xDim,yDim,zDim));

    f["/A/AX/0"] = gauge_ax
    f["/A/AY/0"] = gauge_ay
    f["/A/AZ/0"] = gauge_az


#write_wfc("../data_2D_example/wfc_evi_0", "../data_2D_example/wfc_ev_0",
#          512, 512, 1, False, 0, 0,
#          output_filename="../data_2D_example/input.h5")
write_gauge("/media/james/ExtraDrive1/GPUE/data/Axgauge",
            "/media/james/ExtraDrive1/GPUE/data/Aygauge",
            "/media/james/ExtraDrive1/GPUE/data/Azgauge",
            512, 512, 1, False, 0, 0,
            output_filename="../data_2D_example/input.h5")

