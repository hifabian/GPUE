from gen_data import *

item = wfc_density(filename="../3d_rot/data.h5", gstate=True, i = 100000)
#item = var("/A/AY/0", filename="../3d_rot/data.h5")
print(len(item))
to_vtk(item, 128, 128, 128, "check.vtk")
