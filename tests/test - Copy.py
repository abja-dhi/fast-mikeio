import os
os.chdir(os.path.dirname(__file__))
import fastmikeio
import numpy as np
from mikecore.DfsuFile import DfsuFile
np.set_printoptions(suppress=True, formatter={'float_kind': '{:.6f}'.format})

dfsu = DfsuFile.Open("Small_Vertical.dfsu")

X = dfsu.X
Y = dfsu.Y
Z = dfsu.Z

print(Z)

from mikecore.DfsuBuilder import DfsuBuilder, DfsuFileType


# import mikeio
# ds = mikeio.read("Small_3D.dfsu")
# print(ds.geometry.element_coordinates[32])
# et = ds.geometry.element_table
# print(et[32])
# nc = ds.geometry.node_coordinates
# print(nc[et[32]])