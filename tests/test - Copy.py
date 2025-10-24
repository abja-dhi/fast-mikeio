import os
os.chdir(os.path.dirname(__file__))
import fastmikeio
import numpy as np
from mikecore.DfsuFile import DfsuFile

fast = fastmikeio.read("Small_3D.dfsu")
print(fast.geometry.et.shape)
fast.close()

# dfsu = DfsuFile.Open("Vertical.dfsu")
# for item in dfsu.ItemInfo:
#     print(item.Name, item)
# Z = dfsu.Z
# print("\n\n\n")
# print(Z.shape)
# z_data_0 = dfsu.ReadItemTimeStep(1, 0).Data
# z_data_1 = dfsu.ReadItemTimeStep(1, 1).Data
# tss_data = dfsu.ReadItemTimeStep(2, 0).Data
# print(z_data_0)
# print(z_data_1)
# dfsu.Close()

from mikecore.DfsuBuilder import DfsuBuilder, DfsuFileType
