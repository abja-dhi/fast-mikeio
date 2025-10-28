import os
os.chdir(os.path.dirname(__file__))
import fastmikeio
from fastmikeio import Dfsu
import numpy as np
from mikecore.DfsFile import DfsFile
from mikecore.DfsuBuilder import DfsuBuilder, DfsuFileType
from mikecore.DfsFactory import DfsFactory
from scipy.interpolate import griddata







    



if __name__ == "__main__":
    # dfsu = fastmikeio.read("3D.dfsu")
    # x = [-117.2482047163, -117.2464082003]
    # y = [10.33932763618, 10.53730369851]
    # dfsu.vertical_extractor(x, y, output_filename="3D_vertical_profile_test.dfsu")

    # dfsu = fastmikeio.read("Small_3D.dfsu")
    # x = [-116.05, -115.5]
    # y = [11.9, 11.9]
    # dfsu.vertical_extractor(x, y, output_filename="Small_3D_vertical_profile_test.dfsu")

    dfsu = fastmikeio.read(r"\\usden1-stor2\projects\41807449_WS\Models\HD\Setup\Production\Scenario_2b-delay_half_spill\DRT_HD_MT_scenario_2b-delay_half_spill.m3fm - Result Files\3d_mt.dfsu")
    x = [787230, 792205]
    y = [2441086, 2436899]
    dfsu.vertical_extractor(x, y, r"\\usden1-stor2\projects\41807449_WS\Figures\ssc_vertical_statistics\test_vertical.dfsu")