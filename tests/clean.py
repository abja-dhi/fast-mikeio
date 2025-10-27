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
    dfsu = fastmikeio.read("3D.dfsu")
    x = [-117.2482047163, -117.2464082003]
    y = [10.33932763618, 10.53730369851]
    dfsu.vertical_extractor(x, y, output_filename="3D_vertical_profile_test.dfsu")