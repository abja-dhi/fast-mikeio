import os
os.chdir(os.path.dirname(__file__))
import fastmikeio
from mikecore.DfsuFile import DfsuFile
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # dfsu = fastmikeio.read("3D.dfsu")
    # dfsu.plot.contourf(item_idx=0, time_idx=dfsu.n_timesteps-1, layer_idx=0)
    # plt.show()
    # x = [-117.2482047163, -117.2464082003]
    # y = [10.33932763618, 10.53730369851]
    # dfsu.vertical_extractor(x, y, output_filename="3D_vertical_profile_test.dfsu")

    # dfsu = fastmikeio.read("Small_3D.dfsu")
    # x = [-116.05, -115.5]
    # y = [11.9, 11.9]
    # vertical = dfsu.vertical_extractor(x, y, output_filename="Small_3D_vertical_profile_test.dfsu")
    # levels = np.logspace(np.log10(0.01), np.log10(10), 100)
    # cbar_levels = [0.01, 0.1, 1, 10]
    # vertical.plot.contourf(item_idx=1, time_idx=0, levels=levels, cbar_levels=cbar_levels, extend='max', show_mesh=False)
    # plt.show()

    # dfsu = fastmikeio.read(r"\\usden1-stor2\projects\41807449_WS\Models\HD\Setup\Production\Scenario_2b-delay_half_spill\DRT_HD_MT_scenario_2b-delay_half_spill.m3fm - Result Files\3d_mt.dfsu")
    # x = [787230, 792205]
    # y = [2441086, 2436899]
    # dfsu.vertical_extractor(x, y, r"\\usden1-stor2\projects\41807449_WS\Figures\ssc_vertical_statistics\test_vertical.dfsu")

    fname = "test_vertical.dfsu"
    dfsu = fastmikeio.read(fname, unit_conversion=1e16)
    levels = np.logspace(np.log10(0.01), np.log10(10), 100)
    cbar_levels = [0.01, 0.1, 1, 10]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax = dfsu.plot.contourf(item_idx=6, time_idx=dfsu.n_timesteps-1, levels=levels, cbar_levels=cbar_levels, ax=ax)
    ax = dfsu.plot.bathy(ax=ax)
    plt.show()
    dfsu.close()
    
