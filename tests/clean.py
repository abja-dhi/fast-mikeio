import os
os.chdir(os.path.dirname(__file__))
import fastmikeio
from mikecore.DfsuFile import DfsuFile
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fname = "test_horizontal.dfsu"
    dfsu = fastmikeio.read(fname)
    dfsu.plot.NORM = "linear"
    dfsu.plot.CMAP = "jet"
    dfsu.plot.LEVELS = np.linspace(-76, -20, 100)
    dfsu.plot.CBAR_LEVELS = np.arange(-76, -21, 4)
    dfsu.plot.FIGWIDTH = 14
    dfsu.plot.FIGHEIGHT = 10
    dfsu.plot.EXTEND = 'both'
    ax = dfsu.plot.contourf(item_idx=1, time_idx=0, title="Horizontal slice at time index 0")
    # ax.set_aspect('equal')
    plt.show()