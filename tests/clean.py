import os
os.chdir(os.path.dirname(__file__))
import fastmikeio
from mikecore.DfsuFile import DfsuFile
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    fname = "Cross Section A-A'.dfsu"
    dfsu = fastmikeio.read(fname, unit_conversion=1e17)
    dfsu.plot.NORM = "log"
    dfsu.plot.CMAP = "jet"
    dfsu.plot.LEVELS = np.logspace(np.log10(0.001), np.log10(10), 100)
    dfsu.plot.CBAR_LEVELS = np.logspace(np.log10(0.001), np.log10(10), 10)
    dfsu.plot.FIGWIDTH = 20
    dfsu.plot.FIGHEIGHT = 10
    dfsu.plot.EXTEND = 'max'
    ax = dfsu.plot.bathy(figwidth=20, figheight=10)
    dfsu.plot.max(ax=ax, item_idx=1)
    plt.show()