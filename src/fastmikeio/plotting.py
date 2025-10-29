import matplotlib as mpl
import matplotlib.pyplot as plt
import cycler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .dfsu3dsigma import Dfsu3DSigma, DfsuVerticalProfileSigma

class MatplotlibShell:
    
    class dhi_colors:
        blue1 = '#04426e'
        blue2 = '#4d9ab3'
        blue3 = '#0493b2'
        blue4 = '#c3dde5'
        
        green1 = '#93c47d' #'#01be62'
        green2 = '#00b591'
        green3 = '#6ad6af'
        
        gray1 = '#c4c4c4'
        gray2 = '#8b8b8c'
        gray3 = '#686c6e'

        red1= '#c81f00'
        red2 = '#ac1817'

        yellow1 = '#ffbb3c'
        dhi_yellow2 = '#ebd844'

        orange1 = '#ec8833'
        orange2 = '#d3741c'
        
    mpl.rcParams['font.size'] = 9
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['lines.color'] = 'black'
    mpl.rcParams['patch.edgecolor'] = 'white'
    mpl.rcParams['axes.grid.which'] = 'major'
    mpl.rcParams['lines.markersize'] = 1.6
    mpl.rcParams['ytick.labelsize'] = 8
    mpl.rcParams['xtick.labelsize'] = 8
    mpl.rcParams['ytick.labelright'] = False
    mpl.rcParams['xtick.labeltop'] = False
    mpl.rcParams['ytick.right'] = True
    mpl.rcParams['xtick.top'] = True
    mpl.rcParams['ytick.major.right'] = True
    mpl.rcParams['xtick.major.top'] = True
    mpl.rcParams['axes.labelweight'] = 'normal'
    mpl.rcParams['legend.fontsize'] = 8
    mpl.rcParams['legend.framealpha']= 0.5
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['axes.titleweight'] ='normal'
    mpl.rcParams['font.family'] ='monospace'
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.linewidth'] = 1.25
    mpl.rcParams['xtick.major.size'] = 5.0
    mpl.rcParams['xtick.minor.size'] = 3.0
    mpl.rcParams['ytick.major.size'] = 5.0
    mpl.rcParams['ytick.minor.size'] = 3.0
    mpl.rcParams['figure.dpi'] = 300.0
    colors = 2*['#283747','#0051a2', '#41ab5d', '#feb24c', '#93003a']
    line_style = 5*['-'] + 5*['--']
    mpl.rcParams['axes.prop_cycle'] = cycler.cycler('color',colors) +cycler.cycler('linestyle',line_style)
    alpha = 0.7
    to_rgba = mpl.colors.ColorConverter().to_rgba#
    color_list=[]
    for i, col in enumerate(mpl.rcParams['axes.prop_cycle']):
        color_list.append(to_rgba(col['color'], alpha))
    mpl.rcParams['axes.prop_cycle'] = cycler.cycler(color=color_list)
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'

    def subplots(**kwargs):
        figheight = kwargs.get('figheight', 10)
        figwidth = kwargs.get('figwidth', 14)
        figheight = figheight / 2.54  # Convert cm to inches
        figwidth = figwidth / 2.54   # Convert cm to inches
        nrow = kwargs.get('nrow', 1)
        ncol = kwargs.get('ncol', 1)
        sharex = kwargs.get('sharex', False)
        sharey = kwargs.get('sharey', False)
        width_ratios = kwargs.get('width_ratios', [1]*ncol)
        height_ratios = kwargs.get('height_ratios', [1]*nrow)
        fig, axs = plt.subplots(figsize = (figwidth,figheight),
                            nrows = nrow,
                            ncols = ncol,
                            gridspec_kw = {'width_ratios': width_ratios, 'height_ratios': height_ratios},
                            sharex = sharex,
                            sharey = sharey
                            )
        if nrow*ncol>1:
            for i,ax in enumerate(axs.reshape(-1)): 
                ax.grid(alpha = 0.25)
        else:
            axs.grid(alpha = 0.25)
            
        return fig, axs
    

class Plot:
    def __init__(self, dfsu: 'Dfsu3DSigma | DfsuVerticalProfileSigma'):
        self.dfsu = dfsu

    @staticmethod
    def print_number(number):
        """
        Prints a number with varying decimal places based on its value.

        - If the number is greater than or equal to 1, it is printed with 0 decimal places.
        - If the number is less than 1, it prints with as many decimal places as required to show the significant digits.

        Parameters:
        number (float): The number to be printed.
        """
        number = round(number,6)
        if number >= 1:
            #print(f"{number:.0f}")  # Print with 0 decimal places for numbers >= 1
            out = f"{number:.0f}"
        else:
            # Count how many non-zero decimals are present after the decimal point
            decimals = len(str(number).split('.')[1].rstrip('0'))
            #print(f"{number:.{decimals}f}")  # Print with the calculated decimal precision
            out = f"{number:.{decimals}f}"
        return out
    @staticmethod
    def _add_colorbar(ax, fig_obj, label, levels, cbar_ticks=None, pad=0.05, extend="max", orientation='vertical'):
        if orientation == 'horizontal':
            cax = make_axes_locatable(ax).append_axes("bottom", size="5%", pad=pad)
        else:
            cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=pad)
        colorbar = plt.colorbar(
            fig_obj,
            label=label,
            cax=cax,
            ticks=levels,
            boundaries=levels,
            extend=extend,
            orientation=orientation
        )
        colorbar.set_ticks(levels)
        if cbar_ticks is not None:
            colorbar.set_ticklabels(cbar_ticks)
        else:
            colorbar.set_ticklabels([Plot.print_number(i) for i in levels])
    def _get_tris(self, z, x_offset=0.0, y_offset=0.0):
        et = self.dfsu.geometry.et_2d
        nc = self.dfsu.geometry.nc_2d
        ec = self.dfsu.geometry.ec_2d
        nc = nc.copy()
        nc[:, 0] = nc[:, 0] + x_offset
        nc[:, 1] = nc[:, 1] + y_offset
        ec = ec.copy()
        ec[:, 0] = ec[:, 0] + x_offset
        ec[:, 1] = ec[:, 1] + y_offset
        nc_min_x = np.min(nc[:, 0])
        nc_max_x = np.max(nc[:, 0])
        nc_min_y = np.min(nc[:, 1])
        nc_max_y = np.max(nc[:, 1])
        if (nc_max_x - nc_min_x > 3000) or (nc_max_y - nc_min_y > 3000):
            nc = nc / 1000.0
            ec = ec / 1000.0 
        
        elem_table, _, z = self._create_tri_only_element_table(et, ec, data=z)
        triang = tri.Triangulation(nc[:, 0], nc[:, 1], elem_table)
        return triang
    @staticmethod
    def _create_tri_only_element_table(element_table, element_coordinates, data):
        if len(element_table.shape) == 1:
            element_table = np.stack(element_table)
        
        if element_table.shape[1] == 3:
            return element_table, element_coordinates, data
        else:
            # Split elements into two triangles and assign the element value to both triangles
            new_element_table = []
            new_data = []
            for i, element in enumerate(element_table):
                new_element_table.append([element[0], element[1], element[2]])
                new_data.append(data[i])
                new_element_table.append([element[0], element[2], element[3]])
                new_data.append(data[i])
            new_element_table = np.array(new_element_table)
            new_data = np.array(new_data)
            element_table = new_element_table
            enc = element_coordinates[element_table]     # Coordinates of the element nodes
            ec = np.mean(enc, axis=1)   # Element center coordinates
            return element_table, ec, new_data
    @staticmethod
    def _set_ax_properties(ax, title="", xlabel="", ylabel=""):
        # ax.set_aspect('equal')
        ax.grid(alpha=0.25)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    def _parse_kwargs(self, kwargs, item_idx=0):
        out = {}
        out["figwidth"] = kwargs.get('figwidth', 14)
        out["figheight"] = kwargs.get('figheight', 10)
        out["cmap"] = kwargs.get('cmap', 'turbo')
        out["levels"] = kwargs.get('levels', None)
        out["norm"] = kwargs.get('norm', 'log')
        out["bottom_threshold"] = kwargs.get('bottom_threshold', 1e-6)
        out["show_mesh"] = kwargs.get('show_mesh', False)
        out["mesh_alpha"] = kwargs.get('mesh_alpha', 0.5)
        out["mesh_color"] = kwargs.get('mesh_color', 'gray')
        out["mesh_lw"] = kwargs.get('mesh_lw', 0.5)
        out["extend"] = kwargs.get('extend', 'neither')
        out["add_colorbar"] = kwargs.get('add_colorbar', True)
        out["cbar_ticks"] = kwargs.get('cbar_ticks', None)
        out["cbar_orientation"] = kwargs.get('cbar_orientation', 'vertical')
        out["cbar_label"] = kwargs.get('cbar_label', f"{self.dfsu.ItemInfo[item_idx].Name} ({self.dfsu.ItemInfo[item_idx].Quantity.UnitDescription})")
        out["cbar_levels"] = kwargs.get('cbar_levels', out["levels"])
        out["title"] = kwargs.get('title', '')
        out["xlabel"] = kwargs.get('xlabel', '')
        out["ylabel"] = kwargs.get('ylabel', '')
        out["zorder"] = kwargs.get('zorder', 1)
        out["x_offset"] = kwargs.get('x_offset', 0.0)
        out["y_offset"] = kwargs.get('y_offset', 0.0)
        for key, value in kwargs.items():
            if key not in out:
                out[key] = value
        return out  


class Plot3DSigma(Plot):
    def __init__(self, dfsu: 'Dfsu3DSigma'):
        super().__init__(dfsu)

    def contourf(self, ax=None, data=None, item_idx=0, layer_idx=0, time_idx=None, **kwargs):
        prop = self._parse_kwargs(kwargs, item_idx=item_idx)
        if data is None:
            time_idx = self.dfsu.n_timesteps - 1 if time_idx is None else time_idx
            assert isinstance(item_idx, int), "item_idx must be an integer."
            assert isinstance(layer_idx, int), "layer_idx must be an integer."
            assert isinstance(time_idx, int), "time_idx must be an integer."
            data = self.dfsu.get_data(item_idx=item_idx, time_idx=time_idx, layer_idx=layer_idx).squeeze()
        node_data = self.dfsu.get_node_data(data, extrapolate=True)
        masked_data = np.where(node_data <= prop["bottom_threshold"], prop["bottom_threshold"], node_data)
        triang = self._get_tris(node_data, x_offset=prop["x_offset"], y_offset=prop["y_offset"])
        if prop["levels"] is None:
            vmin = prop["bottom_threshold"]
            vmax = np.nanmax(node_data)
            if prop["norm"] == 'log':
                prop["levels"] = np.logspace(np.log10(vmin), np.log10(vmax), 100)
            else:
                prop["levels"] = np.linspace(vmin, vmax, 100)
        else:
            vmin = prop["levels"][0]
            vmax = prop["levels"][-1]
        norm = LogNorm(vmin=vmin, vmax=vmax) if prop["norm"] == 'log' else Normalize(vmin=vmin, vmax=vmax)
        if ax is None:
            fig, ax = MatplotlibShell.subplots(nrow=1, ncol=1, figwidth=prop["figwidth"], figheight=prop["figheight"])
        if prop["show_mesh"]:
            ax.triplot(triang, color=prop["mesh_color"], linewidth=prop["mesh_lw"], alpha=prop["mesh_alpha"])
        fig_obj = ax.tricontourf(triang, masked_data, cmap=prop["cmap"], norm=norm, extend=prop["extend"], levels=prop["levels"], zorder=prop["zorder"])
        if prop["add_colorbar"]:
            self._add_colorbar(ax, fig_obj, levels=prop["cbar_levels"], cbar_ticks=prop["cbar_ticks"], extend=prop["extend"], label=prop["cbar_label"], orientation=prop["cbar_orientation"])
        Plot._set_ax_properties(ax, title=prop["title"], xlabel=prop["xlabel"], ylabel=prop["ylabel"])
        return ax

    def quantile(self, q, ax=None, item_idx=None, layer_idx=None, **kwargs):
        item_idx = 0 if item_idx is None else item_idx
        layer_idx = 0 if layer_idx is None else layer_idx
        assert 0 <= q <= 1, "Quantile q must be between 0 and 1."
        assert isinstance(item_idx, int), "item_idx must be an integer."
        assert isinstance(layer_idx, int), "layer_idx must be an integer."

        data = self.dfsu.statistics.quantile(q=q, item_idx=item_idx, layer_idx=layer_idx).squeeze()
        ax = self.contourf(ax=ax, data=data, **kwargs)
        return ax

    def max(self, ax=None, item_idx=None, layer_idx=None, **kwargs):
        item_idx = 0 if item_idx is None else item_idx
        layer_idx = 0 if layer_idx is None else layer_idx
        assert isinstance(item_idx, int), "item_idx must be an integer."
        assert isinstance(layer_idx, int), "layer_idx must be an integer."

        data = self.dfsu.statistics.max(item_idx=item_idx, layer_idx=layer_idx).squeeze()
        ax = self.contourf(ax=ax, data=data, **kwargs)
        return ax

    def min(self, ax=None, item_idx=None, layer_idx=None, **kwargs):
        item_idx = 0 if item_idx is None else item_idx
        layer_idx = 0 if layer_idx is None else layer_idx
        assert isinstance(item_idx, int), "item_idx must be an integer."
        assert isinstance(layer_idx, int), "layer_idx must be an integer."

        data = self.dfsu.statistics.min(item_idx=item_idx, layer_idx=layer_idx).squeeze()
        ax = self.contourf(ax=ax, data=data, **kwargs)
        return ax

    def mean(self, ax=None, item_idx=None, layer_idx=None, **kwargs):
        item_idx = 0 if item_idx is None else item_idx
        layer_idx = 0 if layer_idx is None else layer_idx
        assert isinstance(item_idx, int), "item_idx must be an integer."
        assert isinstance(layer_idx, int), "layer_idx must be an integer."

        data = self.dfsu.statistics.mean(item_idx=item_idx, layer_idx=layer_idx).squeeze()
        ax = self.contourf(ax=ax, data=data, **kwargs)
        return ax



class PlotVerticalProfileSigma(Plot):
    def __init__(self, dfsu: 'DfsuVerticalProfileSigma'):
        super().__init__(dfsu)

    def contourf(self, ax=None, data=None, item_idx=0, time_idx=None, **kwargs):
        prop = self._parse_kwargs(kwargs, item_idx=item_idx)
        if data is None:
            time_idx = self.dfsu.n_timesteps - 1 if time_idx is None else time_idx
            assert isinstance(item_idx, int), "item_idx must be an integer."
            assert isinstance(time_idx, int), "time_idx must be an integer."
            data = np.repeat(self.dfsu.get_data(item_idx=item_idx, time_idx=time_idx, layer_idx=None, reshape=False).squeeze(), 2)
        node_data = self.dfsu.get_node_data(data, extrapolate=True)
        masked_data = np.where(node_data <= prop["bottom_threshold"], prop["bottom_threshold"], node_data)
        et_2d = self.dfsu.geometry.et_2d
        nc_2d = self.dfsu.geometry.nc_2d
        triang = tri.Triangulation(nc_2d[:, 0], nc_2d[:, 1], et_2d)
        if prop["levels"] is None:
            vmin = np.nanmin(node_data)
            vmax = np.nanmax(node_data)
            if prop["norm"] == 'log':
                prop["levels"] = np.logspace(np.log10(vmin), np.log10(vmax), 100)
            else:
                prop["levels"] = np.linspace(vmin, vmax, 100)
        else:
            vmin = prop["levels"][0]
            vmax = prop["levels"][-1]
        norm = LogNorm(vmin=vmin, vmax=vmax) if prop["norm"] == 'log' else Normalize(vmin=vmin, vmax=vmax)
        if ax is None:
            fig, ax = MatplotlibShell.subplots(nrow=1, ncol=1, figwidth=prop["figwidth"], figheight=prop["figheight"])
        if prop["show_mesh"]:
            ax.triplot(triang, color=prop["mesh_color"], linewidth=prop["mesh_lw"], alpha=prop["mesh_alpha"])
        fig_obj = ax.tricontourf(triang, masked_data, cmap=prop["cmap"], norm=norm, extend=prop["extend"], levels=prop["levels"], zorder=prop["zorder"])
        if prop["add_colorbar"]:
            self._add_colorbar(ax, fig_obj, levels=prop["cbar_levels"], cbar_ticks=prop["cbar_ticks"], extend=prop["extend"], label=prop["cbar_label"], orientation=prop["cbar_orientation"])
        Plot._set_ax_properties(ax, title=prop["title"], xlabel=prop["xlabel"], ylabel=prop["ylabel"])
        return ax
    
    def bathy(self, ax=None, **kwargs):
        nc_2d = self.dfsu.geometry.nc_2d
        n_layers = self.dfsu.geometry.n_layers
        s = nc_2d[::(n_layers+1), 0]
        z = nc_2d[::(n_layers+1), 1]
        if ax is None:
            fig, ax = MatplotlibShell.subplots(nrow=1, ncol=1)
        ax.plot(s, z, color='black')
        ymin = ax.get_ylim()[0]
        print(ymin)
        ax.fill_between(s, ymin, z, color='lightgray')
        # ax.set_xlabel('Distance along profile')
        return ax