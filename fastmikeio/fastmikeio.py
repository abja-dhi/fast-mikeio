from mikecore.DfsuFile import DfsuFile, DfsuFileType
from mikecore.MeshFile import MeshFile
from mikecore.eum import eumQuantity, eumUnit, eumItem
import numpy as np
from tqdm import trange
from scipy.sparse import csr_matrix
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
import cycler
from matplotlib.colors import LogNorm, Normalize
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
        if kwargs.get('figheight'): figheight = kwargs.get('figheight')
        else: figheight = 4.25*(1+(5**.5))/2
        
        if kwargs.get('figwidth'): figwidth = kwargs.get('figwidth')
        else: figwidth = figheight
        
        if kwargs.get('nrow'): nrow = kwargs.get('nrow')
        else: nrow = 1
        
        if kwargs.get('ncol'): ncol = kwargs.get('ncol')
        else: ncol = 1    
        
        if kwargs.get('sharex'): sharex = kwargs.get('sharex')
        else: sharex = False   
        
        if kwargs.get('sharey'): sharey = kwargs.get('sharey')
        else: sharey = False 
        
        if kwargs.get('width_ratios'): width_ratios = kwargs.get('width_ratios')
        else: width_ratios = [1]*ncol

        if kwargs.get('height_ratios'): height_ratios = kwargs.get('height_ratios')
        else: height_ratios = [1]*nrow   
        
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

class DfsuGeometry:
    def __init__(self, dfsu: DfsuFile):
        self.dfsu = dfsu

    @property
    def n_layers(self):
        return self.dfsu.NumberOfLayers
    @property
    def n_nodes(self):
        return self.dfsu.NumberOfNodes
    @property
    def n_elements(self):
        return self.dfsu.NumberOfElements
    @property
    def n_nodes2d(self):
        return self.n_nodes // (self.n_layers + 1)
    @property
    def X(self):
        return self.dfsu.X
    @property
    def Y(self):
        return self.dfsu.Y
    @property
    def Z(self):
        return self.dfsu.Z
    
    @property
    def HAB(self):
        zn = self.Z[0:(self.n_layers+1)]
        HAB = zn - zn[0]
        return HAB
    @property
    def sigma_fraction(self):
        z = self.nc[:, 2].reshape(self.n_nodes2d, self.n_layers + 1)
        total_depth = z[:, -1] - z[:, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            sigma = np.diff(z, axis=1) / total_depth[:, np.newaxis]
            sigma[~np.isfinite(sigma)] = 0.0  # handle division by zero
        return np.mean(sigma, axis=0)
    @property
    def et(self):
        return np.stack(self.dfsu.ElementTable, axis=1).T - 1
    @property
    def nc(self):
        return np.stack((self.X, self.Y, self.Z), axis=1)
    @property
    def ec(self):
        return self.dfsu.CalculateElementCenterCoordinates()
    @property
    def nc_2d(self):
        return self._get_bottom_layer_nodes()
    @property
    def et_2d(self):
        return self._get_bottom_triangles()
    @property
    def ec_2d(self):
        nc_2d = self.nc_2d[:, :2]
        ec = nc_2d[self.et_2d]
        return ec.mean(axis=1)
    @property
    def et_2d_3d(self):
        n_layers = self.n_layers
        bottom_layer_elements = self.et[::n_layers]  # every n_layers-th element is the bottom prism
        return bottom_layer_elements[:, :3]  # get node indices in bottom layer
    @property
    def edges_2d(self):
        et_2d = self.et_2d
        edges = set()
        for elem in et_2d:
            n1, n2, n3 = elem
            edge1 = tuple(sorted((n1, n2)))
            edge2 = tuple(sorted((n2, n3)))
            edge3 = tuple(sorted((n3, n1)))
            edges.update([edge1, edge2, edge3])
        return np.array(list(edges))
    @property
    def _tri2d(self) -> tri.Triangulation:
        return tri.Triangulation(self.nc_2d[:, 0], self.nc_2d[:, 1], self.et_2d)

    def _get_bottom_layer_nodes(self):
        nc = self.nc
        n_layers = self.n_layers
        return nc[np.arange(0, nc.shape[0], n_layers + 1), :]

    def _get_bottom_triangles(self):
        n_layers = self.n_layers
        bottom_layer_elements = self.et[::n_layers]  # every n_layers-th element is the bottom prism
        bottom_triangles = bottom_layer_elements[:, :3] // (n_layers + 1)  # get node indices in bottom layer
        return bottom_triangles
    
    def to_mesh(self, fname):
        quantity = eumQuantity(eumItem.eumIBathymetry, eumUnit.eumUmeter)
        wktstring = self.dfsu.Projection.WKTString
        nc_2d = self.nc_2d
        et_2d = self.et_2d
        nodeIds = np.arange(1, nc_2d.shape[0] + 1)
        x = nc_2d[:, 0]
        y = nc_2d[:, 1]
        z = nc_2d[:, 2]
        nodeCodes = np.zeros_like(nodeIds)
        elemIds = np.arange(1, et_2d.shape[0] + 1)
        elemTypes = np.full_like(elemIds, 21)  # type 21 = triangle
        connectivity = et_2d + 1  # MikeCore uses 1-based indexing
        mesh = MeshFile.Create(eumQuantity=quantity, wktString=wktstring,
                               nodeIds=nodeIds, x=x, y=y, z=z, nodeCode=nodeCodes,
                               elmtIds=elemIds, elmtTypes=elemTypes, connectivity=connectivity)
        mesh.Write(fname)

    def find_closest_element(self, point):
        """Find the index of the closest element to a given point in 2D space"""
        distances = np.sqrt(np.sum((self.geometry.ec_2d - point)**2, axis=1))
        return np.argmin(distances)

class DfsuStatistics:
    def __init__(self, dfsu):
        self.dfsu = dfsu

    def quantile(self, q, item_idx=0, layer_idx=0):
        data = self.dfsu.get_data(item_idx=item_idx, layer_idx=layer_idx).squeeze()
        assert 0 <= q <= 1, "Quantile q must be between 0 and 1."
        out = np.quantile(data, q, axis=0)
        return out
    
    def max(self, item_idx=0, layer_idx=0):
        data = self.dfsu.get_data(item_idx=item_idx, layer_idx=layer_idx).squeeze()
        out = np.max(data, axis=0)
        return out
        
    def min(self, item_idx=0, layer_idx=0):
        data = self.dfsu.get_data(item_idx=item_idx, layer_idx=layer_idx).squeeze()
        out = np.min(data, axis=0)
        return out

    def mean(self, item_idx=0, layer_idx=0):
        data = self.dfsu.get_data(item_idx=item_idx, layer_idx=layer_idx).squeeze()
        out = np.mean(data, axis=0)
        return out

class DfsuPlot:
    def __init__(self, dfsu):
        self.dfsu = dfsu

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
        DfsuPlot._set_ax_properties(ax, title=prop["title"], xlabel=prop["xlabel"], ylabel=prop["ylabel"])
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
            colorbar.set_ticklabels([DfsuPlot.print_number(i) for i in levels])
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
        ax.set_aspect('equal')
        ax.grid(alpha=0.25)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    def _parse_kwargs(self, kwargs, item_idx=0):
        out = {}
        out["figwidth"] = kwargs.get('figwidth', 6)
        out["figheight"] = kwargs.get('figheight', 5)
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

class Dfsu:
    def __init__(self, filename, unit_conversion=1):
        self.filename = filename
        self.unit_conversion = unit_conversion
        self.dfsu = DfsuFile.Open(filename)
        if self.dfsu.DfsuFileType != DfsuFileType.Dfsu3DSigma:
            raise ValueError("Only 3D sigma dfsu files are supported at this stage.")
        self.geometry = DfsuGeometry(self.dfsu)
        self.statistics = DfsuStatistics(self)
        self.plot = DfsuPlot(self)
        self.has_static = False
        self.ItemInfo = self.dfsu.ItemInfo
        if self.geometry.X.shape == self.dfsu.ReadItemTimeStep(1, 0).Data.shape:
            self.has_static = True
            self.ItemInfo = self.dfsu.ItemInfo[1:]  # skip static item
        
    def close(self):
        self.dfsu.Close()

    @property
    def n_items(self):
        return len(self.dfsu.ItemInfo)
    @property
    def n_timesteps(self):
        return self.dfsu.NumberOfTimeSteps


    def to_mesh(self, fname):
        self.geometry.to_mesh(fname)

    def vertical_extractor(self, x, y, items=None, times=None):
        x = np.array(x).flatten()
        y = np.array(y).flatten()
        finder = self.geometry._tri2d.get_trifinder()
        elements_2d = finder(x, y)
        n_layers = self.geometry.n_layers
        elements = np.array([np.arange(i * n_layers, (i + 1) * n_layers) for i in elements_2d])
        data = self.get_data(item_idx=items, time_idx=times, reshape=False)
        vertical_data = data[:, :, elements]
        return vertical_data

    def get_data(self, item_idx=None, time_idx=None, layer_idx=None, reshape=True):
        """
        Get all data as a 3D numpy array: (items, times, layers, 2d_nodes)
        """
        # Required variables
        ec_2d = self.geometry.ec_2d
        n2d = ec_2d.shape[0] 
        n_layers = self.geometry.n_layers
        n_timesteps = self.n_timesteps
        n_items = self.n_items

        if time_idx is None: time_idx = range(n_timesteps)
        if item_idx is None: item_idx = range(n_items)
        if layer_idx is None: layer_idx = range(n_layers)
        if isinstance(item_idx, int): item_idx = [item_idx]
        if isinstance(time_idx, int): time_idx = [time_idx]
        if isinstance(layer_idx, int): layer_idx = [layer_idx]
        if self.has_static:
            added = 2
        else:
            added = 1
        if reshape:
            data = np.empty((len(item_idx), len(time_idx), len(layer_idx), n2d), dtype=np.float32)
        else:
            data = np.empty((len(item_idx), len(time_idx), self.geometry.ec[0].shape[0]), dtype=np.float32)
        for i_item, itm in enumerate(trange(len(item_idx), desc="Items")):
            itm = item_idx[itm]
            for i_time, t in enumerate(trange(len(time_idx), desc="Time steps", leave=False)):
                t = time_idx[t]
                if not reshape:
                    full_data = self.dfsu.ReadItemTimeStep(itm + added, t).Data
                    data[i_item, i_time, :] = full_data
                else:
                    full_data = self.dfsu.ReadItemTimeStep(itm + added, t).Data.reshape((n2d, n_layers))
                    sel = full_data[:, layer_idx]
                    data[i_item, i_time, :, :] = sel.T
        data *= self.unit_conversion
        return data
  
    def get_node_data(self, data, extrapolate=True):
        et = self.geometry.et_2d
        nc = self.geometry.nc_2d
        ec = self.geometry.ec_2d
        connectivity_matrix = self._create_node_element_matrix(et, nc.shape[0])
        node_centered_data = np.zeros(shape=nc.shape[0])
        node_indices = trange(connectivity_matrix.shape[0], desc="Nodes")
        args = (connectivity_matrix, ec, nc, data, extrapolate)
        with ThreadPoolExecutor(max_workers=40) as executor:
            results = list(executor.map(
                lambda n: self._process_node(n, *args), 
                node_indices
            ))

        node_centered_data = np.array(results)
        return node_centered_data

    @staticmethod
    def _create_node_element_matrix(element_table, num_nodes):
        row_ind = element_table.ravel()
        col_ind = np.repeat(np.arange(element_table.shape[0]), element_table.shape[1])
        data = np.ones(len(row_ind), dtype=int)
        connectivity_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(num_nodes, element_table.shape[0]))
        return connectivity_matrix
    @staticmethod
    def _process_node(n, connectivity_matrix, ec, nc, data, extrapolate):
        item = connectivity_matrix.getrow(n).indices
        I = ec[item][:, :2] - nc[n][:2]
        I2 = (I**2).sum(axis=0)
        Ixy = (I[:, 0] * I[:, 1]).sum(axis=0)
        lamb = I2[0] * I2[1] - Ixy**2
        omega = np.zeros(1)
        if lamb > 1e-10 * (I2[0] * I2[1]):
            lambda_x = (Ixy * I[:, 1] - I2[1] * I[:, 0]) / lamb
            lambda_y = (Ixy * I[:, 0] - I2[0] * I[:, 1]) / lamb
            omega = 1.0 + lambda_x * I[:, 0] + lambda_y * I[:, 1]
            if not extrapolate:
                omega[np.where(omega > 2)] = 2
                omega[np.where(omega < 0)] = 0
        if omega.sum() > 0:
            node_centered_data = np.sum(omega * data[item]) / np.sum(omega)
        else:
            InvDis = [1 / np.hypot(case[0], case[1]) for case in ec[item][:, :2] - nc[n][:2]]
            node_centered_data = np.sum(InvDis * data[item]) / np.sum(InvDis)
        return node_centered_data

    def __str__(self):
        desc = f"Dfsu file: {self.filename}\n"
        desc += f"Number of items: {len(self.ItemInfo)}\n"
        for i, item in enumerate(self.ItemInfo):
            desc += f"Item {i}: {item.Name}, Quantity: {item.Quantity}\n"
        return desc


def read(filename, unit_conversion=1):
    dfsu = Dfsu(filename, unit_conversion=unit_conversion)
    return dfsu



if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(__file__))
    fname = r"\\usden1-stor2\projects\41807449_WS\Models\HD\Setup\Production\Scenario_2b-delay_half_spill\DRT_HD_MT_scenario_2b-delay_half_spill.m3fm - Result Files\3d_mt.dfsu"
    dfsu = read(fname, unit_conversion=1000)
    # quantile = dfsu.statistics.quantile(q=0.95, item_idx=5, layer_idx=0)
    # print(np.max(quantile), np.min(quantile))
    levels = np.logspace(np.log10(0.01), np.log10(10), 100)
    cbar_levels = [0.01, 0.1, 1, 10]
    item_idx = 5
    layer_idx = 0
    xlabel = "Easting (m)"
    ylabel = "Northing (m)"
    cbar_label = "SSC (mg/L)"
    extend = 'max'
    
    ax = dfsu.plot.min(item_idx=item_idx, layer_idx=layer_idx, levels=levels, cbar_levels=cbar_levels, show_mesh=False, xlabel=xlabel, ylabel=ylabel, title="95th Percentile of SSC (mg/L)", cbar_label=cbar_label, extend=extend)
    fig = ax.get_figure()
    fig.savefig("0-Min.png", dpi=300, bbox_inches='tight')
    ax = dfsu.plot.mean(item_idx=item_idx, layer_idx=layer_idx, levels=levels, cbar_levels=cbar_levels, show_mesh=False, xlabel=xlabel, ylabel=ylabel, title="95th Percentile of SSC (mg/L)", cbar_label=cbar_label, extend=extend)
    fig = ax.get_figure()
    fig.savefig("1-Mean.png", dpi=300, bbox_inches='tight')
    ax = dfsu.plot.quantile(q=0.95, item_idx=item_idx, layer_idx=layer_idx, levels=levels, cbar_levels=cbar_levels, show_mesh=False, xlabel=xlabel, ylabel=ylabel, title="95th Percentile of SSC (mg/L)", cbar_label=cbar_label, extend=extend)
    fig = ax.get_figure()
    fig.savefig("2-0.95.png", dpi=300, bbox_inches='tight')
    ax = dfsu.plot.max(item_idx=item_idx, layer_idx=layer_idx, levels=levels, cbar_levels=cbar_levels, show_mesh=False, xlabel=xlabel, ylabel=ylabel, title="95th Percentile of SSC (mg/L)", cbar_label=cbar_label, extend=extend)
    fig = ax.get_figure()
    fig.savefig("3-Max.png", dpi=300, bbox_inches='tight')
    # plt.show()
    dfsu.close()