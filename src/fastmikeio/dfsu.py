from mikecore.DfsuFile import DfsuFile, DfsuFileType
from mikecore.DfsFile import DfsFile
from mikecore.DfsuBuilder import DfsuBuilder

import numpy as np
from scipy.sparse import csr_matrix
from concurrent.futures import ThreadPoolExecutor
from tqdm import trange

from .geometry import Geometry3DSigma, GeometryVerticalProfileSigma, Point
from .statistics import Statistics3DSigma, StatisticsVerticalProfileSigma
from .plotting import Plot3DSigma, PlotVerticalProfileSigma    


class Dfsu:
    def __init__(self, filename, unit_conversion=1):
        self.filename = filename
        self.unit_conversion = unit_conversion
        self.dfsu = DfsuFile.Open(filename)
        self.ItemInfo = self.dfsu.ItemInfo
        
    def close(self):
        self.dfsu.Close()

    @property
    def n_items(self):
        return len(self.dfsu.ItemInfo)
    @property
    def n_timesteps(self):
        return self.dfsu.NumberOfTimeSteps


    def get_data(self, item_idx=None, time_idx=None, layer_idx=None, reshape=True):
        """
        Get all data as a 3D numpy array: (items, times, layers, 2d_nodes)
        """
        # Required variables
        n2d = int(self.geometry.ec.shape[0] / self.geometry.n_layers)
        n_layers = self.geometry.n_layers
        n_timesteps = self.n_timesteps
        n_items = self.n_items

        if time_idx is None: time_idx = range(n_timesteps)
        if item_idx is None: item_idx = range(n_items)
        if layer_idx is None: layer_idx = range(n_layers)
        if isinstance(item_idx, int): item_idx = [item_idx]
        if isinstance(time_idx, int): time_idx = [time_idx]
        if isinstance(layer_idx, int): layer_idx = [layer_idx]
        if reshape:
            data = np.empty((len(item_idx), len(time_idx), len(layer_idx), n2d), dtype=np.float32)
        else:
            data = np.empty((len(item_idx), len(time_idx), self.geometry.ec.shape[0]), dtype=np.float32)
        for i_item, itm in enumerate(trange(len(item_idx), desc="Items")):
            itm = item_idx[itm]
            for i_time, t in enumerate(trange(len(time_idx), desc="Time steps", leave=False)):
                t = time_idx[t]
                if not reshape:
                    full_data = self.dfsu.ReadItemTimeStep(itm + 1, t).Data
                    data[i_item, i_time, :] = full_data
                else:
                    full_data = self.dfsu.ReadItemTimeStep(itm + 1, t).Data.reshape((n2d, n_layers))
                    sel = full_data[:, layer_idx]
                    data[i_item, i_time, :, :] = sel.T
        data *= self.unit_conversion
        return data
  
    def get_node_data(self, data, extrapolate=True):
        et = self.geometry.et_2d # zero based
        ec = self.geometry.ec_2d
        nc = self.geometry.nc_2d
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
    

class Dfsu3DSigma(Dfsu):
    def __init__(self, filename, unit_conversion=1):
        super().__init__(filename, unit_conversion=unit_conversion)
        self.geometry = Geometry3DSigma(self.dfsu)
        self.statistics = Statistics3DSigma(self)
        self.plot = Plot3DSigma(self)

    def to_mesh(self, fname):
        self.geometry.to_mesh(fname)

    def vertical_extractor(self, x, y, output_filename=None):
        p0 = Point(x[0], y[0], 0)
        p1 = Point(x[1], y[1], 0)
        n_layers = self.geometry.n_layers
        intersections, node_left, node_right, d_left, d_right, elements = self.geometry.get_intersection_nodes(p0, p1)
        nodes_l = []
        nodes_r = []
        weights_l = []
        weights_r = []
        for i in range(len(node_left)):
            total_d = d_left[i] + d_right[i]
            for l in range(n_layers + 1):
                nodes_l.append(node_left[i] + l)
                nodes_r.append(node_right[i] + l)
                weights_l.append(d_right[i] / total_d)
                weights_r.append(d_left[i] / total_d)
        nodes_l = np.array(nodes_l)
        nodes_r = np.array(nodes_r)
        weights_l = np.array(weights_l)
        weights_r = np.array(weights_r)
        X = np.repeat([pt.x for pt in intersections], self.geometry.n_layers+1).flatten()
        Y = np.repeat([pt.y for pt in intersections], self.geometry.n_layers+1).flatten()
        z = self.geometry.Z
        Z = z[nodes_l] * weights_r + z[nodes_r] * weights_l
        
        et = []
        for i in trange(len(intersections)-1, desc="Creating element table"):
            offset1 = i * (n_layers + 1)
            offset2 = (i+1) * (n_layers + 1)
            for l in range(n_layers):
                n1 = offset1 + l
                n2 = offset2 + l
                n3 = offset2 + l + 1
                n4 = offset1 + l + 1
                et.append([n1, n2, n3, n4])
        et = np.array(et) + 1  # 1-based indexing
        element_indices = []
        for elem in elements:
            for l in range(n_layers):
                element_indices.append(elem*n_layers + l)
        element_indices = np.array(element_indices)
        
        
        builder = DfsuBuilder.Create(DfsuFileType.DfsuVerticalProfileSigma)
        builder.FileTitle = self.dfsu.FileTitle + " - Vertical Profile"
        builder.SetProjection(self.dfsu.Projection)
        builder.SetNodes(X, Y, Z, np.zeros_like(X, dtype=np.int32))
        builder.SetElements(et)
        builder.DeleteValueByte = DfsFile.DefaultDeleteValueByte
        builder.DeleteValueDouble = DfsFile.DefaultDeleteValueDouble
        builder.DeleteValueFloat = DfsFile.DefaultDeleteValueFloat
        builder.DeleteValueInt = DfsFile.DefaultDeleteValueInt
        builder.DeleteValueUnsignedInt = DfsFile.DefaultDeleteValueUnsignedInt

        builder.SetZUnit(self.dfsu.ZUnit)
        builder.SetNumberOfSigmaLayers(self.dfsu.NumberOfSigmaLayers)
        builder.SetTimeInfo(self.dfsu.StartDateTime, self.dfsu.TimeStepInSeconds)

        for i in range(1, len(self.dfsu.ItemInfo)):
            itemInfo = self.dfsu.ItemInfo[i]
            builder.AddDynamicItem(itemInfo.Name, itemInfo.Quantity)
        if output_filename is None:
            output_filename = self.filename.replace(".dfsu", "_vertical_profile.dfsu")
        file = builder.CreateFile(output_filename)

        for t in trange(self.n_timesteps, desc="Writing time steps"):
            z_dynamic = self.dfsu.ReadItemTimeStep(1, t).Data
            z_vals = z_dynamic[nodes_l] * weights_r + z_dynamic[nodes_r] * weights_l
            file.WriteItemTimeStep(1, t, t, z_vals.astype(np.float32))
            for i in range(1, len(self.dfsu.ItemInfo)):
                data = self.dfsu.ReadItemTimeStep(i+1, t).Data
                data_vals = data[element_indices]
                file.WriteItemTimeStep(i+1, t, t, data_vals.astype(np.float32))
        file.Close()

        return DfsuVerticalProfileSigma(output_filename, unit_conversion=self.unit_conversion)

    
class DfsuVerticalProfileSigma(Dfsu):
    def __init__(self, filename, unit_conversion=1):
        super().__init__(filename, unit_conversion=unit_conversion)
        self.geometry = GeometryVerticalProfileSigma(self.dfsu)
        self.statistics = StatisticsVerticalProfileSigma(self)
        self.plot = PlotVerticalProfileSigma(self)
