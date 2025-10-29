import numpy as np
from mikecore.DfsuFile import DfsuFile
from mikecore.MeshFile import MeshFile
from mikecore.eum import eumQuantity, eumItem, eumUnit
import matplotlib.tri as tri
from collections import defaultdict



from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .dfsu3dsigma import Dfsu3DSigma
    from .dfsuverticalprofilesigma import DfsuVerticalProfileSigma


class Point:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

    def plot(self, ax):
        ax.scatter(self.x, self.y, c='red')
        return ax
    
    def distance(self, point):
        return np.sqrt((self.x - point.x)**2 + (self.y - point.y)**2)

    def __str__(self):
        return f"Point({self.x}, {self.y}, {self.z})"
    
    def __repr__(self):
        return f"Point({self.x}, {self.y}, {self.z})"

class Line:
    def __init__(self, p0: Point, p1: Point):
        self.p0 = p0
        self.p1 = p1
        self.m = (p1.y - p0.y) / (p1.x - p0.x)
        self.b = p0.y - self.m * p0.x

    def get_intersect(self, line):
        x = (line.b - self.b) / (self.m - line.m)
        y = self.m * x + self.b
        return Point(x, y)
    
    def contains(self, pt: Point, eps=1e-9):
        if (min(self.p0.x, self.p1.x) - eps <= pt.x <= max(self.p0.x, self.p1.x) + eps and
            min(self.p0.y, self.p1.y) - eps <= pt.y <= max(self.p0.y, self.p1.y) + eps):
            return True
        return False

    def has_intersect(self, line):
        if self.m == line.m:
            return False  # Parallel lines
        inter_pt = self.get_intersect(line)
        if self.contains(inter_pt) and line.contains(inter_pt):
            return True
        return False
    
    def plot(self, ax):
        ax.plot([self.p0.x, self.p1.x], [self.p0.y, self.p1.y], c='blue')
        return ax
    
    def __str__(self):
        return f"Line({self.p0}, {self.p1})"

class Polygon:
    def __init__(self, inputs):
        if isinstance(inputs[0], Point):
            self.vertices = inputs
            edges = []
            for i in range(len(inputs)):
                p0 = inputs[i]
                p1 = inputs[(i + 1) % len(inputs)]
                edges.append(Line(p0, p1))
            self.edges = edges
        else:
            self.edges = inputs
            self.vertices = []
            for edge in inputs:
                self.vertices.append(edge.p0)
        
    def get_intersects(self, line):
        intersect_pts = []
        for edge in self.edges:
            if edge.has_intersect(line):
                inter_pt = edge.get_intersect(line)
                intersect_pts.append(inter_pt)
        return intersect_pts
    
    def has_intersect(self, line):
        for edge in self.edges:
            if edge.has_intersect(line):
                return True
        return False
    
    def contains(self, point):
        # Ray casting algorithm to determine if point is inside polygon
        x, y = point.x, point.y
        n = len(self.vertices)
        inside = False

        p1x, p1y = self.vertices[0].x, self.vertices[0].y
        for i in range(1, n + 1):
            p2x, p2y = self.vertices[i % n].x, self.vertices[i % n].y
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside
    
    def plot(self, ax):
        for edge in self.edges:
            edge.plot(ax)
        return ax
    
    def __str__(self):
        return f"Polygon({self.vertices})"

class Geometry:
    def __init__(self, dfsu: DfsuFile):
        self.dfsu = dfsu

    @property
    def is_geo(self):
        return self.dfsu.Projection.WKTString == "LONG/LAT"
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
        return np.stack(self.dfsu.CalculateElementCenterCoordinates(), axis=1)
    
    def _get_bottom_layer_nodes(self):
        nc = self.nc
        n_layers = self.n_layers
        return nc[::(n_layers + 1), :]

    def find_closest_element(self, point):
        """Find the index of the closest element to a given point in 2D space"""
        distances = np.sqrt(np.sum((self.geometry.ec_2d - point)**2, axis=1))
        return np.argmin(distances)
    
    

class Geometry3DSigma(Geometry):
    def __init__(self, dfsu: DfsuFile):
        super().__init__(dfsu)

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
    
    def _get_bottom_triangles(self):
        n_layers = self.n_layers
        bottom_layer_elements = self.et[::n_layers]  # every n_layers-th element is the bottom prism
        bottom_triangles = bottom_layer_elements[:, :3] // (n_layers + 1)  # get node indices in bottom layer
        return bottom_triangles

    def get_intersection_nodes(self, p0: Point, p1: Point):
        cross_line = Line(p0, p1)
        nc = self.nc
        et = self.et
        n_layers = self.n_layers
        bottom_triangles = et[::n_layers, :3]
        edges = defaultdict(list)
        start_elem = None
        end_elem = None
        found_start = False
        found_end = False
        for t, tri in enumerate(bottom_triangles):
            n1, n2, n3 = tri
            current_elem = Polygon([Point(nc[n1,0], nc[n1,1], nc[n1,2]),
                                    Point(nc[n2,0], nc[n2,1], nc[n2,2]),
                                    Point(nc[n3,0], nc[n3,1], nc[n3,2])])
            if not found_start:
                if current_elem.contains(p0):
                    start_elem = (t, tri)
                    found_start = True
            if not found_end:
                if current_elem.contains(p1):
                    end_elem = (t, tri)
                    found_end = True
            edge1 = tuple(sorted((n1, n2)))
            edge2 = tuple(sorted((n2, n3)))
            edge3 = tuple(sorted((n3, n1)))
            edges[edge1].append(t)
            edges[edge2].append(t)
            edges[edge3].append(t)
        
        intersections = []
        elements = []
        node_left = []
        node_right = []
        d_left = []
        d_right = []
        intersections = []
        for edge in edges.keys():
            p1 = Point(nc[edge[0],0], nc[edge[0],1], nc[edge[0],2])
            p2 = Point(nc[edge[1],0], nc[edge[1],1], nc[edge[1],2])
            line = Line(p1, p2)
            if not line.has_intersect(cross_line):
                continue
            intersect = line.get_intersect(cross_line)
            intersections.append(intersect)
            d1 = p1.distance(intersect)
            d2 = p2.distance(intersect)
            node_left.append(edge[0])
            node_right.append(edge[1])
            d_left.append(d2)
            d_right.append(d1)
            elements.extend(edges[edge])

        dists = np.array([p0.distance(pt) for pt in intersections])
        sorted_idx = np.argsort(dists)
        intersections = np.array(intersections)[sorted_idx]
        node_left = np.array(node_left)[sorted_idx]
        node_right = np.array(node_right)[sorted_idx]
        d_left = np.array(d_left)[sorted_idx]
        d_right = np.array(d_right)[sorted_idx]
        
        elements = np.unique(np.array(elements))
        tri = bottom_triangles[elements]  
        centers = (np.mean(nc[tri], axis=1))[:, :2]
        p0_xy = [p0.x, p0.y]
        dists = np.linalg.norm(centers - p0_xy, axis=1)
        elements = elements[np.argsort(dists)]

        return intersections, node_left, node_right, d_left, d_right, elements

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

class GeometryVerticalProfileSigma(Geometry):
    def __init__(self, dfsu: DfsuFile):
        super().__init__(dfsu)



    @property
    def nc_1d(self):
        return self.nc[::(self.n_layers + 1), :]
    
    @staticmethod
    def _get_dist_geo(lon, lat, lon1, lat1):
        # assuming input in degrees!
        R = 6371e3  # Earth radius in metres
        dlon = np.deg2rad(lon1 - lon)
        dlon[dlon > np.pi] = dlon[dlon > np.pi] - 2 * np.pi
        dlon[dlon < -np.pi] = dlon[dlon < -np.pi] + 2 * np.pi
        dlat = np.deg2rad(lat1 - lat)
        x = dlon * np.cos(np.deg2rad((lat + lat1) / 2))
        y = dlat
        d = R * np.sqrt(np.square(x) + np.square(y))
        return d

    def _relative_cumulative_distance(self):
        nc = self.nc_1d
        is_geo = self.is_geo
        if is_geo:
            lon, lat = nc[:-1, 0], nc[:-1, 1]
            lon1, lat1 = nc[1:, 0], nc[1:, 1]
            R = 6371e3  # Earth radius in metres
            dlon = np.deg2rad(lon1 - lon)
            dlon = np.where(dlon > np.pi, dlon - 2 * np.pi, dlon)
            dlon = np.where(dlon < -np.pi, dlon + 2 * np.pi, dlon)
            dlat = np.deg2rad(lat1 - lat)
            x = dlon * np.cos(np.deg2rad((lat + lat1) / 2))
            y = dlat
            dists = R * np.sqrt((x ** 2) + (y ** 2))
        else:
            diffs = np.diff(nc[:, :2], axis=0)
            dists = np.sqrt((diffs ** 2).sum(axis=1))
        d = np.concatenate(([0], np.cumsum(dists)))
        return d
    @property
    def nc_2d(self):
        s_coordinates = self._relative_cumulative_distance()
        s = np.repeat(s_coordinates, self.n_layers + 1)
        z = self.Z
        nc_2d = np.stack((s, z), axis=1)
        return nc_2d
    @property
    def et_2d(self):
        """Returns triangular element table adjusted for 2D coordinates."""
        et = self.et
        new_et = []
        for elem in et:
            new_et.append([elem[0], elem[1], elem[2]])
            new_et.append([elem[0], elem[2], elem[3]])
        return np.array(new_et)
    @property
    def ec_2d(self):
        et_relative = self.et_2d
        nc_relative = self.nc_2d
        enc = nc_relative[et_relative]     # Coordinates of the element nodes
        ec_relative = np.mean(enc, axis=1)   # Element center coordinates
        return ec_relative
