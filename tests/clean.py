import os
os.chdir(os.path.dirname(__file__))
import fastmikeio
from fastmikeio import Dfsu
import numpy as np
from mikecore.DfsuBuilder import DfsuBuilder, DfsuFileType
from mikecore.DfsFactory import DfsFactory
from scipy.interpolate import griddata

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
    
    def plot(self, ax):
        for edge in self.edges:
            edge.plot(ax)
        return ax
    
    def __str__(self):
        return f"Polygon({self.vertices})"

def get_intersection_nodes(dfsu: Dfsu, p0: Point, p1: Point):
    nc_2d = dfsu.geometry.nc_2d  # (N_nodes_2d, 2)
    edges_2d = dfsu.geometry.edges_2d  # (N_edges_2d, 2)
    cross_line = Line(p0, p1)
    intersection_points = []
    for edge in edges_2d:
        p1 = Point(nc_2d[edge[0],0], nc_2d[edge[0],1], nc_2d[edge[0],2])
        p2 = Point(nc_2d[edge[1],0], nc_2d[edge[1],1], nc_2d[edge[1],2])
        line = Line(p1, p2)
        if not line.has_intersect(cross_line):
            continue
        intersect = line.get_intersect(cross_line)
        d1 = p1.distance(intersect)
        d2 = p2.distance(intersect)
        intersect.z = (p1.z * d2 + p2.z * d1) / (d1 + d2)
        intersection_points.append(intersect)
    intersection_points.sort(key=lambda point: p0.distance(point))
    return intersection_points


def vertical_profile(dfsu: Dfsu, output_fname):
    builder = DfsuBuilder.Create(DfsuFileType.DfsuVerticalProfileSigma)
    builder.FileTitle = dfsu.dfsu.FileTitle + " - Vertical Profile"
    builder.SetProjection(dfsu.dfsu.Projection)



if __name__ == "__main__":
    dfsu = fastmikeio.read("Small_3D.dfsu")
    print(dfsu.geometry.nc[:, 2])
    quit()
    x = [-116.05, -115.5]
    y = [11.9, 11.9]
    p0 = Point(x[0], y[0])
    p1 = Point(x[1], y[1])

    intersections = get_intersection_nodes(dfsu, p0, p1)
    fractions = np.array([1] + list(np.round(1 - np.cumsum(dfsu.geometry.sigma_fraction), 6)))
    
    X = np.repeat([pt.x for pt in intersections], dfsu.geometry.n_layers+1).flatten()
    Y = np.repeat([pt.y for pt in intersections], dfsu.geometry.n_layers+1).flatten()
    Z = np.array([fractions * pt.z for pt in intersections]).flatten()
    print(Z.shape)
    # print(intersections)