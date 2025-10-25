import os
os.chdir(os.path.dirname(__file__))
import fastmikeio
import numpy as np
from mikecore.DfsuFile import DfsuFile




class Point:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

    def plot(self, ax):
        ax.scatter(self.x, self.y, c='red')
        return ax

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

dfsu = fastmikeio.read("Small_3D.dfsu")
x = [-116.05, -115.5]
y = [11.9, 11.9]

p0 = Point(x[0], y[0])
p1 = Point(x[1], y[1])
cross_line = Line(p0, p1)


nc = dfsu.geometry.nc   # (N_nodes, 3)
et = dfsu.geometry.et   # (N_elements, N_nodes_per_element)
nc_2d = dfsu.geometry.nc_2d  # (N_nodes_2d, 2)
et_2d = dfsu.geometry.et_2d  # (N_elements, N_nodes_per_element_2d)
et_2d_3d = dfsu.geometry.et_2d_3d  # (N_elements_2d, 3)
intersect_elems = []
for elem_idx, elem_nodes in enumerate(et_2d_3d):
    poly = Polygon([Point(nc[n,0], nc[n,1], nc[n,2]) for n in elem_nodes])
    if poly.has_intersect(cross_line):
        intersect_elems.append(elem_idx)
    
print(intersect_elems)

dfsu.close()