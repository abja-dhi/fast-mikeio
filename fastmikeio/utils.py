


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