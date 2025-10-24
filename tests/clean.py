import os
os.chdir(os.path.dirname(__file__))
import fastmikeio
import numpy as np


def _plane_from_segment_xy(p0, p1):
    """
    Vertical plane passing through 2D segment p0->p1 (z arbitrary).
    Returns (n, c, d_hat, L) where:
      n: plane normal (nx, ny, 0) with ||n|| = length of segment
      c: n · P0_3D, where P0_3D = (p0x, p0y, 0)
      d_hat: unit direction vector of the segment in XY (dx, dy, 0)
      L: segment length in XY
    Plane equation: n · X = c
    """
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    d = p1 - p0
    L = np.hypot(d[0], d[1])
    if L == 0.0:
        raise ValueError("p0 and p1 must be distinct.")
    # vertical plane normal = d x k = (-dy, dx, 0)
    n = np.array([-d[1], d[0], 0.0], dtype=float)  # not normalized (||n|| = L)
    c = n[0]*p0[0] + n[1]*p0[1]  # n·(p0,0)
    d_hat = np.array([d[0]/L, d[1]/L, 0.0], dtype=float)
    return n, c, d_hat, L

def _edge_table(k):
    if k == 8:  # hex
        return np.array([
            [0,1],[1,2],[2,3],[3,0],
            [4,5],[5,6],[6,7],[7,4],
            [0,4],[1,5],[2,6],[3,7],
        ], dtype=int)
    if k == 6:  # prism
        return np.array([
            [0,1],[1,2],[2,0],
            [3,4],[4,5],[5,3],
            [0,3],[1,4],[2,5],
        ], dtype=int)
    raise ValueError(f"Unsupported element with {k} nodes. Only 6 (prism) or 8 (hex) supported.")

def _unique_with_tol(points_uv, tol=1e-10):
    if points_uv.size == 0:
        return points_uv, np.empty(0, dtype=int)
    key = np.round(points_uv / tol).astype(np.int64)
    hashes = key[:,0] * 73856093 ^ key[:,1] * 19349663
    order = np.argsort(hashes, kind="mergesort")
    hashes = hashes[order]
    uniq = np.ones_like(hashes, dtype=bool)
    uniq[1:] = hashes[1:] != hashes[:-1]
    unique_idx = order[uniq]
    inv = np.empty(points_uv.shape[0], dtype=int)
    inv[unique_idx] = np.arange(unique_idx.size)
    last_unique = unique_idx[np.searchsorted(unique_idx, order, side="right")-1]
    inv[order[~uniq]] = inv[last_unique[~uniq]]
    return points_uv[unique_idx], inv

def _order_convex_polygon(pts_uv):
    if pts_uv.shape[0] <= 2:
        return np.arange(pts_uv.shape[0])
    c = pts_uv.mean(axis=0)
    ang = np.arctan2(pts_uv[:,1] - c[1], pts_uv[:,0] - c[0])
    return np.argsort(ang)

def _intersect_element_with_vertical_plane(nodes_xyz, elem_nodes, p0, n, c, d_hat, L, restrict_to_segment=True, eps=1e-12):
    """
    Returns:
      pts_xyz:   (m,3)
      pts_uv:    (m,2)  (u,z)
      edge_ends: (m,2)  original node indices (endpoints) of crossed edge for each point
    """
    k = len(elem_nodes)
    edges = _edge_table(k)

    verts = nodes_xyz[elem_nodes]  # (k,3)
    sd = verts[:,0]*n[0] + verts[:,1]*n[1] - c

    cross_pts_xyz = []
    cross_pts_uv  = []
    edge_ends     = []

    for e0_local, e1_local in edges:
        s0 = sd[e0_local]; s1 = sd[e1_local]
        v0 = verts[e0_local]; v1 = verts[e1_local]
        g0 = elem_nodes[e0_local]; g1 = elem_nodes[e1_local]
        pair = (min(g0, g1), max(g0, g1))

        if (abs(s0) <= eps and abs(s1) <= eps):
            for v in (v0, v1):
                u = (v[0]-p0[0])*d_hat[0] + (v[1]-p0[1])*d_hat[1]
                if (not restrict_to_segment) or (-eps <= u <= L + eps):
                    cross_pts_xyz.append(v)
                    cross_pts_uv.append([u, v[2]])
                    edge_ends.append(pair)
        elif abs(s0) <= eps or abs(s1) <= eps:
            v = v0 if abs(s0) <= eps else v1
            u = (v[0]-p0[0])*d_hat[0] + (v[1]-p0[1])*d_hat[1]
            if (not restrict_to_segment) or (-eps <= u <= L + eps):
                cross_pts_xyz.append(v)
                cross_pts_uv.append([u, v[2]])
                edge_ends.append(pair)
        elif s0 * s1 < 0.0:
            t = s0 / (s0 - s1)
            v = v0 + t*(v1 - v0)
            u = (v[0]-p0[0])*d_hat[0] + (v[1]-p0[1])*d_hat[1]
            if (not restrict_to_segment) or (-eps <= u <= L + eps):
                cross_pts_xyz.append(v)
                cross_pts_uv.append([u, v[2]])
                edge_ends.append(pair)

    if not cross_pts_xyz:
        return np.empty((0,3)), np.empty((0,2)), np.empty((0,2), dtype=int)

    pts_xyz = np.asarray(cross_pts_xyz, float)
    pts_uv  = np.asarray(cross_pts_uv,  float)
    edge_ends = np.asarray(edge_ends, dtype=int)

    # deduplicate within element
    uniq_uv, inv = _unique_with_tol(pts_uv, tol=1e-10)
    # keep first occurrence of each unique key
    _, keep_idx = np.unique(inv, return_index=True)
    pts_xyz = pts_xyz[keep_idx]
    pts_uv  = uniq_uv
    edge_ends = edge_ends[keep_idx]

    if pts_uv.shape[0] >= 3:
        ord_idx = _order_convex_polygon(pts_uv)
        pts_uv = pts_uv[ord_idx]
        pts_xyz = pts_xyz[ord_idx]
        edge_ends = edge_ends[ord_idx]

    return pts_xyz, pts_uv, edge_ends

def cross_section_vertical_plane_mesh(nodes_xyz, elems, p0, p1, restrict_to_segment=True, chunk_elems=20000):
    """
    Returns:
      xs_nodes_xyz : (P,3)
      xs_nodes_uv  : (P,2)
      xs_elems     : list[np.ndarray]
      xs_elem_to_3d_elem : (len(xs_elems),)
      xs_node_edge_ends  : (P,2)  <-- NEW: for each cross-section node, the (node_i, node_j) of the 3D edge
    """
    nodes_xyz = np.asarray(nodes_xyz, float)
    E = np.asarray(elems)
    if E.ndim != 2 or E.shape[1] not in (6,8):
        raise ValueError("`elems` must be shape (M,6) or (M,8)")

    if np.issubdtype(E.dtype, np.integer):
        valid_mask = (E >= 0)
        E_valid = np.where(valid_mask, E, -1)
    else:
        valid_mask = ~np.isnan(E)
        E_valid = np.where(valid_mask, E, -1).astype(np.int64)

    n, c, d_hat, L = _plane_from_segment_xy(p0, p1)

    all_pts_xyz = []
    all_pts_uv  = []
    all_edge_ends = []  # (node_i, node_j) per unique cross-section node
    xs_elems = []
    xs_elem_to_3d = []

    tol = 1e-10
    uv_key_to_idx = {}
    def _key_uv(u, z):
        return (int(np.round(u/tol)), int(np.round(z/tol)))

    M = E_valid.shape[0]
    for s in range(0, M, chunk_elems):
        e = min(M, s + chunk_elems)
        rows = E_valid[s:e]
        for j, row in enumerate(rows):
            elem_nodes = row[row >= 0]
            if elem_nodes.size == 0:
                continue
            pts_xyz, pts_uv, edge_ends = _intersect_element_with_vertical_plane(
                nodes_xyz, elem_nodes, p0=np.array([p0[0], p0[1], 0.0]),
                n=n, c=c, d_hat=d_hat, L=L, restrict_to_segment=restrict_to_segment
            )
            m = pts_uv.shape[0]
            if m < 2:
                continue

            poly_idx = []
            for q in range(m):
                u, z = float(pts_uv[q,0]), float(pts_uv[q,1])
                key = _key_uv(u, z)
                if key in uv_key_to_idx:
                    idx = uv_key_to_idx[key]
                else:
                    idx = len(all_pts_uv)
                    uv_key_to_idx[key] = idx
                    all_pts_uv.append([u, z])
                    all_pts_xyz.append(pts_xyz[q].tolist())
                    # store canonical sorted pair for stability
                    ni, nj = edge_ends[q]
                    all_edge_ends.append([min(ni, nj), max(ni, nj)])
                poly_idx.append(idx)

            if len(poly_idx) >= 3:
                xs_elems.append(np.array(poly_idx, dtype=np.int64))
                xs_elem_to_3d.append(s + j)

    if not xs_elems:
        return (np.empty((0,3)), np.empty((0,2)), [], np.empty(0, dtype=int), np.empty((0,2), dtype=int))

    xs_nodes_uv  = np.asarray(all_pts_uv,  float)
    xs_nodes_xyz = np.asarray(all_pts_xyz, float)
    xs_elem_to_3d = np.asarray(xs_elem_to_3d, dtype=int)
    xs_node_edge_ends = np.asarray(all_edge_ends, dtype=int)

    return xs_nodes_xyz, xs_nodes_uv, xs_elems, xs_elem_to_3d, xs_node_edge_ends


if __name__ == "__main__":
    dfsu = fastmikeio.read("Small_3D.dfsu")
    x = [-116.05, -115.5]
    y = [11.9, 11.9]
    p0 = np.array([x[0], y[0]])
    p1 = np.array([x[1], y[1]])

    nc = dfsu.geometry.nc
    et = dfsu.geometry.et

    print(nc.shape)

    xs_nodes_xyz, xs_nodes_uv, xs_elems, xs_elem_to_3d, xs_node_edge_ends = cross_section_vertical_plane_mesh(nc, et, p0, p1)
    print(xs_nodes_xyz.shape)