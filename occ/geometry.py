"""Camera geometry for UCFTCam2TWorld: frustum lifting tables and global composition.

Conventions (from the dataset docs):
  * World: CARLA left-handed, +x east, +y south, +z up, yaw about +z in degrees.
  * Sensor frame: +x forward, +y right, +z up.
  * Optical frame: +x right, +y down, +z forward;  optical = S2O @ sensor.
  * Local grid (16x16, 4 m cells): forward = 64 - (row+0.5)*4, right = (col+0.5)*4 - 32.
    Camera-centric with yaw only (no pitch): forward = dx*cos + dy*sin, right = -dx*sin + dy*cos.
  * Global grid: row = (y - oy)/4, col = (x - ox)/4, shape 136 x 153.
"""
import json
import math
import os

import numpy as np

DATA_ROOT = "/lustre/fs1/home/mu290328/datasets/UCFTCam2TWorld/dataset"
CELL = 4.0
L = 16                    # local grid cells per side
RANGE = L * CELL          # 64 m
DUMP = L * L              # index 256 = "outside the local grid"

S2O = np.array([[0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [1.0, 0.0, 0.0]])  # optical = S2O @ sensor


def load_cameras(root=DATA_ROOT):
    meta = json.load(open(os.path.join(root, "cameras.json")))
    K = np.asarray(meta["intrinsics_K"], dtype=np.float64)
    cams = []
    for c in meta["cameras"]:
        cams.append({
            "id": c["id"],
            "x": c["location"][0], "y": c["location"][1], "z": c["location"][2],
            "yaw": c["rotation_deg"]["yaw"],
            "cam_to_world": np.asarray(c["cam_to_world"], dtype=np.float64),
        })
    assert [c["id"] for c in cams] == list(range(len(cams)))
    return K, cams


def load_info(root=DATA_ROOT):
    return json.load(open(os.path.join(root, "dataset_info.json")))


def load_fov_mask(root=DATA_ROOT):
    return np.load(os.path.join(root, "meta", "local_fov_mask.npz"))["cam_000"].astype(np.float32)


def lid_bin_centers(D, dmin, dmax):
    """Linearly-increasing-discretisation depth bin centers (finer bins near the camera)."""
    i = np.arange(D + 1, dtype=np.float64)
    edges = dmin + (dmax - dmin) * i * (i + 1) / (D * (D + 1))
    return 0.5 * (edges[:-1] + edges[1:])


def frustum_cell_tables(K, cams, H=544, W=960, stride=16, D=64, dmin=2.0, dmax=75.0):
    """Per-camera lookup: (depth bin, feature row, feature col) -> local-grid cell index.

    Returns (tables, tables_flip), each int16 [n_cams, D*fh*fw]; value in [0,255] is
    row*16+col of the local grid, 256 means outside the grid.  tables_flip is the
    table to use when the input image was mirrored horizontally (BEV mirrors too).
    Depth bins are planar depth (optical z), so a detection's z lands in its own bin.
    """
    fh, fw = H // stride, W // stride
    jj, ii = np.meshgrid(np.arange(fw), np.arange(fh))          # (fh, fw)
    u = (jj + 0.5) * stride
    v = (ii + 0.5) * stride
    rays = np.stack([u, v, np.ones_like(u)], axis=-1) @ np.linalg.inv(K).T   # (fh,fw,3)
    d = lid_bin_centers(D, dmin, dmax)
    pts_opt = rays[None] * d[:, None, None, None]               # (D,fh,fw,3), z == d
    pts_sensor = pts_opt @ S2O                                  # row-vec: sensor = S2O^T @ opt

    n = len(cams)
    tables = np.empty((n, D * fh * fw), dtype=np.int16)
    tables_flip = np.empty_like(tables)
    for c in cams:
        M = c["cam_to_world"]
        Xw = pts_sensor @ M[:3, :3].T + M[:3, 3]
        dx = Xw[..., 0] - c["x"]
        dy = Xw[..., 1] - c["y"]
        yaw = math.radians(c["yaw"])
        cy, sy = math.cos(yaw), math.sin(yaw)
        fwd = dx * cy + dy * sy
        right = -dx * sy + dy * cy
        row = np.floor((RANGE - fwd) / CELL)
        col = np.floor((right + RANGE / 2.0) / CELL)
        ok = (row >= 0) & (row < L) & (col >= 0) & (col < L)
        idx = np.where(ok, row * L + col, DUMP).astype(np.int16)      # (D,fh,fw)
        tables[c["id"]] = idx.reshape(-1)
        m = idx[:, :, ::-1]                                           # mirrored image columns
        mf = np.where(m == DUMP, DUMP, (m // L) * L + (L - 1 - m % L)).astype(np.int16)
        tables_flip[c["id"]] = mf.reshape(-1)
    return tables, tables_flip


# ---------------------------------------------------------------- composition ----
def compose_tables(cams, info, sub=4):
    """Per-camera lookup: local-cell subpoint -> global-grid cell index (int32).

    Each local cell is split into sub*sub points, each carrying value/sub^2.
    Returns idx [n_cams, L*L*sub*sub]; GR*GC means "outside the town grid".
    """
    ox, oy = info["global_grid"]["origin_world_xy"]
    GR, GC = info["global_grid"]["shape_rows_cols"]
    off = (np.arange(sub) + 0.5) / sub * CELL
    rows = np.arange(L)
    cols = np.arange(L)
    fw = RANGE - (rows[:, None] * CELL + off[None, :])          # (L, sub)
    rt = (cols[:, None] * CELL + off[None, :]) - RANGE / 2.0    # (L, sub)
    FW = np.broadcast_to(fw[:, None, :, None], (L, L, sub, sub)).reshape(-1)
    RT = np.broadcast_to(rt[None, :, None, :], (L, L, sub, sub)).reshape(-1)

    out = np.empty((len(cams), L * L * sub * sub), dtype=np.int32)
    for c in cams:
        yaw = math.radians(c["yaw"])
        cy, sy = math.cos(yaw), math.sin(yaw)
        wx = c["x"] + FW * cy - RT * sy
        wy = c["y"] + FW * sy + RT * cy
        gr = np.floor((wy - oy) / CELL).astype(np.int64)
        gc = np.floor((wx - ox) / CELL).astype(np.int64)
        ok = (gr >= 0) & (gr < GR) & (gc >= 0) & (gc < GC)
        out[c["id"]] = np.where(ok, gr * GC + gc, GR * GC).astype(np.int32)
    return out, (GR, GC), sub


def compose_max(local_maps, comp_idx, grid_shape, sub):
    """Per-cell max over the 141 per-camera splats. local_maps: [n_cams, L, L]."""
    GR, GC = grid_shape
    ncell = GR * GC
    out = np.zeros(ncell, dtype=np.float64)
    s2 = sub * sub
    for ci in range(local_maps.shape[0]):
        vals = np.broadcast_to(local_maps[ci].reshape(L, L, 1, 1) / s2,
                               (L, L, sub, sub)).reshape(-1)
        splat = np.bincount(comp_idx[ci], weights=vals, minlength=ncell + 1)[:ncell]
        np.maximum(out, splat, out=out)
    return out.reshape(GR, GC)


def coverage_map(fov_mask, comp_idx, grid_shape, sub):
    """How many cameras' FOV wedges cover each global cell (by >=50% of the cell)."""
    GR, GC = grid_shape
    ncell = GR * GC
    cov = np.zeros((GR, GC), dtype=np.int32)
    s2 = sub * sub
    n = comp_idx.shape[0]
    for ci in range(n):
        vals = np.broadcast_to(fov_mask.reshape(L, L, 1, 1) / s2,
                               (L, L, sub, sub)).reshape(-1)
        splat = np.bincount(comp_idx[ci], weights=vals, minlength=ncell + 1)[:ncell]
        cov += (splat.reshape(GR, GC) > 0.5).astype(np.int32)
    return cov
