"""Pre-training geometry checks. Fails loudly if the lifting tables or the
composition splat disagree with the dataset's own labels."""
import json
import os

import numpy as np

from .geometry import (DATA_ROOT, compose_max, compose_tables, frustum_cell_tables,
                       lid_bin_centers, load_cameras, load_fov_mask, load_info, L, DUMP)

D, DMIN, DMAX = 64, 2.0, 75.0
STRIDE, H, W = 16, 544, 960


def check_tables():
    K, cams = load_cameras()
    tables, _ = frustum_cell_tables(K, cams, H, W, STRIDE, D, DMIN, DMAX)
    bins = lid_bin_centers(D, DMIN, DMAX)
    fh, fw = H // STRIDE, W // STRIDE
    fd = os.path.join(DATA_ROOT, "episodes", "ep_000", "frame_00000")
    dets = json.load(open(os.path.join(fd, "detections.json")))
    lv = np.load(os.path.join(fd, "local_occupancy_visible.npz"))

    ok = tot = dumped = 0
    for cam_key, ds in dets.items():
        cid = int(cam_key.split("_")[1])
        gt = lv[cam_key]
        for d in ds:
            x, y, z = d["center_cam_optical"]
            if z < DMIN or z > DMAX:
                continue
            u = K[0, 0] * x / z + K[0, 2]
            v = K[1, 1] * y / z + K[1, 2]
            if not (0 <= u < W and 0 <= v < 540):
                continue
            i, j = min(int(v // STRIDE), fh - 1), min(int(u // STRIDE), fw - 1)
            b = int(np.argmin(np.abs(bins - z)))
            cell = int(tables[cid, (b * fh + i) * fw + j])
            if cell == DUMP:
                # legitimate for detections between the 64 m grid edge and the
                # 70 m label range; tracked separately
                dumped += 1
                continue
            tot += 1
            r, c = cell // L, cell % L
            r0, r1 = max(r - 1, 0), min(r + 2, L)
            c0, c1 = max(c - 1, 0), min(c + 2, L)
            if gt[r0:r1, c0:c1].sum() > 0.01:
                ok += 1
    rate = ok / max(tot, 1)
    print(f"[tables] detections on occupied GT cells: {ok}/{tot} = {rate:.3f} "
          f"(+{dumped} outside the 64 m grid)")
    assert rate > 0.92, "frustum tables disagree with labels"
    assert dumped / max(tot + dumped, 1) < 0.20, "too many detections land off-grid"


def check_compose():
    K, cams = load_cameras()
    info = load_info()
    fov = load_fov_mask()
    comp_idx, gshape, sub = compose_tables(cams, info)
    for ep, f in [("ep_000", 0), ("ep_007", 100), ("ep_013", 150)]:
        fd = os.path.join(DATA_ROOT, "episodes", ep, f"frame_{f:05d}")
        lv = np.load(os.path.join(fd, "local_occupancy_visible.npz"))
        gv = np.load(os.path.join(fd, "global_occupancy_visible.npy")).astype(np.float64)
        maps = np.stack([lv[f"cam_{c:03d}"] for c in range(len(cams))]) * fov[None]
        comp = compose_max(maps, comp_idx, gshape, sub)
        corr = np.corrcoef(comp.ravel(), gv.ravel())[0, 1]
        p, g = comp >= 0.25, gv >= 0.25
        tp = (p & g).sum()
        f1 = 2 * tp / max(p.sum() + g.sum(), 1)
        print(f"[compose] {ep}/f{f}: count {comp.sum():.1f} vs gt {gv.sum():.1f} "
              f"corr {corr:.3f} F1 {f1:.3f}")
        assert corr > 0.85, "composition geometry looks wrong"


if __name__ == "__main__":
    check_tables()
    check_compose()
    print("SELFCHECK PASS")
