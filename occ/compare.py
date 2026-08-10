"""Head-to-head: the visible-label model vs the full-label model.

The question worth answering is not just "which scores higher" -- the two are scored
against different targets -- but whether the full-label model genuinely infers vehicles
that are hidden from the camera.  So this measures both models on the cells where a
vehicle exists but *nothing of it is visible*: any credit there can only come from
context (an occluding bus, a queue at a light, lane structure), never from pixels
of the vehicle itself.
"""
import argparse
import json
import os

import numpy as np
import torch

from .data import VAL_EPS, cache_paths
from .evaluate import Predictor
from .geometry import load_fov_mask

OCC = 0.25          # a cell counts as occupied at a quarter vehicle of mass
QUIET = 0.05        # below this, a cell shows essentially nothing of a vehicle


def rate(num, den):
    return float(num) / max(float(den), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-visible", default="runs/r50_lss_v1/best.pt")
    ap.add_argument("--ckpt-full", default="runs/r50_lss_full_v1/best.pt")
    ap.add_argument("--cache", default="runs/cache")
    ap.add_argument("--out", default="runs/comparison.json")
    ap.add_argument("--frame-step", type=int, default=4)
    args = ap.parse_args()

    fov = load_fov_mask()
    m = fov > 0.5
    lab_vis = np.load(cache_paths(args.cache, "visible")[0], mmap_mode="r")
    lab_full = np.load(cache_paths(args.cache, "full")[0], mmap_mode="r")
    frames = [(ei, f) for ei in VAL_EPS for f in range(0, 200, args.frame_step)]

    acc = {}
    for name, ckpt in [("visible_model", args.ckpt_visible), ("full_model", args.ckpt_full)]:
        pred = Predictor(ckpt)
        s = dict(hid_cells=0, hid_hit=0, hid_mass_gt=0.0, hid_mass_pred=0.0,
                 vis_cells=0, vis_hit=0, views=0,
                 cnt_pred=0.0, cnt_vis_gt=0.0, cnt_full_gt=0.0)
        for k, (ei, f) in enumerate(frames):
            p = pred.predict_frame(ei, f)                      # [141,16,16]
            v = np.asarray(lab_vis[ei, f], dtype=np.float32)
            fu = np.asarray(lab_full[ei, f], dtype=np.float32)
            hidden = np.clip(fu - v, 0.0, None)

            # cells holding a hidden vehicle with no visible trace of it
            hid = (hidden >= OCC) & (v < QUIET) & m[None]
            s["hid_cells"] += int(hid.sum())
            s["hid_hit"] += int(((p >= OCC) & hid).sum())
            s["hid_mass_gt"] += float(hidden[hid].sum())
            s["hid_mass_pred"] += float(p[hid].sum())

            # cells the camera can plainly see -- the easy stuff must not regress
            vis = (v >= OCC) & m[None]
            s["vis_cells"] += int(vis.sum())
            s["vis_hit"] += int(((p >= OCC) & vis).sum())

            s["cnt_pred"] += float((p * fov[None]).sum())
            s["cnt_vis_gt"] += float((v * fov[None]).sum())
            s["cnt_full_gt"] += float((fu * fov[None]).sum())
            s["views"] += p.shape[0]
            if k % 50 == 0:
                print(f"{name}: frame {k}/{len(frames)}", flush=True)
        acc[name] = {
            "hidden_cell_recall": rate(s["hid_hit"], s["hid_cells"]),
            "hidden_mass_recovered": rate(s["hid_mass_pred"], s["hid_mass_gt"]),
            "visible_cell_recall": rate(s["vis_hit"], s["vis_cells"]),
            "vehicles_per_view_pred": rate(s["cnt_pred"], s["views"]),
            "hidden_cells_evaluated": s["hid_cells"],
            "visible_cells_evaluated": s["vis_cells"],
        }
        del pred
        torch.cuda.empty_cache()

    ref = {
        "vehicles_per_view_visible_gt": rate(s["cnt_vis_gt"], s["views"]),
        "vehicles_per_view_full_gt": rate(s["cnt_full_gt"], s["views"]),
        "frames_evaluated": len(frames),
    }
    out = {"reference": ref, **acc}
    with open(args.out, "w") as fp:
        json.dump(out, fp, indent=2)
    print(json.dumps(out, indent=2))
    print("COMPARE DONE")


if __name__ == "__main__":
    main()
