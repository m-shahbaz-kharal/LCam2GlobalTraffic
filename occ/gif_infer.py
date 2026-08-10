"""Run the trained model over a contiguous run of frames and cache the predictions.

The GIF renderer (occ.gif_render) is CPU-only and reads this cache, so the layout can
be iterated on without touching a GPU.
"""
import argparse
import os

import numpy as np
import torch

from .data import EPISODES
from .evaluate import Predictor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/r50_lss_v1/best.pt")
    ap.add_argument("--ep", type=int, default=15)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="runs/gifs/preds.npz")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    pred = Predictor(args.ckpt)
    frames = list(range(args.start, args.start + args.n))
    out = np.zeros((len(frames), 141, 16, 16), dtype=np.float32)
    for k, f in enumerate(frames):
        out[k] = pred.predict_frame(args.ep, f)
        print(f"{EPISODES[args.ep]} frame {f} -> {out[k].sum():.1f} veh (raw)", flush=True)
    np.savez_compressed(args.out, pred=out.astype(np.float16),
                        episode=np.int32(args.ep), frames=np.asarray(frames, np.int32))
    print("wrote", args.out, "GIF_INFER_DONE")


if __name__ == "__main__":
    main()
