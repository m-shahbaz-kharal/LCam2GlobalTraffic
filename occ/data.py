"""Label cache and the (image -> 16x16 visible-occupancy) dataset."""
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import DATA_ROOT

cv2.setNumThreads(0)

N_CAMS = 141
N_FRAMES = 200
EPISODES = [f"ep_{i:03d}" for i in range(16)]
TRAIN_EPS = list(range(0, 13))
VAL_EPS = list(range(13, 16))


def cache_paths(cache_dir):
    return (os.path.join(cache_dir, "labels_visible.npy"),
            os.path.join(cache_dir, "global_visible_counts.npy"))


def _load_one(args):
    root, ep, f = args
    fd = os.path.join(root, "episodes", ep, f"frame_{f:05d}")
    z = np.load(os.path.join(fd, "local_occupancy_visible.npz"))
    lab = np.stack([z[f"cam_{c:03d}"] for c in range(N_CAMS)]).astype(np.float16)
    gcount = float(np.load(os.path.join(fd, "global_occupancy_visible.npy")).sum())
    return ep, f, lab, gcount


def build_cache(cache_dir, root=DATA_ROOT, workers=16):
    """labels [16,200,141,16,16] float16 + per-frame global visible counts [16,200]."""
    from multiprocessing import Pool
    os.makedirs(cache_dir, exist_ok=True)
    lab_path, cnt_path = cache_paths(cache_dir)
    if os.path.exists(lab_path) and os.path.exists(cnt_path):
        print("cache exists, skipping")
        return
    labels = np.zeros((len(EPISODES), N_FRAMES, N_CAMS, 16, 16), dtype=np.float16)
    counts = np.zeros((len(EPISODES), N_FRAMES), dtype=np.float32)
    jobs = [(root, ep, f) for ep in EPISODES for f in range(N_FRAMES)]
    with Pool(workers) as pool:
        for k, (ep, f, lab, gcount) in enumerate(pool.imap_unordered(_load_one, jobs, chunksize=8)):
            ei = EPISODES.index(ep)
            labels[ei, f] = lab
            counts[ei, f] = gcount
            if k % 400 == 0:
                print(f"cache {k}/{len(jobs)}", flush=True)
    np.save(lab_path, labels)
    np.save(cnt_path, counts)
    print("cache written:", lab_path)


class OccDataset(Dataset):
    """One sample = one camera picture + its visible 16x16 occupancy grid."""

    def __init__(self, cache_dir, episodes, root=DATA_ROOT, frame_step=1, hflip=False):
        self.root = root
        self.hflip = hflip
        lab_path, _ = cache_paths(cache_dir)
        self.labels = np.load(lab_path, mmap_mode="r")
        self.index = [(ei, f, c)
                      for ei in episodes
                      for f in range(0, N_FRAMES, frame_step)
                      for c in range(N_CAMS)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        ei, f, cam = self.index[i]
        path = os.path.join(self.root, "episodes", EPISODES[ei],
                            f"frame_{f:05d}", "rgb", f"cam_{cam:03d}.jpg")
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:                      # rare Lustre hiccup: one retry
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"unreadable image: {path}")
        img = img[:, :, ::-1]                # BGR -> RGB
        target = np.asarray(self.labels[ei, f, cam], dtype=np.float32)
        flip = 0
        if self.hflip and random.random() < 0.5:
            img = img[:, ::-1]
            target = target[:, ::-1].copy()
            flip = 1
        return {
            "img": torch.from_numpy(np.ascontiguousarray(img)),          # uint8 HWC
            "target": torch.from_numpy(target),
            "cam": cam,
            "flip": flip,
        }
