"""Compute perceptual fingerprints for every image and cluster near-duplicates.

Purpose: the random 70/15/15 split scatters near-identical variants of the same
source crest across train/val/test, which inflates accuracy. To measure that, we
first need to know which images are variants of each other.

Fingerprint is a 16x16 greyscale thumbnail composited on white. That catches
rescaled / recompressed / lightly-shifted duplicates. It does NOT catch heavy
rotations or extreme crops of the same source render -- see the README caveat.
"""

import glob
import os

import numpy as np
from PIL import Image

from data import DATA_ROOT

CACHE = "checkpoints/fingerprints.npz"
SIZE = 16


def compute_fingerprints(data_root=DATA_ROOT):
    paths = sorted(glob.glob(os.path.join(data_root, "*", "*.png")))
    vecs = np.zeros((len(paths), SIZE * SIZE), dtype=np.float32)

    for i, p in enumerate(paths):
        img = Image.open(p).convert("RGBA")
        canvas = Image.new("RGB", img.size, (255, 255, 255))
        canvas.paste(img, (0, 0), img)
        thumb = canvas.convert("L").resize((SIZE, SIZE), Image.BILINEAR)
        v = np.asarray(thumb, dtype=np.float32).ravel() / 255.0
        # Contrast-normalise so brightness/contrast jitter doesn't split a group.
        v -= v.mean()
        norm = np.linalg.norm(v)
        vecs[i] = v / norm if norm > 1e-6 else v

    labels = np.array([os.path.basename(os.path.dirname(p)) for p in paths])
    return paths, vecs, labels


def cluster_within_class(vecs, labels, threshold):
    """Union-find over pairs closer than `threshold` (cosine distance).

    Clustering is done per class -- images of different clubs are never grouped,
    which is what we want since groups only exist to keep variants together.
    """
    group_ids = np.full(len(vecs), -1, dtype=np.int64)
    next_group = 0

    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls)
        V = vecs[idx]

        # Vectors are unit-norm, so cosine distance = 1 - V @ V.T
        sim = V @ V.T
        adjacency = sim >= (1.0 - threshold)

        parent = list(range(len(idx)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in zip(*np.nonzero(np.triu(adjacency, k=1))):
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                parent[ra] = rb

        local = {}
        for i in range(len(idx)):
            root = find(i)
            if root not in local:
                local[root] = next_group
                next_group += 1
            group_ids[idx[i]] = local[root]

    return group_ids, next_group


def load_or_compute():
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return list(d["paths"]), d["vecs"], d["labels"]

    paths, vecs, labels = compute_fingerprints()
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, paths=np.array(paths), vecs=vecs, labels=labels)
    return paths, vecs, labels


if __name__ == "__main__":
    paths, vecs, labels = load_or_compute()
    print(f"Fingerprinted {len(paths)} images\n")

    print(f"{'threshold':>10} {'groups':>8} {'imgs/group':>11} {'largest':>8} {'singletons':>11}")
    for t in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]:
        gids, n = cluster_within_class(vecs, labels, t)
        sizes = np.bincount(gids)
        print(f"{t:>10.3f} {n:>8} {len(paths)/n:>11.1f} {sizes.max():>8} "
              f"{int((sizes == 1).sum()):>11}")
