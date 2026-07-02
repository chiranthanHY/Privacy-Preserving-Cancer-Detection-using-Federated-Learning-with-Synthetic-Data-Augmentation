"""
ml/partition.py
Splits data/X_all.npy + y_all.npy into 3 equal "hospital node" subsets
(IID split -- each node gets a random, representative sample of all classes)
plus one shared test set.

Output: data/node1/X.npy, y.npy   (same for node2, node3)
        data/test/X.npy,  y.npy
"""
import os
import numpy as np

# Resolve paths relative to the project root (parent of this ml/ directory)
_ML_DIR       = os.path.dirname(os.path.abspath(__file__))
_ROOT         = os.path.dirname(_ML_DIR)
DATA_DIR      = os.path.join(_ROOT, "data")
TEST_FRACTION = 0.20   # 20% shared test set, remaining 80% split across 3 nodes


def main():
    X = np.load(os.path.join(DATA_DIR, "X_all.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_all.npy"))
    n = len(X)
    print(f"Total samples: {n}")

    rng = np.random.default_rng(seed=42)     # fixed seed = reproducible split
    indices = rng.permutation(n)

    test_size = int(TEST_FRACTION * n)
    test_idx = indices[:test_size]
    remaining_idx = indices[test_size:]

    node_splits = np.array_split(remaining_idx, 3)
    node_map = {
        "node1": node_splits[0],
        "node2": node_splits[1],
        "node3": node_splits[2],
        "test": test_idx,
    }

    for name, idx in node_map.items():
        out_dir = os.path.join(DATA_DIR, name)
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, "X.npy"), X[idx])
        np.save(os.path.join(out_dir, "y.npy"), y[idx])
        print(f"{name}: {len(idx)} samples -> {out_dir}/X.npy, {out_dir}/y.npy")


if __name__ == "__main__":
    main()
