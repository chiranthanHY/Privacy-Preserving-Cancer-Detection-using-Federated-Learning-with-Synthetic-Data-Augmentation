"""
ml/explore_data.py
Run this FIRST, before preprocess.py. Confirms:
  - class counts per folder (expect imbalance: normal > carcinoma classes)
  - images actually load correctly
  - a visual sanity check via a sample grid
"""
import os
import cv2
import matplotlib.pyplot as plt

# Resolve paths relative to the project root (parent of this ml/ directory)
_ML_DIR     = os.path.dirname(os.path.abspath(__file__))
_ROOT       = os.path.dirname(_ML_DIR)
RAW_DIR     = os.path.join(_ROOT, "data", "raw")
_SAMPLE_OUT = os.path.join(_ROOT, "data", "sample_grid.png")
CLASSES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "squamous.cell.carcinoma",
    "normal",
]


def class_counts():
    counts = {}
    for c in CLASSES:
        class_dir = os.path.join(RAW_DIR, c)
        if os.path.isdir(class_dir):
            n = len([f for f in os.listdir(class_dir)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        else:
            n = 0
        counts[c] = n
    return counts


def show_samples(n_per_class=3, save_path=None):
    if save_path is None:
        save_path = _SAMPLE_OUT
    fig, axes = plt.subplots(len(CLASSES), n_per_class,
                              figsize=(3 * n_per_class, 3 * len(CLASSES)))
    for row, c in enumerate(CLASSES):
        class_dir = os.path.join(RAW_DIR, c)
        if not os.path.isdir(class_dir):
            continue
        files = [f for f in os.listdir(class_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))][:n_per_class]
        for col, fname in enumerate(files):
            img = cv2.imread(os.path.join(class_dir, fname))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax = axes[row, col] if len(CLASSES) > 1 else axes[col]
            ax.imshow(img)
            ax.set_title(c, fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path)
    print(f"Saved sample grid to {save_path}")


def main():
    counts = class_counts()
    print("Class counts:")
    for c, n in counts.items():
        print(f"  {c}: {n} images")
    total = sum(counts.values())
    print(f"Total: {total} images")

    if total == 0:
        print("[ERROR] No images found. Check that data/raw/<class_name>/ exists "
              "and matches the folder names in CLASSES above.")
        return

    show_samples()


if __name__ == "__main__":
    main()
