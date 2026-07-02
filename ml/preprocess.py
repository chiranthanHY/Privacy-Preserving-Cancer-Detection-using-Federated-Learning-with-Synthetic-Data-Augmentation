"""
ml/preprocess.py
Loads every image from data/raw/<class_name>/, resizes to 224x224,
normalises pixels to 0-1, assigns integer labels, saves X and y as .npy arrays.

Output: data/X_all.npy  (shape N x 224 x 224 x 3)
        data/y_all.npy  (shape N,)
"""
import os
import cv2
import numpy as np

# Resolve paths relative to the project root (parent of this ml/ directory)
_ML_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.dirname(_ML_DIR)
RAW_DIR  = os.path.join(_ROOT, "data", "raw")
OUT_DIR  = os.path.join(_ROOT, "data")
IMG_SIZE = 224

CLASSES = [
    "adenocarcinoma",
    "large.cell.carcinoma",
    "squamous.cell.carcinoma",
    "normal",
]


def load_images():
    X, y = [], []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(RAW_DIR, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARN] Missing folder: {class_dir}")
            continue

        files = [f for f in os.listdir(class_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"Loading {len(files)} images for class '{class_name}' (label={label_idx})")

        for fname in files:
            img_path = os.path.join(class_dir, fname)
            img = cv2.imread(img_path)          # OpenCV reads as BGR by default
            if img is None:
                print(f"[WARN] Could not read {img_path}, skipping")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # convert BGR -> RGB
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label_idx)

    X = np.array(X, dtype=np.float32) / 255.0   # normalise 0-1
    y = np.array(y, dtype=np.int64)
    return X, y


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    X, y = load_images()
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
    np.save(os.path.join(OUT_DIR, "X_all.npy"), X)
    np.save(os.path.join(OUT_DIR, "y_all.npy"), y)
    print("Saved data/X_all.npy and data/y_all.npy")


if __name__ == "__main__":
    main()
