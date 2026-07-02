"""
ml/train_centralized.py
Trains the CNN on all 3 nodes' data merged together -- the CENTRALIZED
BASELINE. This represents the accuracy "ceiling": what you'd get if
privacy weren't a concern and all hospitals just shared raw data.
Federated learning (Rohith's part) should approach but not exceed this.

Output: models/centralized_model.h5
        results/centralized_metrics.json
"""
import os
import sys
import json
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

# Ensure ml/ is on the path so 'from model import build_cnn' works regardless of cwd
_ML_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_ML_DIR)
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from model import build_cnn

# Resolve paths relative to the project root
DATA_DIR    = os.path.join(_ROOT, "data")
MODEL_DIR   = os.path.join(_ROOT, "models")
RESULTS_DIR = os.path.join(_ROOT, "results")


def load_all_nodes():
    Xs, ys = [], []
    for node in ["node1", "node2", "node3"]:
        Xs.append(np.load(os.path.join(DATA_DIR, node, "X.npy")))
        ys.append(np.load(os.path.join(DATA_DIR, node, "y.npy")))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    X_train, y_train = load_all_nodes()
    X_test = np.load(os.path.join(DATA_DIR, "test", "X.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "test", "y.npy"))

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    model = build_cnn()
    model.fit(X_train, y_train, epochs=15, validation_split=0.2, batch_size=32)

    model.save(os.path.join(MODEL_DIR, "centralized_model.h5"))
    print(f"Saved {MODEL_DIR}/centralized_model.h5")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {"accuracy": acc, "report": report}
    out_path = os.path.join(RESULTS_DIR, "centralized_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Test accuracy: {acc:.4f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
