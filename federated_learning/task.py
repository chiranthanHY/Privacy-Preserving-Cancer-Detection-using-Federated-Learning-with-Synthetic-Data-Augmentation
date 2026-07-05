"""
federated_learning/task.py
Data loading and model utilities for Federated Learning.

Loads pre-partitioned .npy files created by ml/partition.py.
Provides the CNN model wrapper for Flower's ClientApp/ServerApp.
"""
import os
import sys
import numpy as np

# Ensure ml/ is on the path so 'from model import build_cnn' works
_FL_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_FL_DIR)
_ML_DIR = os.path.join(_ROOT, "ml")
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)

from model import build_cnn

DATA_DIR = os.path.join(_ROOT, "data")


def load_data(partition_id: int, num_partitions: int = 3):
    """Load pre-partitioned data for a specific node.

    Args:
        partition_id: Node index (0, 1, or 2)
        num_partitions: Total number of nodes (default 3)

    Returns:
        (x_train, y_train, x_test, y_test) as numpy arrays
    """
    node_id = partition_id + 1  # partition_id 0 -> node1, etc.
    node_dir = os.path.join(DATA_DIR, f"node{node_id}")

    x_train = np.load(os.path.join(node_dir, "X.npy"))
    y_train = np.load(os.path.join(node_dir, "y.npy"))

    # Use the shared test set
    test_dir = os.path.join(DATA_DIR, "test")
    x_test = np.load(os.path.join(test_dir, "X.npy"))
    y_test = np.load(os.path.join(test_dir, "y.npy"))

    return x_train, y_train, x_test, y_test


def load_model(learning_rate: float = 0.001):
    """Load and compile the CNN model.

    Uses the same architecture as the centralized baseline (ml/model.py).
    Wraps build_cnn() to allow custom learning rate for FL experiments.
    """
    model = build_cnn()

    # Recompile with custom learning rate if needed
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
