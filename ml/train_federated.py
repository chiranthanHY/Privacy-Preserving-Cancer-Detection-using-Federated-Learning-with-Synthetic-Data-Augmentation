#!/usr/bin/env python3
"""
ml/train_federated.py
Standalone Federated Learning simulation using FedAvg.

Simulates 3 hospital nodes, each training locally on their partition,
then averages weights on a central server. No Flower dependency needed.

Usage:
    python ml/train_federated.py

Reads:   data/node{1,2,3}/X.npy, y.npy  +  data/test/X.npy, y.npy
Uses:    ml/model.py  (build_cnn)
Writes:  models/federated_model.h5
         results/fl_metrics.json
"""
import os
import sys
import json
import time
import numpy as np

# ── Ensure GPU / CUDA libs are on LD_LIBRARY_PATH before importing TF ──────
import glob
venv_nvidia = os.path.join(os.path.dirname(sys.executable), "..", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia")
lib_dirs = glob.glob(f"{venv_nvidia}/*/lib")
local_nvidia = os.path.expanduser("~/.local/lib/python3.14/site-packages/nvidia")
lib_dirs += glob.glob(f"{local_nvidia}/*/lib")
if lib_dirs:
    existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + (f":{existing_ld}" if existing_ld else "")
    import ctypes
    for d in lib_dirs:
        for f in glob.glob(f"{d}/*.so*"):
            try:
                ctypes.CDLL(f)
            except Exception:
                pass

import tensorflow as tf

# Enable GPU memory growth so TensorFlow doesn't pre-allocate all GPU VRAM
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

# ── Project paths ────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# Ensure ml/ is importable
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from model import build_cnn

# ── FL Hyperparameters ───────────────────────────────────────────────────────
NUM_ROUNDS = 10
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_NODES = 3
FRACTION_TRAIN = 0.67  # ~2 of 3 clients per round


def load_node_data(node_id: int):
    """Load X.npy and y.npy for a given node (1-indexed)."""
    node_dir = os.path.join(DATA_DIR, f"node{node_id}")
    X = np.load(os.path.join(node_dir, "X.npy"))
    y = np.load(os.path.join(node_dir, "y.npy"))
    return X, y


def load_test_data():
    """Load the shared test set."""
    test_dir = os.path.join(DATA_DIR, "test")
    X = np.load(os.path.join(test_dir, "X.npy"))
    y = np.load(os.path.join(test_dir, "y.npy"))
    return X, y


def create_model():
    """Create and compile the CNN with the FL learning rate."""
    from tensorflow.keras.optimizers import Adam

    model = build_cnn()
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def fedavg_aggregate(global_weights, client_results):
    """
    Federated Averaging: weighted average of client model weights,
    proportional to each client's number of training samples.
    """
    total_samples = sum(n for _, n in client_results)
    new_weights = []

    for layer_idx in range(len(global_weights)):
        layer_sum = np.zeros_like(global_weights[layer_idx])
        for client_weights, num_samples in client_results:
            layer_sum += client_weights[layer_idx] * (num_samples / total_samples)
        new_weights.append(layer_sum)

    return new_weights


def train_client(node_id, global_weights):
    """
    Simulate one client's local training:
    1. Set weights from server
    2. Train locally for LOCAL_EPOCHS
    3. Return updated weights + metrics
    """
    X_train, y_train = load_node_data(node_id)

    model = create_model()
    model.set_weights(global_weights)

    history = model.fit(
        X_train,
        y_train,
        epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=0,
    )

    train_loss = history.history["loss"][-1]
    train_acc = history.history["accuracy"][-1]
    weights = model.get_weights()
    tf.keras.backend.clear_session()

    return weights, len(X_train), train_loss, train_acc


def evaluate_global(global_weights, X_test, y_test):
    """Evaluate the global model on the shared test set."""
    model = create_model()
    model.set_weights(global_weights)
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    tf.keras.backend.clear_session()
    return loss, acc


def main():
    print()
    print("=" * 60)
    print("  Federated Learning — Cancer Detection (Local Simulation)")
    print("=" * 60)
    print(f"  Strategy:          FedAvg")
    print(f"  Rounds:            {NUM_ROUNDS}")
    print(f"  Nodes:             {NUM_NODES}")
    print(f"  Local epochs:      {LOCAL_EPOCHS}")
    print(f"  Batch size:        {BATCH_SIZE}")
    print(f"  Learning rate:     {LEARNING_RATE}")
    print(f"  Fraction per round: {FRACTION_TRAIN}")
    print("=" * 60)
    print()

    # ── Verify data exists ───────────────────────────────────────────────────
    print("Verifying data partitions...")
    for node_id in range(1, NUM_NODES + 1):
        X, y = load_node_data(node_id)
        print(f"  Node {node_id}: X={X.shape}, y={y.shape}")
    X_test, y_test = load_test_data()
    print(f"  Test:   X={X_test.shape}, y={y_test.shape}")
    print()

    # ── Initialize global model ──────────────────────────────────────────────
    print("Initializing global model...")
    global_model = create_model()
    global_weights = global_model.get_weights()

    # Initial evaluation
    init_loss, init_acc = evaluate_global(global_weights, X_test, y_test)
    print(f"  Initial test accuracy: {init_acc:.4f} (random weights)")
    print()

    # ── FL Training Loop ─────────────────────────────────────────────────────
    round_metrics = []
    start_time = time.time()

    for round_num in range(1, NUM_ROUNDS + 1):
        round_start = time.time()
        print(f"─── Round {round_num}/{NUM_ROUNDS} ", "─" * 40)

        # Select clients for this round (simulate fraction_train)
        num_selected = max(1, int(NUM_NODES * FRACTION_TRAIN))
        selected_nodes = sorted(
            np.random.choice(range(1, NUM_NODES + 1), size=num_selected, replace=False)
        )
        print(f"  Selected nodes: {list(selected_nodes)}")

        # Train selected clients locally
        client_results = []
        for node_id in selected_nodes:
            weights, n_samples, loss, acc = train_client(node_id, global_weights)
            client_results.append((weights, n_samples))
            print(
                f"  Node {node_id}: {n_samples} samples, "
                f"loss={loss:.4f}, acc={acc:.4f}"
            )

        # Aggregate weights (FedAvg)
        global_weights = fedavg_aggregate(global_weights, client_results)

        # Evaluate global model on test set
        test_loss, test_acc = evaluate_global(global_weights, X_test, y_test)
        round_time = time.time() - round_start

        round_info = {
            "round": round_num,
            "selected_nodes": list(map(int, selected_nodes)),
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "round_time_sec": round(round_time, 1),
        }
        round_metrics.append(round_info)

        print(
            f"  ► Global test:  loss={test_loss:.4f}, "
            f"acc={test_acc:.4f}  ({round_time:.1f}s)"
        )
        print()

    total_time = time.time() - start_time

    # ── Save final model ─────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    final_model = create_model()
    final_model.set_weights(global_weights)

    model_path = os.path.join(MODEL_DIR, "federated_model.h5")
    final_model.save(model_path)
    model_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"✅  Saved: {model_path}  ({model_size:.1f} MB)")

    # ── Final evaluation ─────────────────────────────────────────────────────
    final_loss, final_acc = evaluate_global(global_weights, X_test, y_test)

    # ── Save metrics ─────────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_data = {
        "strategy": "FedAvg",
        "num_rounds": NUM_ROUNDS,
        "num_nodes": NUM_NODES,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "fraction_train": FRACTION_TRAIN,
        "initial_test_accuracy": float(init_acc),
        "final_test_loss": float(final_loss),
        "final_test_accuracy": float(final_acc),
        "total_training_time_sec": round(total_time, 1),
        "round_metrics": round_metrics,
    }

    metrics_path = os.path.join(RESULTS_DIR, "fl_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"✅  Saved: {metrics_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  FL Training Complete!")
    print("=" * 60)
    print(f"  Final test accuracy:  {final_acc:.4f}")
    print(f"  Final test loss:      {final_loss:.4f}")
    print(f"  Total time:           {total_time:.1f}s")
    print(f"  Model saved to:       models/federated_model.h5")
    print(f"  Metrics saved to:     results/fl_metrics.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
