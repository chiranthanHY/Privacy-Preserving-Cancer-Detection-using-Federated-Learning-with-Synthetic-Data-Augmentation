"""
federated_learning/client_app.py
Flower ClientApp — runs on each "hospital node".

Each client:
  1. Receives global model weights from the server
  2. Trains locally on its own data (data/node{X}/)
  3. Sends updated weights + metrics back to the server
"""
import numpy as np
import tensorflow as tf
from flwr.clientapp import ClientApp
from flwr.app import Message, Context, ArrayRecord, MetricRecord, RecordDict

from .task import load_data, load_model


def _set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data and return updated weights."""
    _set_seed()

    # Read config from pyproject.toml (with defaults for direct simulation)
    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 3)
    learning_rate = context.run_config.get("learning-rate", 0.001)
    local_epochs = context.run_config.get("local-epochs", 3)
    batch_size = context.run_config.get("batch-size", 32)

    print(f"\n[Client {partition_id}] Training on local data...")

    # Load local data
    x_train, y_train, _, _ = load_data(partition_id, num_partitions)
    print(f"[Client {partition_id}] Loaded {len(x_train)} training samples")

    # Load model and set weights from server
    model = load_model(learning_rate=learning_rate)
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    # Train locally
    history = model.fit(
        x_train,
        y_train,
        epochs=local_epochs,
        batch_size=batch_size,
        validation_split=0.1,  # Use 10% of local data for validation
        verbose=1,
    )

    # Extract metrics
    train_loss = history.history["loss"][-1]
    train_acc = history.history["accuracy"][-1]
    print(f"[Client {partition_id}] Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")

    # Pack and return updated weights + metrics
    model_record = ArrayRecord(model.get_weights())
    metrics = {
        "num-examples": len(x_train),
        "train_loss": float(train_loss),
        "train_acc": float(train_acc),
    }
    content = RecordDict({
        "arrays": model_record,
        "metrics": MetricRecord(metrics),
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local test data."""
    _set_seed()

    partition_id = context.node_config.get("partition-id", 0)
    num_partitions = context.node_config.get("num-partitions", 3)
    learning_rate = context.run_config.get("learning-rate", 0.001)

    print(f"\n[Client {partition_id}] Evaluating on local data...")

    # Load local data (use the test split)
    _, _, x_test, y_test = load_data(partition_id, num_partitions)

    # Load model and set weights from server
    model = load_model(learning_rate=learning_rate)
    model.set_weights(msg.content["arrays"].to_numpy_ndarrays())

    # Evaluate
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"[Client {partition_id}] Eval loss: {loss:.4f}, acc: {acc:.4f}")

    # Return metrics (no model weights since we didn't train)
    metrics = {
        "num-examples": len(x_test),
        "eval_loss": float(loss),
        "eval_acc": float(acc),
    }
    content = RecordDict({
        "metrics": MetricRecord(metrics),
    })
    return Message(content=content, reply_to=msg)
