"""
federated_learning/server_app.py
Flower ServerApp — orchestrates the Federated Learning process.

The server:
  1. Initializes a random model
  2. Sends it to randomly selected clients
  3. Receives updated weights and averages them (FedAvg)
  4. Repeats for N rounds
  5. Saves the final model as .h5 for Pavan (backend)
"""
import os
import json
from pprint import pprint

from flwr.serverapp import ServerApp, Grid
from flwr.serverapp.strategy import FedAvg
from flwr.app import ArrayRecord, Context, RecordDict

from .task import load_model

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
RESULTS_DIR = os.path.join(_ROOT, "results")

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Read config from pyproject.toml
    num_rounds = context.run_config["num-server-rounds"]
    fraction_train = context.run_config["fraction-train"]
    save_model = context.run_config["save-model"]
    learning_rate = context.run_config["learning-rate"]

    print("\n" + "=" * 60)
    print("  Federated Learning — Cancer Detection")
    print("=" * 60)
    print(f"  Strategy:     FedAvg")
    print(f"  Round budget: {num_rounds}")
    print(f"  Fraction:     {fraction_train} (2 of 3 clients)")
    print(f"  Learning rate: {learning_rate}")
    print("=" * 60 + "\n")

    # Create initial (randomly initialized) model
    model = load_model(learning_rate=learning_rate)
    initial_arrays = ArrayRecord(model.get_weights())

    # Define FedAvg strategy
    strategy = FedAvg(
        fraction_train=fraction_train,
    )

    # Run the FL loop
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        num_rounds=num_rounds,
    )

    # Log final metrics
    print("\n" + "=" * 60)
    print("  FL Training Complete")
    print("=" * 60)

    if hasattr(result, "metrics") and result.metrics:
        print("\nFinal server-side metrics:")
        pprint(result.metrics)

    # Save final model
    if save_model:
        os.makedirs(MODEL_DIR, exist_ok=True)
        ndarrays = result.arrays.to_numpy_ndarrays()
        final_model_path = os.path.join(MODEL_DIR, "federated_model.h5")
        model.set_weights(ndarrays)
        model.save(final_model_path)
        print(f"\nSaved final model to {final_model_path}")

        # Save metrics to JSON
        os.makedirs(RESULTS_DIR, exist_ok=True)
        metrics_path = os.path.join(RESULTS_DIR, "fl_metrics.json")
        metrics_data = {
            "num_rounds": num_rounds,
            "fraction_train": fraction_train,
            "learning_rate": learning_rate,
        }
        if hasattr(result, "metrics") and result.metrics:
            metrics_data["final_metrics"] = result.metrics
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"Saved metrics to {metrics_path}")

    print("=" * 60 + "\n")
