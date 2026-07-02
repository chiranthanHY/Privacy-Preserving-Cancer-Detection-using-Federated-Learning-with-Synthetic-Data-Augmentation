"""
ml/model.py
Defines build_cnn(): returns a COMPILED but UNTRAINED Keras CNN model.

Imported by:
  - ml/train_centralized.py         (Chiranthan)
  - backend/fl_client.py            (Rohith)
  - ml/train_fl_augmented.py        (Shree Gowda)

This file must never train the model itself -- it only defines the
architecture, so every teammate gets an identical starting point.
"""
from tensorflow.keras import layers, models


def build_cnn(input_shape=(224, 224, 3), num_classes=4):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",  # labels are ints, not one-hot
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":
    # quick sanity check: prints the architecture + parameter counts
    m = build_cnn()
    m.summary()
