"""
Quick training script - MNIST + simple dense model (TensorFlow/Keras).
"""

import argparse

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_model() -> keras.Model:
    """Create a simple fully-connected MNIST classifier."""
    return keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple MNIST model with Keras")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    args = parser.parse_args()

    print("=" * 70)
    print("SENTRA Example Use Case - MNIST with a Simple Dense Neural Network")
    print("=" * 70)

    # 1. Load dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 2. Normalize images to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # 3. Build & compile model
    model = build_model()
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 4. Train
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

    # 5. Evaluate
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    # 6. Quick predictions
    predictions = model.predict(x_test[:5])
    for i, pred in enumerate(predictions):
        print(f"Image {i}: Predicted={pred.argmax()}, True={y_test[i]}")


if __name__ == "__main__":
    # ensure TF logging is less noisy
    tf.get_logger().setLevel("ERROR")
    main()
