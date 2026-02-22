# ======================================================================
# SECTION 7 · FULL MNIST CNN TRAINING DEMO
# ======================================================================
# A practical CNN trained on MNIST to demonstrate the full pipeline:
# Conv → Pool → Conv → Pool → Flatten → Dense → Dropout → Output
# ======================================================================

import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

print("\n" + "=" * 60)
print("SECTION 7: MNIST CNN Training")
print("=" * 60)

# Load MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Reshape to (N, 28, 28, 1) and normalize to [0, 1]
X_train = X_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# Build a simple CNN for MNIST
mnist_cnn = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    # Conv block 1
    layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
    layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    # Conv block 2
    layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
    layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
    layers.MaxPooling2D(pool_size=2),
    # Classifier
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(10, activation="softmax"),
], name="MNIST_CNN")

mnist_cnn.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

mnist_cnn.summary()

# Train
print("\n--- Training MNIST CNN ---")
history = mnist_cnn.fit(X_train, y_train, epochs=5, batch_size=128,
                        validation_split=0.1, verbose=1)

# Evaluate
test_loss, test_acc = mnist_cnn.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# Plot training curves
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Accuracy Curve")

plt.tight_layout()
plt.savefig("mnist_cnn_training.png", dpi=100)
plt.close()
print("Saved mnist_cnn_training.png")

print("\n" + "=" * 60)
print("  All CNN sections completed successfully")
print("=" * 60)
