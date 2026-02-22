"""
Convolutional Neural Networks — Chapter 13
==========================================
Covers:
  1. Convolution Layer (filters, strides, padding)
  2. Pooling Layer (max pool, avg pool)
  3. LeNet-5 Architecture
  4. AlexNet Architecture (with LRN)
  5. GoogLeNet / Inception Module
  6. ResNet (skip connections, residual units)
  7. Full MNIST CNN training demo

Uses TensorFlow 2.x / Keras API.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt

# ======================================================================
# SECTION 1 · CONVOLUTION LAYER
# ======================================================================
# Each neuron connects only to pixels in its receptive field (f_h × f_w).
# Filters slide across the image with a given stride.
# SAME padding → zero-pad so output size = ceil(input/stride)
# VALID padding → no padding, may drop edge pixels
#
# Output of neuron at (i, j, k):
#   z_{i,j,k} = b_k + ΣΣΣ x_{i',j',k'} · w_{u,v,k',k}
#   where i' = u·s_h + f_h - 1,  j' = v·s_w + f_w - 1
# ======================================================================

print("=" * 60)
print("SECTION 1: Convolution Layer")
print("=" * 60)

# Demo: apply hand-crafted vertical & horizontal filters to sample images
from sklearn.datasets import load_sample_images

dataset = np.array(load_sample_images().images, dtype=np.float32)
batch_size, height, width, channels = dataset.shape
print(f"Sample images shape: {dataset.shape}")  # (2, 427, 640, 3)

# Create 2 filters: vertical line and horizontal line (7×7 kernel)
filters_manual = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters_manual[:, 3, :, 0] = 1  # vertical line filter
filters_manual[3, :, :, 1] = 1  # horizontal line filter

# Apply convolution using tf.nn.conv2d
# strides: [batch, height, width, channels] — typically [1, s_h, s_w, 1]
output = tf.nn.conv2d(dataset, filters_manual,
                      strides=[1, 2, 2, 1], padding="SAME")
print(f"Convolution output shape: {output.shape}")

# Visualize the horizontal filter output for the first image
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(dataset[0].astype(np.uint8))
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title("Horizontal Filter Output")
plt.imshow(output[0, :, :, 1], cmap="gray")
plt.axis("off")
plt.tight_layout()
plt.savefig("conv_filter_demo.png", dpi=100)
plt.close()
print("Saved conv_filter_demo.png")

# Using Keras Conv2D layer (learned filters)
conv_layer = layers.Conv2D(filters=32, kernel_size=3, strides=1,
                           padding="same", activation="relu")
sample_out = conv_layer(dataset)
print(f"Keras Conv2D output shape: {sample_out.shape}")


# ======================================================================
# SECTION 2 · POOLING LAYER
# ======================================================================
# Subsamples (shrinks) input to reduce computation, memory, and params.
# Max pooling: takes the max value in each receptive field.
# Avg pooling: takes the mean.
# Typical: 2×2 kernel, stride 2 → drops 75% of input values.
# ======================================================================

print("\n" + "=" * 60)
print("SECTION 2: Pooling Layer")
print("=" * 60)

# Max pooling
max_pool = tf.nn.max_pool2d(dataset, ksize=2, strides=2, padding="VALID")
print(f"Max pool output shape: {max_pool.shape}")

# Average pooling
avg_pool = tf.nn.avg_pool2d(dataset, ksize=2, strides=2, padding="VALID")
print(f"Avg pool output shape: {avg_pool.shape}")

# Visualize pooling effect
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(dataset[0].astype(np.uint8))
plt.axis("off")
plt.subplot(1, 3, 2)
plt.title("Max Pooled (2×2, stride 2)")
plt.imshow(max_pool[0].numpy().astype(np.uint8))
plt.axis("off")
plt.subplot(1, 3, 3)
plt.title("Avg Pooled (2×2, stride 2)")
plt.imshow(avg_pool[0].numpy().astype(np.uint8))
plt.axis("off")
plt.tight_layout()
plt.savefig("pooling_demo.png", dpi=100)
plt.close()
print("Saved pooling_demo.png")


# ======================================================================
# SECTION 3 · LeNet-5 (1998)
# ======================================================================
# Input(1,32,32) → C1(6,5×5) → S2(AvgPool 2×2) → C3(16,5×5) →
# S4(AvgPool 2×2) → C5(120,5×5) → F6(84) → Out(10)
# Original used tanh activation; modern version uses ReLU.
# ======================================================================

print("\n" + "=" * 60)
print("SECTION 3: LeNet-5")
print("=" * 60)


def build_lenet5(input_shape=(32, 32, 1), num_classes=10):
    """LeNet-5 architecture (modernized with ReLU instead of tanh)."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        # C1: 6 filters, 5×5 kernel, stride 1
        layers.Conv2D(6, kernel_size=5, strides=1, activation="tanh",
                      padding="valid", name="C1"),
        # S2: average pooling 2×2, stride 2
        layers.AveragePooling2D(pool_size=2, strides=2, name="S2"),
        # C3: 16 filters, 5×5 kernel, stride 1
        layers.Conv2D(16, kernel_size=5, strides=1, activation="tanh",
                      padding="valid", name="C3"),
        # S4: average pooling 2×2, stride 2
        layers.AveragePooling2D(pool_size=2, strides=2, name="S4"),
        # C5: 120 filters, 5×5 → output 1×1
        layers.Conv2D(120, kernel_size=5, strides=1, activation="tanh",
                      padding="valid", name="C5"),
        layers.Flatten(),
        # F6: 84 units
        layers.Dense(84, activation="tanh", name="F6"),
        # Output: 10 classes
        layers.Dense(num_classes, activation="softmax", name="Output"),
    ], name="LeNet5")
    return model


lenet = build_lenet5()
lenet.summary()


# ======================================================================
# SECTION 4 · AlexNet (2012) with LRN
# ======================================================================
# Input(3,224,224) → C1(96,11×11,s4) → MaxPool → C3(256,5×5) →
# MaxPool → C5(384,3×3) → C6(384,3×3) → C7(256,3×3) →
# F8(4096) → F9(4096) → Out(1000)
# Uses ReLU, Dropout, LRN.
#
# LRN: b_i = a_i * (k + α Σ a_j²)^(-β)
#   AlexNet: r=2, α=0.00002, β=0.75, k=1
# ======================================================================

print("\n" + "=" * 60)
print("SECTION 4: AlexNet")
print("=" * 60)


def build_alexnet(input_shape=(224, 224, 3), num_classes=1000):
    """AlexNet architecture with Local Response Normalization."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        # C1: 96 filters, 11×11, stride 4, SAME padding
        layers.Conv2D(96, kernel_size=11, strides=4, padding="same",
                      activation="relu", name="C1"),
        # LRN (using Lambda — TF's tf.nn.local_response_normalization)
        layers.Lambda(lambda x: tf.nn.local_response_normalization(
            x, depth_radius=2, alpha=0.00002, beta=0.75, bias=1.0),
            name="LRN1"),
        # S2: max pooling 3×3, stride 2
        layers.MaxPooling2D(pool_size=3, strides=2, padding="valid", name="S2"),
        # C3: 256 filters, 5×5, stride 1, SAME
        layers.Conv2D(256, kernel_size=5, strides=1, padding="same",
                      activation="relu", name="C3"),
        layers.Lambda(lambda x: tf.nn.local_response_normalization(
            x, depth_radius=2, alpha=0.00002, beta=0.75, bias=1.0),
            name="LRN2"),
        # S4: max pooling 3×3, stride 2
        layers.MaxPooling2D(pool_size=3, strides=2, padding="valid", name="S4"),
        # C5, C6, C7: 3×3 convolutions
        layers.Conv2D(384, kernel_size=3, padding="same", activation="relu",
                      name="C5"),
        layers.Conv2D(384, kernel_size=3, padding="same", activation="relu",
                      name="C6"),
        layers.Conv2D(256, kernel_size=3, padding="same", activation="relu",
                      name="C7"),
        layers.MaxPooling2D(pool_size=3, strides=2, padding="valid"),
        layers.Flatten(),
        # F8, F9: fully connected with dropout
        layers.Dense(4096, activation="relu", name="F8"),
        layers.Dropout(0.5),
        layers.Dense(4096, activation="relu", name="F9"),
        layers.Dropout(0.5),
        # Output
        layers.Dense(num_classes, activation="softmax", name="Output"),
    ], name="AlexNet")
    return model


alexnet = build_alexnet()
alexnet.summary()


# ======================================================================
# SECTION 5 · GoogLeNet / INCEPTION MODULE (2014)
# ======================================================================
# Inception module: 4 parallel paths, concatenated by depth.
#   Path 1: 1×1 conv
#   Path 2: 1×1 conv (bottleneck) → 3×3 conv
#   Path 3: 1×1 conv (bottleneck) → 5×5 conv
#   Path 4: 3×3 max pool → 1×1 conv
# All use stride 1, SAME padding → same height/width.
# 1×1 convs reduce dimensionality (bottleneck) and add nonlinearity.
# ======================================================================

print("\n" + "=" * 60)
print("SECTION 5: Inception Module (GoogLeNet)")
print("=" * 60)


def inception_module(x, f1, f3_reduce, f3, f5_reduce, f5, pool_proj,
                     name="inception"):
    """
    Inception module with 4 parallel paths.

    Args:
        x: input tensor
        f1: number of 1×1 filters (path 1)
        f3_reduce: number of 1×1 bottleneck filters before 3×3 (path 2)
        f3: number of 3×3 filters (path 2)
        f5_reduce: number of 1×1 bottleneck filters before 5×5 (path 3)
        f5: number of 5×5 filters (path 3)
        pool_proj: number of 1×1 filters after max pool (path 4)
    """
    # Path 1: 1×1 conv
    path1 = layers.Conv2D(f1, 1, padding="same", activation="relu",
                          name=f"{name}_1x1")(x)

    # Path 2: 1×1 bottleneck → 3×3 conv
    path2 = layers.Conv2D(f3_reduce, 1, padding="same", activation="relu",
                          name=f"{name}_3x3_reduce")(x)
    path2 = layers.Conv2D(f3, 3, padding="same", activation="relu",
                          name=f"{name}_3x3")(path2)

    # Path 3: 1×1 bottleneck → 5×5 conv
    path3 = layers.Conv2D(f5_reduce, 1, padding="same", activation="relu",
                          name=f"{name}_5x5_reduce")(x)
    path3 = layers.Conv2D(f5, 5, padding="same", activation="relu",
                          name=f"{name}_5x5")(path3)

    # Path 4: 3×3 max pool → 1×1 conv
    path4 = layers.MaxPooling2D(3, strides=1, padding="same",
                                name=f"{name}_pool")(x)
    path4 = layers.Conv2D(pool_proj, 1, padding="same", activation="relu",
                          name=f"{name}_pool_proj")(path4)

    # Concatenate along depth (axis=3)
    return layers.Concatenate(axis=-1, name=f"{name}_concat")(
        [path1, path2, path3, path4])


def build_googlenet(input_shape=(224, 224, 3), num_classes=1000):
    """Simplified GoogLeNet architecture."""
    inp = layers.Input(shape=input_shape)

    # Initial conv layers — reduce spatial dimensions
    x = layers.Conv2D(64, 7, strides=2, padding="same", activation="relu",
                      name="conv1")(inp)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1")(x)

    x = layers.Conv2D(64, 1, padding="same", activation="relu",
                      name="conv2_reduce")(x)
    x = layers.Conv2D(192, 3, padding="same", activation="relu",
                      name="conv2")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool2")(x)

    # Inception modules (3a, 3b)
    x = inception_module(x, 64, 96, 128, 16, 32, 32, name="inc_3a")
    x = inception_module(x, 128, 128, 192, 32, 96, 64, name="inc_3b")
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool3")(x)

    # Inception modules (4a-4e)
    x = inception_module(x, 192, 96, 208, 16, 48, 64, name="inc_4a")
    x = inception_module(x, 160, 112, 224, 24, 64, 64, name="inc_4b")
    x = inception_module(x, 128, 128, 256, 24, 64, 64, name="inc_4c")
    x = inception_module(x, 112, 144, 288, 32, 64, 64, name="inc_4d")
    x = inception_module(x, 256, 160, 320, 32, 128, 128, name="inc_4e")
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool4")(x)

    # Inception modules (5a, 5b)
    x = inception_module(x, 256, 160, 320, 32, 128, 128, name="inc_5a")
    x = inception_module(x, 384, 192, 384, 48, 128, 128, name="inc_5b")

    # Global average pooling → eliminates need for multiple FC layers
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return Model(inputs=inp, outputs=out, name="GoogLeNet")


googlenet = build_googlenet()
print(f"GoogLeNet — Total params: {googlenet.count_params():,}")
print(f"  (≈ 10× fewer than AlexNet's ~60M)")


# ======================================================================
# SECTION 6 · ResNet (2015)
# ======================================================================
# Key idea: skip connections (shortcut connections)
#   network learns residual f(x) = h(x) - x instead of h(x) directly
#
# Residual unit: input → Conv(3×3,BN,ReLU) → Conv(3×3,BN) → ADD input → ReLU
#
# When dimensions change (stride 2):
#   skip connection uses 1×1 conv with stride 2 to match shape
#
# ResNet-34: 3 RUs×64 → 4 RUs×128 → 6 RUs×256 → 3 RUs×512
# ======================================================================

print("\n" + "=" * 60)
print("SECTION 6: ResNet")
print("=" * 60)


def residual_unit(x, filters, strides=1, name="res"):
    """
    A single residual unit with skip connection.

    If strides > 1 or filters change, the skip connection uses
    a 1×1 conv to match dimensions.
    """
    # Skip connection
    shortcut = x
    if strides > 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same",
                                 use_bias=False, name=f"{name}_skip_conv")(x)
        shortcut = layers.BatchNormalization(name=f"{name}_skip_bn")(shortcut)

    # Main path: Conv → BN → ReLU → Conv → BN
    x = layers.Conv2D(filters, 3, strides=strides, padding="same",
                      use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.ReLU(name=f"{name}_relu1")(x)

    x = layers.Conv2D(filters, 3, strides=1, padding="same",
                      use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)

    # Add skip connection and apply ReLU
    x = layers.Add(name=f"{name}_add")([x, shortcut])
    x = layers.ReLU(name=f"{name}_relu2")(x)
    return x


def build_resnet34(input_shape=(224, 224, 3), num_classes=1000):
    """
    ResNet-34 architecture.
    3 RUs × 64 → 4 RUs × 128 → 6 RUs × 256 → 3 RUs × 512
    """
    inp = layers.Input(shape=input_shape)

    # Initial conv + pool (like GoogLeNet)
    x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False,
                      name="conv1")(inp)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same", name="pool1")(x)

    # Residual blocks
    # Block 1: 3 RUs × 64 filters
    for i in range(3):
        x = residual_unit(x, 64, strides=1, name=f"res2_{i}")

    # Block 2: 4 RUs × 128 filters (first RU has stride 2)
    for i in range(4):
        s = 2 if i == 0 else 1
        x = residual_unit(x, 128, strides=s, name=f"res3_{i}")

    # Block 3: 6 RUs × 256 filters
    for i in range(6):
        s = 2 if i == 0 else 1
        x = residual_unit(x, 256, strides=s, name=f"res4_{i}")

    # Block 4: 3 RUs × 512 filters
    for i in range(3):
        s = 2 if i == 0 else 1
        x = residual_unit(x, 512, strides=s, name=f"res5_{i}")

    # Global average pooling → FC
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    return Model(inputs=inp, outputs=out, name="ResNet34")


resnet = build_resnet34()
print(f"ResNet-34 — Total params: {resnet.count_params():,}")


# ======================================================================
# SECTION 7 · FULL MNIST CNN TRAINING DEMO
# ======================================================================
# A practical CNN trained on MNIST to demonstrate the full pipeline:
# Conv → Pool → Conv → Pool → Flatten → Dense → Dropout → Output
# ======================================================================

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
