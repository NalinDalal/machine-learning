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