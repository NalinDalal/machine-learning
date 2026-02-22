
# ======================================================================
# SECTION 2 · POOLING LAYER
# ======================================================================
# Subsamples (shrinks) input to reduce computation, memory, and params.
# Max pooling: takes the max value in each receptive field.
# Avg pooling: takes the mean.
# Typical: 2×2 kernel, stride 2 → drops 75% of input values.
# ======================================================================
import matplotlib.pyplot as plt

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