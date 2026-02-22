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
from tensorflow.keras import layers, Model

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


