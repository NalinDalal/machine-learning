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

from tensorflow.keras import layers, Model

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