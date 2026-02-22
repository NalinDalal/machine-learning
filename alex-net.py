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