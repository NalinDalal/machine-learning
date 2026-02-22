# ======================================================================
# SECTION 3 · LeNet-5 (1998)
# ======================================================================
# Input(1,32,32) → C1(6,5×5) → S2(AvgPool 2×2) → C3(16,5×5) →
# S4(AvgPool 2×2) → C5(120,5×5) → F6(84) → Out(10)
# Original used tanh activation; modern version uses ReLU.
# ======================================================================
from tensorflow import keras
from tensorflow.keras import layers, Model

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
