import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np

def load_mnist():
    """ """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"].astype(np.uint8)
    return X, y

def plot_digit(data):
    """

    :param data: 

    """
    image = data.reshape(28, 28)
    plt.imshow(image, cmap="binary")
    plt.axis("off")
    plt.show()

