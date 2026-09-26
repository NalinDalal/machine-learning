from sklearn.datasets import fetch_mldata

mnist = fetch_mldata("MNIST original")
mnist

X, y = mnist["data"], mnist["target"]
X.shape
y.shape
