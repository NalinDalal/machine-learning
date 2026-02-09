import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils import load_mnist, plot_digit

X, y = load_mnist()
X_train, X_test = X[:60000], X[60000:]

noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
y_train_mod = X_train

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train_mod, y_train_mod)

some_index = 0
noise_test = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise_test
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)

