import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from utils import load_mnist

X, y = load_mnist()
X_train, y_train = X[:60000], y[:60000]

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)

print("F1 Score (macro):", f1_score(y_multilabel, y_train_knn_pred, average="macro"))

