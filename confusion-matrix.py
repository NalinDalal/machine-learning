from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from utils import load_mnist

X, y = load_mnist()
X_train, y_train = X[:60000], y[:60000]
y_train_5 = (y_train == 5)

sgd_clf = SGDClassifier(random_state=42)
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

cm = confusion_matrix(y_train_5, y_train_pred)
print("Confusion Matrix:\n", cm)

