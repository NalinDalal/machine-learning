from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from utils import load_mnist

X, y = load_mnist()
X_train, y_train = X[:60000], y[:60000]

sgd_clf = SGDClassifier(random_state=42)
rf_clf = RandomForestClassifier(random_state=42)

print("SGD Accuracy:", cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))
print("RF Accuracy:", cross_val_score(rf_clf, X_train, y_train, cv=3, scoring="accuracy"))

