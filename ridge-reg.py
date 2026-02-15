# ridge_regression.py
# Demonstrates Ridge Regression (L2 Regularization)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Generate quadratic data (nonlinear)
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# Define a helper function to plot predictions
def plot_model(model, X, y, label=None, color=None):
    """

    :param model: param X:
    :param y: param label:  (Default value = None)
    :param color: Default value = None)
    :param X: param label:  (Default value = None)
    :param label:  (Default value = None)

    """
    X_new = np.linspace(-3, 3, 100).reshape(100, 1)
    y_new = model.predict(X_new)
    plt.plot(X_new, y_new, color=color, linewidth=2, label=label)
    plt.plot(X, y, "b.", alpha=0.3)
    plt.xlabel("X")
    plt.ylabel("y")

# Create polynomial regression models with different alpha values
ridge_1 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("ridge_reg", Ridge(alpha=1, solver="cholesky"))
])

ridge_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("ridge_reg", Ridge(alpha=10, solver="cholesky"))
])

ridge_0 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("ridge_reg", Ridge(alpha=0, solver="cholesky"))  # behaves like LinearRegression
])

# Fit models
ridge_0.fit(X, y)
ridge_1.fit(X, y)
ridge_10.fit(X, y)

# Plot results
plt.figure(figsize=(8, 6))
plot_model(ridge_0, X, y, label="No Regularization (α=0)", color="g")
plot_model(ridge_1, X, y, label="Moderate Regularization (α=1)", color="r")
plot_model(ridge_10, X, y, label="High Regularization (α=10)", color="k")
plt.legend()
plt.title("Ridge Regression (L2 Regularization Effect)")
plt.show()

