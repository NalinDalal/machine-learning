
# lasso_regression.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Generate quadratic data
np.random.seed(42)
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# Pipelines for Lasso with different alphas
lasso_1 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lasso_reg", Lasso(alpha=0.01, max_iter=10000))
])

lasso_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lasso_reg", Lasso(alpha=1, max_iter=10000))
])

# Fit
lasso_1.fit(X, y)
lasso_10.fit(X, y)

# Plot
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
plt.plot(X, y, "b.", alpha=0.3)
plt.plot(X_new, lasso_1.predict(X_new), "r-", linewidth=2, label="α=0.01 (less regularized)")
plt.plot(X_new, lasso_10.predict(X_new), "k-", linewidth=2, label="α=1 (more regularized)")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Lasso Regression (L1 Regularization Effect)")
plt.legend()
plt.show()

