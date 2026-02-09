m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

from sklearn.preprocessing import PolynomialFeatures


# Transform features to polynomial (degree 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]
X_poly[0]

#fitting a linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_

# predict ne wpoints for curve
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)


# Plot original data + predicted curve
plt.figure(figsize=(8, 6))
plt.plot(X_new, y_new, "r-", linewidth=2, label="Prediction")
plt.plot(X, y, "b.", label="Training data")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polynomial Regression (degree=2)")
plt.legend()
plt.show()

