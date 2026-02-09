from sklearn.linear_model import SGDRegressor

sgd_lasso = SGDRegressor(penalty="l1", alpha=0.1)
sgd_lasso.fit(X, y.ravel())

