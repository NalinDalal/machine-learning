import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dataSetURL = "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Galton.csv"
galton = pd.read_csv(dataSetURL)

# Clean column names
galton.columns = [c.lower() for c in galton.columns]  # lowerise them

X = galton["parent"].values.reshape(-1, 1)  # reshape to be a colum vector
y = galton["child"].values  # target variable

model = LinearRegression()  # create a linear regression model from package
model.fit(X, y)  # fit the model on the data variables

print(f"Slope (Regression coefficient): {model.coef_[0]:.3f}")
print(f"Intercept: {model.intercept_:.3f}")

plt.figure(figsize=(8, 6))  # create a figure of size 8x6
sns.scatterplot(
    x="parent", y="child", data=galton, alpha=0.5
)  # name the fields on graph
plt.plot(
    X, model.predict(X), color="red", lw=2, label=f"Slope = {model.coef_[0]:.2f}"
)  # plot the regression line
plt.xlabel("Mid-parent height (inches)")
plt.ylabel("Child height (inches)")
plt.title("Galton's Family Likeness in Stature (Real Data)")
plt.legend()
plt.show()
