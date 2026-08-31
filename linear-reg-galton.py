import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

url = "https://vincentarelbundock.github.io/Rdatasets/csv/HistData/Galton.csv"
galton = pd.read_csv(url)

# Clean column names
galton.columns = [c.lower() for c in galton.columns]

X = galton["parent"].values.reshape(-1, 1)
y = galton["child"].values

model = LinearRegression()
model.fit(X, y)

print(f"Slope (Regression coefficient): {model.coef_[0]:.3f}")
print(f"Intercept: {model.intercept_:.3f}")

plt.figure(figsize=(8,6))
sns.scatterplot(x="parent", y="child", data=galton, alpha=0.5)
plt.plot(X, model.predict(X), color="red", lw=2, label=f"Slope = {model.coef_[0]:.2f}")
plt.xlabel("Mid-parent height (inches)")
plt.ylabel("Child height (inches)")
plt.title("Galton's Family Likeness in Stature (Real Data)")
plt.legend()
plt.show()

