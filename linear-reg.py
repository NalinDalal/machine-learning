import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.cm as cm

# --- Load data ---
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")

# --- Prepare the data ---
def prepare_country_stats(oecd_bli, gdp_per_capita):
    """

    :param oecd_bli: param gdp_per_capita:
    :param gdp_per_capita: 

    """
    # Life satisfaction
    life_satisfaction = oecd_bli[oecd_bli["Indicator"] == "Life satisfaction"][["Country", "Value"]]
    life_satisfaction.rename(columns={"Value": "Life satisfaction"}, inplace=True)
    
    # GDP per capita
    gdp = gdp_per_capita[["Country", "2015"]].rename(columns={"2015": "GDP per capita"})
    
    # Merge datasets
    country_stats = pd.merge(life_satisfaction, gdp, on="Country")
    return country_stats.dropna()

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = country_stats[["GDP per capita"]].values
y = country_stats["Life satisfaction"].values
countries = country_stats['Country'].tolist()

# --- Scatter plot with colors ---
plt.figure(figsize=(8,6))
color_map = cm.get_cmap('tab10', len(countries))  # generate enough colors automatically
for i, country in enumerate(countries):
    plt.scatter(X[i], y[i], color=color_map(i), label=country, s=100)
plt.xlabel("GDP per capita")
plt.ylabel("Life satisfaction")
plt.title("Life satisfaction vs GDP per capita")
plt.legend()
plt.show()

# --- Train linear regression ---
lin_reg_model = LinearRegression()
lin_reg_model.fit(X, y)

# --- Plot regression line ---
x_fit = np.linspace(X.min(), X.max(), 100).reshape(100, 1)
y_fit = lin_reg_model.predict(x_fit)

plt.figure(figsize=(8,6))
plt.plot(x_fit, y_fit, color='black', linewidth=2, label="Regression line")
for i, country in enumerate(countries):
    plt.scatter(X[i], y[i], color=color_map(i), s=100)
    plt.text(X[i]+200, y[i], country)  # offset for readability

plt.xlabel("GDP per capita")
plt.ylabel("Life satisfaction")
plt.title("Life satisfaction vs GDP per capita with Regression Line")
plt.show()

# --- Predict for Cyprus ---
cyprus_gdp = [[22587]]
predicted_ls = lin_reg_model.predict(cyprus_gdp)
print(f"Predicted Life Satisfaction for Cyprus: {predicted_ls[0]:.2f}")

