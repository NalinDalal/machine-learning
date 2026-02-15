import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("gdp_per_capita.csv", thousands=',', delimiter='\t', encoding='latin1', na_values="n/a")

# Prepare the data
def prepare_country_stats(oecd_bli, gdp_per_capita):
    """

    :param oecd_bli: param gdp_per_capita:
    :param gdp_per_capita: 

    """
    life_satisfaction = oecd_bli[oecd_bli["Indicator"] == "Life satisfaction"][["Country", "Value"]]
    life_satisfaction.rename(columns={"Value": "Life satisfaction"}, inplace=True)
    gdp = gdp_per_capita[["Country", "2015"]].rename(columns={"2015": "GDP per capita"})
    country_stats = pd.merge(life_satisfaction, gdp, on="Country")
    return country_stats.dropna()

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)

# Feature and target
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Scatter plot
country_stats.plot(kind='scatter', x="GDP per capita", y="Life satisfaction")
plt.show()

# Linear regression
lin_reg_model = LinearRegression()
lin_reg_model.fit(X, y)

# Predict Cyprus
X_new = [[22587]]  # Cyprus GDP per capita
print(lin_reg_model.predict(X_new))

