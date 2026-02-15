import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import os

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

# --- Load housing data from local CSV ---
def load_housing_data(housing_path=HOUSING_PATH):
    """

    :param housing_path: Default value = HOUSING_PATH)

    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

# ---- Split dataset into training and test sets ----
def split_train_test(data, test_ratio):
    """

    :param data: param test_ratio:
    :param test_ratio: 

    """
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")

# Use only the training set for visualization and exploration
housing = train_set.copy()

# ---- Basic geographical visualization ----
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.title("California Housing Density (alpha=0.1)")
plt.show()

# ---- Visualize population density and price ----
housing.plot(
    kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population",
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
plt.title("Housing Prices by Location and Population")
plt.show()

# ---- Compute correlation matrix ----
corr_matrix = housing.corr(numeric_only=True)
print("\nCorrelation Matrix:")
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# ---- Scatter matrix for key attributes ----
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
plt.show()

# ---- Focused correlation scatter plot ----
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.title("Correlation: Median Income vs Median House Value")
plt.show()

