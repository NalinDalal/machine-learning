You are given a problem, now to solve it we can define a set of instruction or algorithms 

but how about a problem where you can't solve it with help of algorithms or an algorithm can't be simply defined for it??

Hence we introduced Machine Learning.

Machine Learning is great for:

- Problems for which existing solutions require a lot of hand-tuning or long lists of rules: one Machine Learning algorithm can often simplify code and perform better.
- Complex problems for which there is no good solution at all using a traditional approach: the best Machine Learning techniques can find a solution.
- Fluctuating environments: a Machine Learning system can adapt to new data.
- Getting insights about complex problems and large amounts of data.

Types:

1. UnSupervised
2. Supervised
3. Reinforcement

---

1. Supervised
   training data already has desired solution called **labels**
   A typical supervised learning task is **classification**.
   ex: spam filter, predictors/regression(price of car with features)
   Logistic Regression also is such example

some important supervised learning algorithms
• k-Nearest Neighbors
• Linear Regression
• Logistic Regression
• Support Vector Machines (SVMs)
• Decision Trees and Random Forests
• Neural networks2

2. UnSupervised
   training data is not labelled. The system tries to learn without a teacher.
   some UnSupervised learning algorithms

- Clustering
  - k-Means
  - Hierarchical Cluster Analysis (HCA)
  - Expectation Maximization
- Visualization and dimensionality reduction
  - Principal Component Analysis (PCA)
  - Kernel PCA
  - Locally-Linear Embedding (LLE)
  - t-distributed Stochastic Neighbor Embedding (t-SNE)
- Association rule learning
  - Apriori
  - Eclat

3.  SemiSupervised
    algorithm deals with partially labelled data

4.  Reinforcement
    The learning system, called an agent in this context, can observe the environment, select and perform actions, and get rewards/penalties in return .
    A policy defines what action the agent should choose when it is in a given situation.

        1. Instance Based Learning: the system learns the examples by heart, then generalizes to new cases using a similarity measure
            example: similarity b/w 2 mails to check for spam ones
        2. Model Based Learning: build model out of example then use that model

---

so we made a [python program](./linear-reg.py) to find and plot the data graph
you can see a trend there; looks like life satisfaction goes up more or less linearly as the country’s GDP per capita increases.
hence we will keep the model life selction as linear function of GDP per capita
this is called model selection

sample linear model:
`life_selection={theta_0}+$theta_1 * GDP_Per_Capita`

we use a cost function that measures the distance between the linear model’s predictions and the training examples;
the objective is to minimize this distance.

[training & running linear model via SciKit](./linear-model-scikit.py)

• You studied the data.
• You selected a model.
• You trained it on the training data (i.e., the learning algorithm searched for the model parameter values that minimize a cost function).
• Finally, you applied the model to make predictions on new cases (this is called inference), hoping that this model will generalize well

## Main Challenges

- Insufficient training data
- nonrepresentive training data
- poor quality of data
- irrelevant features
- over fitting{perofrm well on data but doesn't generalises} - reduce via regularization
- underfitting{model is too simple to learn about underlying structure of data}- feed better features, powerful model or reduce constraints

## Performance Measure

**Root Mean Square Error (RMSE)**

$$
\mathrm{RMSE}(\mathbf{X}, h) = \sqrt{ \frac{1}{m} \sum\_{i=1}^{m} \left( h(\mathbf{x}^{(i)}) - y^{(i)} \right)^2 }
$$

**Mean Absolute Error (MAE)**

$$
\mathrm{MAE}(\mathbf{X}, h) = \frac{1}{m} \sum\_{i=1}^{m} \left| h(\mathbf{x}^{(i)}) - y^{(i)} \right|
$$

## WorkSpace

well you gotta start

create your workspace, and install pip locally
create isolated environment there via adding a virtual environment

```sh
python3 -m venv myenv   #myenv is name of environment
source myenv/bin/activate   #to activate environment
```

u can now install dependencies easily

example code:
[to fetch data, plot and show the data](./housing-linear-reg.py)

well you can train and test the dataset too

### Create a Test Set

well you don't need to write new like test cases
just pick random, like 20% of given dataset, set them aside

```python
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
return data.iloc[train_indices], data.iloc[test_indices]
```

use them like:

```python
train_set, test_set = split_train_test(housing, 0.2)
print(len(train_set), "train +", len(test_set), "test")
```

### [Visualise data](./visua-rel.py)

make sure test data is set aside, you will use only training set here strictly.

so you wanna visualise dataset:

```python
housing.plot(kind="scatter", x="longitude", y="latitude")
```

generates a geographical scatterplot of the data

visualise places where high density of data points:

```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
```

now you can spot the pattern that darker is like high population

The radius of each circle represents the district’s population (option s), and the color represents the price (option c). We will use a predefined color map (option cmap) called `jet`, which ranges from blue (low values) to red (high prices):15

```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=housing["population"]/100, label="population",
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
```

### Looking for Correlations

you can easily compute the standard correlation coefficient (also called Pearson’s r) between every pair of attributes using the `corr()` method:
`corr_matrix = housing.corr()`

ranges b/w -1 to 1

other way to check for co-relation b/w attributes: `scatter_matrix`(plots every numerical attribute against every other numerical attribute)

```python
from pandas.tools.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms","housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
```

correlation scatter plot:

```python
housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1)
```

## Prepare the Data for Machine Learning Algorithms

we should prep our data, use functions instead of manual cause we can:

- reproduce transformations easily on any dataset
- build a library of transformation functions that you can reuse in future projects.
- reuse function in live system b/f feeding to algo

first let’s revert to a clean training set (by copying strat_train_set once again),
and let’s separate the predictors and the labels since we don’t necessarily want to apply
the same transformations to the predictors and the target values

```python
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```

### Data Cleaning

noticed earlier that the `total_bedrooms` attribute has some missing values, so let’s fix this. You have three options:

- Get rid of the corresponding districts.
- Get rid of the whole attribute.
- Set the values to some value (zero, the mean, the median, etc.).

```python
housing.dropna(subset=["total_bedrooms"]) # option 1
housing.drop("total_bedrooms", axis=1) # option 2
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median) # option 3
```

if using option3 then make sure to save the median value and use it

scikit has a handy class to take care of missing values: `Imputer`.

```python
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
```

Since the median can only be computed on numerical attributes, we need to create a copy of the data without the text attribute ocean_proximity:

```python
housing_num = housing.drop("ocean_proximity", axis=1)
```

fit the imputer instance to the training data using the `fit()` method:

```python
imputer.fit(housing_num)
X = imputer.transform(housing_num)
```

we can;t be sure where data is missing, so apply to whole dataset
hence now your training set is complete
result is a plain Numpy array containing the transformed features. If you want to
put it back into a Pandas DataFrame, it’s simple:

```python
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
```

### Handling Text & Categorical Attributes

basically convert text-labels to numbers, we use a transformer for this task called LabelEncoder:

```python3
>>> from sklearn.preprocessing import LabelEncoder
>>> encoder = LabelEncoder()
>>> housing_cat = housing["ocean_proximity"]
>>> housing_cat_encoded = encoder.fit_transform(housing_cat)
>>> housing_cat_encoded
array([1, 1, 4, ..., 1, 0, 3])
```

now consider after one-hot encoding we get a matrix with thousands of columns, and the matrix is full of zeros except for one 1 per row.

```python
housing_cat_1hot.toarray()
```

```
array([[ 0., 1., 0., 0., 0.],
[ 0., 1., 0., 0., 0.],
[ 0., 0., 0., 0., 1.],
...,
[ 0., 1., 0., 0., 0.],
[ 1., 0., 0., 0., 0.],
[ 0., 0., 0., 1., 0.]])
```

apply this transformation in 1 shot via `LabelBinarizer` class:

```python
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot
array([[0, 1, 0, 0, 0],
[0, 1, 0, 0, 0],
[0, 0, 0, 0, 1],
...,
[0, 1, 0, 0, 0],
[1, 0, 0, 0, 0],
[0, 0, 0, 1, 0]])
```

### Custom Transformers

Used when built-in transformers aren’t enough (e.g., custom cleanup or new feature creation).

**Requirements:**

- Implement 3 methods:
  - `fit(self, X, y=None)` → returns `self`
  - `transform(self, X)` → returns transformed data
  - (Optionally) inherit from `TransformerMixin` to get `fit_transform()` for free

- Inherit from `BaseEstimator` to gain `get_params()` & `set_params()` for hyperparameter tuning.
- Avoid `*args` / `**kwargs` in constructor → keeps auto-tuning compatible.

**Example:**

```python
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
```

**Takeaway:**

- Add hyperparameters for uncertain steps (e.g., `add_bedrooms_per_room`) → allows grid search optimization.

---

### Feature Scaling

Machine Learning algorithms perform poorly when feature scales differ.

#### 1. **Min-Max Scaling (Normalization)**

- Scales values between 0–1.
- Formula:
  ( X' = (X - X*{min}) / (X*{max} - X\_{min}) )
- Use: `MinMaxScaler(feature_range=(0,1))`
- Sensitive to outliers.

#### 2. **Standardization**

- Subtract mean, divide by standard deviation.
- Result → mean 0, variance 1.
- Use: `StandardScaler()`
- Not bounded but **robust to outliers.**

**Note:** Always `fit` on training data only, then `transform` train/test.

---

### Transformation Pipelines

#### ➤ Purpose:

To chain multiple preprocessing steps together.

#### Example:

```python
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
```

- Each step = (name, transformer)
- All but last must be transformers.
- `fit()` runs each step sequentially.
- `transform()` applies all transformations in order.

---

### Combining Pipelines with `FeatureUnion`

- Runs multiple pipelines in parallel and concatenates results.

```python
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])
```

Each subpipeline can select specific attributes using a **custom DataFrameSelector**.

#### Example:

```python
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None): return self
    def transform(self, X): return X[self.attribute_names].values
```

---

## Model Training & Evaluation

### **1. Linear Regression**

```python
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```

Evaluate with RMSE:

```python
predictions = lin_reg.predict(housing_prepared)
rmse = np.sqrt(mean_squared_error(housing_labels, predictions))
```

→ Underfitting if RMSE is high.

---

### **2. Decision Tree Regressor**

```python
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
```

If RMSE = 0 → Overfitting.

---

### **3. Cross-Validation**

Use to get reliable estimates:

```python
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
```

Displays performance + variability.

---

### **4. Random Forest Regressor**

```python
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
```

Better generalization than Decision Tree (ensemble of trees).

---

## Model Persistence

Save trained models:

```python
from sklearn.externals import joblib
joblib.dump(model, "model.pkl")
model = joblib.load("model.pkl")
```

---

## Fine-Tuning Models

### **1.Grid Search**

Brute-force hyperparameter search using cross-validation.

```python
param_grid = [
  {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
  {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2,3,4]},
]
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
```

- `grid_search.best_params_` → best combination
- `grid_search.best_estimator_` → retrained final model

---

### **2. Randomized Search**

When hyperparameter space is large.

```python
from sklearn.model_selection import RandomizedSearchCV
RandomizedSearchCV(..., n_iter=1000)
```

- Tries random combinations.
- Better control over compute budget.

---

## Ensemble Methods

Combine best models → usually better performance (e.g., Random Forest).
We’ll explore deeper in ensemble learning chapter.

---

## Model Analysis

Inspect feature importance:

```python
importances = grid_search.best_estimator_.feature_importances_
sorted(zip(importances, attributes), reverse=True)
```

→ Helps identify key predictive features (e.g., `median_income`).

---

## Final Evaluation on Test Set

Use unseen data:

```python
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
```

⚠️ Use `.transform()`, **not** `.fit_transform()`.

---

## Launch, Monitor & Maintain

- Deploy system → connect real data sources.
- Add **monitoring** for:
  - Performance degradation (data drift)
  - Input data quality

- Retrain periodically as data evolves.
- Include human evaluation loops if needed (e.g., via crowdsourcing).

---

## Summary Table

| Step                | Tool / Concept                                                       | Purpose                     |
| ------------------- | -------------------------------------------------------------------- | --------------------------- |
| Custom Transformers | `BaseEstimator`, `TransformerMixin`                                  | Add custom data prep logic  |
| Feature Scaling     | `MinMaxScaler`, `StandardScaler`                                     | Normalize features          |
| Pipeline            | `Pipeline()`                                                         | Sequential preprocessing    |
| Combine             | `FeatureUnion()`                                                     | Parallel pipelines          |
| Model Training      | `LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor` | Train regressors            |
| Evaluation          | `cross_val_score`, RMSE                                              | Measure performance         |
| Save Models         | `joblib`                                                             | Serialize models            |
| Fine-Tune           | `GridSearchCV`, `RandomizedSearchCV`                                 | Hyperparameter optimization |
| Analyze             | `feature_importances_`                                               | Understand predictors       |
| Final Test          | `transform()`, RMSE                                                  | Evaluate generalization     |
| Production          | Monitoring + Retraining                                              | Maintain performance        |
