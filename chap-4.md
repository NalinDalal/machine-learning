# Training models

well you saw the algorithm, trained them and wrote your own system without knowing th einternal
let's take a look at linear regression{most basic algorithm}
2 ways to train it:

- closed-form equation directly computing model parameters
- iterative optimization that gradually tweaks model params to minimise cost functions.

1. Linear Regression
2. regularization techniques
3. Logistic Regression
4. SOftmax Regression

## Linear Regression

previously we looked at:

$$
[
life_satisfaction=theta_0+theta_1*GDP_Per_Capita
]
$$

$$
\text{life_satisfaction} = \theta_0 + \theta_1 \times \text{GDP_per_capita}
$$

model is just a **linear function** of the input feature `GDP_per_capita`.
$\theta_0$ and $\theta_1$ are model params

A prediction is made by computing a **weighted sum of the input features**, plus a constant called the **bias term**.

### Linear Regression(model prediction)

$$
y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
$$

where:

- ( y ): predicted value
- ( n ): number of features
- ( x_i ): i-th feature value
- ( \theta_j ): j-th model parameter

### Linear Regression(vectorised form)

$$
\hat{y} = h_\theta(x) = \theta^T \cdot x
$$

Where:

- ( \theta ): model parameter vector
- ( \theta^T ): transpose of θ
- ( x ): instance’s feature vector
- ( x_0 = 1 ) (bias feature)
- ( h\_\theta(x) ): **hypothesis function**

### MSE cost function of linear regression model

$$
MSE(X, h_\theta) = \frac{1}{m} \sum_{i=1}^{m} \left( \theta^T x^{(i)} - y^{(i)} \right)^2
$$

Where ( m ) is the number of training instances.

**Normal Equation**
mathematical equation that minimises the cost function

$$
\hat{\theta} = (X^T X)^{-1} X^T y
$$

theta cap=(X riased to T _ X) raised to -1 _ X raised to T \* y
-( \hat{\theta} )=value of theta minimising cost function

- X = matrix of features
- y vector = vector of target values containing y(1) to y(m)

doing it with help of algorithms

**step 1:**
generate linear looking data

```python
import numpy as np
X=2*np.random.rans(100,1)
y=4 + 3 * X + np.random.randn(100, 1)
```

**step 2:**
compute theta_cap via normal equation
use `inv()` to find inverse of matrix; `dot()` to do matrix multiplication

```python
X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
```

now do predictions:

```python
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new] # add x0 = 1 to each instance
y_predict = X_new_b.dot(theta_best)
y_predict
```

to plot the prediction :

```python
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()
```

complexity goes from

$$
O(n^{2.4}) \text{ to } O(n^3)
$$

## Gradient Descent

general idea of Gradient Descent is to tweak parameters iteratively in order to minimize a cost function.

exactly what is does:
measures the local gradient of the error function with regards to the
parameter vector θ, and it goes in the direction of descending gradient. Once the gradient is zero, you have reached a minimum!

you feel theta with random values and gradually improve it over no of iterations until the algorithm converges
to a minimum.

an important parameter: **learning rate hyperparameter**
if too small then so many iterations to converge{takes long time}
if too large might jump across the valley, never reaching the minima

hence we use the MSE function to converge it
pick 2 points on curve, join them, if line never hits the curve then there are no local minima, only a single global minima

### Batch Gradient Descent

calulate how much cost function will change if you change theta_j just a little bit{partial derivative}

find **partial derivative**:

$$
\frac{\partial}{\partial \theta_j} MSE(\boldsymbol{\theta}) =
\frac{2}{m} \sum_{i=1}^{m}
\left(
\boldsymbol{\theta}^T \cdot \mathbf{x}^{(i)} - y^{(i)}
\right)
x_j^{(i)}
$$

**gradient vector of cost function**

$$
\nabla_{\boldsymbol{\theta}} MSE(\boldsymbol{\theta}) =
\begin{pmatrix}
\frac{\partial}{\partial \theta_0} MSE(\boldsymbol{\theta}) \
\frac{\partial}{\partial \theta_1} MSE(\boldsymbol{\theta}) \
\vdots \
\frac{\partial}{\partial \theta_n} MSE(\boldsymbol{\theta})
\end{pmatrix}
=============

\frac{2}{m}
\mathbf{X}^T \cdot
(\mathbf{X} \cdot \boldsymbol{\theta} - \mathbf{y})
$$

**Gradient Descent Step**

$$
\boldsymbol{\theta}^{(\text{next step})} =
\boldsymbol{\theta} -
\eta ,
\nabla_{\boldsymbol{\theta}} MSE(\boldsymbol{\theta})
$$

_Where:_

- ( \eta ) = learning rate
- ( m ) = number of training instances
- ( \mathbf{X} ) = matrix of input features
- ( \mathbf{y} ) = vector of target values
- ( \boldsymbol{\theta} ) = parameter vector

implementation:

```python
eta = 0.1 # learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2,1) # random initialization

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print(theta)
```

```
array([[ 4.21509616],[ 2.77011339]])
```

### Stochastic Gradient Descent

problem with Gradient Descent: uses whole training set at each step and computes the gradients based only on that single instance.

fast cause low data to iterate over
but less regular, cost jumps are too unpredictable
When the cost function is very irregular, this can actually help the algorithm jump out of local minima, so Stochastic Gradient Descent has a better chance of finding the global minimum than Batch Gradient Descent does.

but algo then stays on dilemma, so gradually reduce the learning rate. to settle at global minima
process is called simulated annealing.

function that determines the learning rate at each iteration is called the learning schedule.

```python
n_epochs = 50
t0, t1 = 5, 50 # learning schedule hyperparameters
def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1) # random initialization

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch * m + i)
        theta = theta - eta * gradients
```

to perform via scikit:

```python
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(n_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())
```

## Polynomial Regression

what if not linear, maybe more complex data
use polymonial regression for it:
add powers of each feature as new features, then train a linear model on this extended set of features

```python
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
X[0]
X_poly[0]

#fitting a linear regression model
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
```

## Learning Curves

plots of the model’s performance on the training set and the validation set as a function of the training set size.

```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []

    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")

#learning curve of model
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
```

## Regularised Linear Models

For a linear model, regularization is typically achieved by constraining the weights of the model.
some of them are:

- Ridge Regression
- Lasso Regression
- Elastic Net

### Ridge Regression (L2 Regularization)

Ridge Regression (or **Tikhonov Regularization**) is a **regularized version of Linear Regression** that penalizes large model weights to prevent overfitting.

It minimizes a **modified cost function**:

$$
J(\theta) = MSE(\theta) + \frac{\alpha}{2} \sum_{i=1}^{n} \theta_i^2
$$

- ( \alpha ) → regularization strength (hyperparameter)
- ( \theta_i ) → model parameters (excluding bias term ( \theta_0 ))
- The term ( \frac{\alpha}{2} \sum\_{i=1}^{n} \theta_i^2 ) is the **L2 penalty**
- Keeps weights small and smoothens the model

Bias term ( \theta_0 ) is **not regularized** (sum starts from ( i=1 )).

In vector form, if ( w = [\theta_1, \theta_2, ..., \theta_n] ):

$$
J(\theta) = MSE(\theta) + \frac{\alpha}{2} | w |_2^2
$$

### Ridge Regression (Closed-form solution)

Ridge Regression has a closed-form solution similar to the Normal Equation:

$$
\hat{\theta}_{ridge} = (X^T X + \alpha I)^{-1} X^T y
$$

where:

- ( I ) is the **identity matrix**
- ( \alpha I ) ensures matrix ( (X^T X + \alpha I) ) is invertible
- As ( \alpha \to 0 ), solution → regular Linear Regression
- As ( \alpha \to \infty ), coefficients shrink toward 0

```python
from sklearn.linear_model import Ridge
ridge_reg = Ridge(alpha=1, solver="cholesky")
ridge_reg.fit(X, y)
ridge_reg.predict([[1.5]])
```

You can also use **Stochastic Gradient Descent** (SGD) with an L2 penalty:

```python
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(penalty="l2", alpha=0.1)
sgd_reg.fit(X, y.ravel())
```

#### Intuition

- Adds a constraint: weights must stay small.
- Reduces model variance → less overfitting.
- But increases bias slightly.
- Works well when:
  - You have **many correlated features**.
  - You want a **simpler, more stable** model.

#### Visual intuition

- L2 penalty forms **circular contours** (unlike Lasso’s diamond-shaped ones).
- Gradient descent moves toward where the cost (MSE + penalty) is minimized.
- As ( \alpha ) increases, the model becomes smoother but less flexible.

### Lasso Regression

**LASO** (Least Absolute Shrinkage and Selection Operator) Regression adds a regularization term to the cost function, but it uses the **ℓ1 norm** of the weight vector instead of half the square of the ℓ2 norm.

#### Cost Function

It minimizes a **modified cost function**:

$$
J(\theta) = MSE(\theta) + \alpha \sum_{i=1}^{n} |\theta_i|
$$

- ( \alpha ) → regularization strength
- ( \theta_i ) → model parameters (excluding bias term ( \theta_0 ))
- ( \sum\_{i=1}^{n} |\theta_i| ) → **L1 penalty** (absolute value)
- Forces some coefficients to **exactly 0**, performing **feature selection** automatically

The bias term ( \theta_0 ) is **not regularized**.

#### Intuition

- Encourages **sparsity** — eliminates less important features by setting their weights to 0
- Performs **feature selection + regularization**
- Good for **high-dimensional data** (many correlated or redundant features)
- The stronger the regularization (higher α), the **more coefficients go to zero**

#### Vector Form

If ( w = [\theta_1, \theta_2, ..., \theta_n] ),

$$
J(\theta) = MSE(\theta) + \alpha | w |_1
$$

where ( | w |\_1 = \sum_i |\theta_i| )

#### Geometric Intuition

- **L1 penalty contours** are **diamond-shaped**, unlike Ridge’s circular contours.
- Gradient descent paths tend to **hit corners (axes)** where some ( \theta_i = 0 ).
- Hence, Lasso often gives **sparse models** (few nonzero coefficients).

[scikit-example](./scikit-lasso.py)
[SGD-example](./sdg-laso.py)
[visualise the effect](./laso-reg-visu.py)

| Aspect   | Ridge (L2)           | Lasso (L1)               |          |     |
| -------- | -------------------- | ------------------------ | -------- | --- |
| Penalty  | ( \sum \theta_i^2 )  | ( \sum                   | \theta_i | )   |
| Effect   | Shrinks coefficients | Shrinks & sets some to 0 |          |     |
| Model    | Smooth               | Sparse                   |          |     |
| Use Case | Correlated features  | Feature selection        |          |     |

### Elastic Net

Elastic Net is a middle ground between Ridge Regression and Lasso Regression.
an `r` parameter decides the mix

`r=0` -> Ridge
`r=1` -> Lasso

```python
from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
elastic_net.predict([[1.5]])
```

### Early Stopping

different way to regularise algo like Gradient Descent is to stop training as soon as the validation error reaches a minimum.
known as early stopping.

1.  **Train and watch the validation error**
    While training your model (like with Gradient Descent), keep checking how well it performs on a **validation set** — not just the training set.

2.  **Stop when validation error starts rising**
    At first, both training and validation errors go down. But after a point, the model starts **overfitting** — training error still decreases, but **validation error increases**.
    -> That’s your signal to **stop training right there**.

3.  **Why it works**
    Early stopping prevents the model from “memorizing” the training data. It’s like catching the perfect balance between **underfitting** and **overfitting** — a free regularization trick.

> “Train until your model starts getting worse on validation — then stop.”

```python
from sklearn.base import clone
sgd_reg = SGDRegressor(n_iter=1, warm_start=True, penalty=None,learning_rate="constant", eta0=0.0005)

minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train_poly_scaled, y_train) # continues where it left off
    y_val_predict = sgd_reg.predict(X_val_poly_scaled)
    val_error = mean_squared_error(y_val_predict, y_val)

    if val_error < minimum_val_error:
        minimum_val_error = val_error
    best_epoch = epoch
    best_model = clone(sgd_reg)

```

## Logistic Regression

well it's like model predicts if an instance belongs to a class or not
if probability>50% then 1{belongs}
else 0{not belongs}

to find probabilities:
we use a sigmoid function:

probability:

[
\hat{p} = P(y = 1 \mid x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
]

Where:

- ( \hat{p} ) → predicted probability that the output is 1
- ( \theta ) → model parameters (weights + bias)
- ( x ) → input features vector
- ( \sigma ) → sigmoid function

sigmoid:

[
\sigma(x) = \frac{1}{1 + e^{-x}}
]

## Training & Cost Function

#### 1. Probability function

[
\hat{p} = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
]

#### 2. Prediction rule

[
\hat{y} =
\begin{cases}
1 & \text{if } \hat{p} \geq 0.5 \
0 & \text{if } \hat{p} < 0.5
\end{cases}
]

#### 3. Cost function (Log Loss)

[
J(\theta) = -\frac{1}{m} \sum\_{i=1}^{m} [y^{(i)} \log(\hat{p}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)})]
]

## Decision Boundary

A **decision boundary** is the line (in 2D), plane (in 3D), or hyperplane (in nD) that **separates the predicted classes** in a classification problem.

It’s where the model is **“undecided”** — i.e. the probability of belonging to class 1 is exactly **0.5**.

- It’s the **line** where the classifier switches its decision.
- Logistic regression always forms a **linear decision boundary** (a straight line or plane).
- Nonlinear models (like polynomial or neural nets) can form **curved** decision boundaries.

#### For Logistic Regression:

We know:
[
\hat{p} = \sigma(\theta^T x)
]

To find the **decision boundary**, set
[
\hat{p} = 0.5
]

Since
[
\sigma(z) = 0.5 \text{ when } z = 0
]

we get:
[
\theta^T x = 0
]

That’s the **equation of the decision boundary**.

#### Example

If
[
\theta_0 + \theta_1 x_1 + \theta_2 x_2 = 0
]

then all points **on this line** have equal probability of being class 0 or class 1.

- Points on one side → predicted **class 1**
- Points on the other side → predicted **class 0**

## **Softmax Regression**

**Softmax Regression** is a **generalization of Logistic Regression** for **multiclass classification** (when you have more than two classes).
Instead of outputting a single probability, it gives a **probability distribution** over all possible classes.

### **Hypothesis Function**

For each class ( k ), compute a **score**:

[
s_k(x) = \theta_k^T x
]

Then apply the **softmax function** to convert these scores into probabilities:

[
\hat{p}*k = \frac{e^{s_k(x)}}{\sum*{j=1}^{K} e^{s_j(x)}}
]

where:

- ( K ) = total number of classes
- ( \hat{p}\_k ) = probability that the instance belongs to class ( k )

### **Prediction Rule**

Choose the class with the **highest probability**:

[
\hat{y} = \arg\max_k \hat{p}_k
]

### **Cost Function (Cross-Entropy Loss)**

To train the model, minimize the **cross-entropy cost**:

[
J(\Theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\hat{p}_k^{(i)})
]

where:

- ( y_k^{(i)} ) = 1 if sample ( i ) belongs to class ( k ), else 0
- ( \hat{p}\_k^{(i)} ) = predicted probability for class ( k )

### **Intuition**

- Each class gets its **own parameter vector** ( \theta_k ).
- The softmax function ensures **all probabilities sum to 1**.
- The model learns to assign **high probability to the correct class**.
- Cross-entropy loss penalizes **wrong confident predictions**.

### **Example (Scikit-Learn)**

```python
from sklearn.linear_model import LogisticRegression
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X_train, y_train)

# predict class probabilities
softmax_reg.predict_proba(X_new)
```
