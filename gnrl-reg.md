# Generalization and Regularization

alright so we've built these fancy neural networks, but here's the thing - just because a model fits the training data well doesn't mean it'll work on new data. this is where **generalization** comes in.

## Core Problem

we typically learn a model by minimizing **training loss**:
$$J(\theta) = \frac{1}{n}\sum_{i=1}^n (y^{(i)} - h_\theta(x^{(i)}))^2$$

but what we ACTUALLY care about is **test error** on unseen examples:
$$L(\theta) = E_{(x,y)\sim D}[(y - h_\theta(x))^2]$$

where $D$ is the test distribution.

**Problem:** even if training error is small, test error might be huge! this is called **OverFitting**.

**UnderFitting:** when training error is large (model too simple)
**OverFitting:** when training error is small but test error is large (model too complex)

## Bias-Variance Tradeoff

### the running example

imagine we have data where $y^{(i)} = h^*(x^{(i)}) + \xi^{(i)}$ where $h^*$ is some quadratic function and $\xi \sim N(0, \sigma^2)$ is noise.

**Fitting a linear model:**

- even with infinite data, a linear model can't capture the quadratic relationship
- this is **large bias** - the model family fundamentally can't represent the true function
- leads to **underfitting**

**Fitting a 5th-degree polynomial:**

- with small data, it fits spurious patterns from the noise
- different training sets give wildly different models
- this is **large variance** - the model is too sensitive to the specific training data
- leads to **overfitting**

**Fitting a Quadratic Model:**

- just right! matches the true function's complexity
- achieves the best bias-variance tradeoff

### Mathematical Decomposition (optional but cool)

for regression, suppose:

- training set $S = \{x^{(i)}, y^{(i)}\}$ where $y^{(i)} = h^*(x^{(i)}) + \xi^{(i)}$
- we learn model $\hat{h}_S$
- test point $(x, y)$ where $y = h^*(x) + \xi$

the mean squared error decomposes as:
$$\text{MSE}(x) = E_{S,\xi}[(y - h_S(x))^2]$$

we can show:
$$\text{MSE}(x) = \sigma^2 + (h^*(x) - h_{\text{avg}}(x))^2 + \text{var}(h_S(x))$$

where $h_{\text{avg}}(x) = E_S[h_S(x)]$ is the "average model" over all datasets.

**breakdown:**

- $\sigma^2$ = unavoidable noise (can't predict randomness)
- $(h^*(x) - h_{\text{avg}}(x))^2$ = **bias²** (model family can't capture true function)
- $\text{var}(h_S(x))$ = **variance** (sensitivity to specific training data)

### Tradeoff

typically there's a tradeoff:

```
Error
  ^
  |     Bias²
  |    /
  |   /  \  Variance
  |  /    \
  | /______\_____ Test Error
  |/
  +---------------> Model Complexity
     optimal
```

- **simple models:** high bias, low variance
- **complex models:** low bias, high variance
- **sweet spot:** balanced tradeoff

## Double Descent phenomenon

BUT WAIT - recent research shows the story is more complicated!

### Model-wise Double Descent

as we increase model complexity (# parameters):

1. error decreases (under-parameterized regime)
2. error peaks around when # parameters ≈ # training examples
3. error decreases AGAIN (over-parameterized regime)

```
Test Error
  ^
  |     /\
  |    /  \
  |   /    \___
  |  /          \___
  +-----|-----------|-----> # Parameters
        n     interpolation
              threshold
   classical     modern
   regime        regime
```

**the peak:** happens when model is just big enough to fit training data perfectly but not bigger

**the descent:** with MORE parameters, test error can actually improve!

### Sample-wise Double Descent

even weirder - sometimes MORE training data can temporarily HURT performance!

as we increase training size $n$:

1. error decreases
2. error peaks around $n \approx d$ (# parameters)
3. error decreases again

**why?** around $n \approx d$, standard training algorithms struggle. they fit the data but in a brittle way.

### what's happening?

the culprit is **implicit regularization** from the optimizer. even without explicit regularization, gradient descent tends to find "nice" solutions in the over-parameterized regime.

**practical implication:** don't be scared of big models! in the over-parameterized regime, bigger can be better.

**the catch:** this peak at $n \approx d$ suggests our algorithms are sub-optimal there. with proper regularization (next section!), we can smooth out this peak.

## Regularization

**Regularization** = adding constraints to prevent overfitting

### Basic Idea

instead of minimizing $J(\theta)$, minimize:
$$J_\lambda(\theta) = J(\theta) + \lambda R(\theta)$$

where:

- $R(\theta)$ = regularizer (measures model complexity)
- $\lambda \geq 0$ = regularization parameter (controls strength)

**intuition:** we want a model that both fits the data AND has low complexity.

### $\ell_2$ regularization (weight decay)

most common: $R(\theta) = \frac{1}{2}\|\theta\|_2^2$

gradient descent becomes:
$$\theta \leftarrow \theta - \eta\nabla J_\lambda(\theta)$$
$$= \theta - \eta\lambda\theta - \eta\nabla J(\theta)$$
$$= (1 - \lambda\eta)\theta - \eta\nabla J(\theta)$$

the term $(1 - \lambda\eta)$ **decays** the weights each step - hence "weight decay"!

### $\ell_1$ regularization (LASSO)

$R(\theta) = \|\theta\|_1$ encourages **sparsity** (many $\theta_i = 0$)

**why?** if we believe the true model only uses a few features, $\ell_1$ helps find it.

note: $\|\theta\|_0$ (# non-zeros) is what we really want, but it's not differentiable. $\|\theta\|_1$ is a good continuous surrogate.

### other regularization techniques

**in deep learning:**

- dropout (randomly drop neurons during training)
- data augmentation (create more training data)
- batch normalization
- early stopping
- and many more...

### Implicit Regularization

here's something mind-blowing: **the optimizer itself regularizes!**

even without explicit $R(\theta)$, gradient descent has preferences:

- in over-parameterized linear models, it finds the minimum norm solution
- in neural networks, it tends to find "flatter" minima

**why does this matter?**

- explains why big neural networks don't overfit as badly as theory predicts
- different optimizers (SGD, Adam) have different implicit biases
- even if training loss is zero, choice of optimizer affects test error!

**practical insight:** tuning your optimizer (learning rate, batch size, momentum) isn't just about training speed - it affects generalization!

## Model Selection

okay so we have this regularization parameter $\lambda$. or maybe we're choosing between different model architectures. how do we pick?

### Hold-Out Cross Validation

simplest approach:

1. split data into training (70%) and validation (30%)
2. train each model on training set
3. pick model with lowest validation error
4. optionally: retrain on full dataset

**pros:** simple, fast
**cons:** "wastes" 30% of data

### K-Fold Cross Validation

better when data is scarce:

1. split data into $k$ folds (typically $k=10$)
2. for each model:
   - for each fold $j$:
     - train on all folds except $j$
     - test on fold $j$
   - average the $k$ test errors
3. pick model with lowest average error
4. retrain on full dataset

**pros:** uses all data, more reliable estimates
**cons:** $k$ times more expensive

### Leave-One-Out Cross Validation

extreme case: $k = n$ (each fold is one example)

- most expensive
- most reliable when data is very scarce

## Bayesian Statistics and Regularization

quick detour - there's a bayesian interpretation of regularization!

**frequentist view:** $\theta$ is unknown but fixed, we estimate it

**bayesian view:** $\theta$ is a random variable with a prior $p(\theta)$

given training data $S$, the posterior is:
$$p(\theta|S) = \frac{p(S|\theta)p(\theta)}{p(S)}$$

### MAP Estimation

instead of maximum likelihood:
$$\theta_{MLE} = \arg\max_\theta \prod_{i=1}^n p(y^{(i)}|x^{(i)}, \theta)$$

we do maximum a posteriori:
$$\theta_{MAP} = \arg\max_\theta \prod_{i=1}^n p(y^{(i)}|x^{(i)}, \theta) \cdot p(\theta)$$

**the connection:** if we use prior $\theta \sim N(0, \tau^2 I)$, then:
$$\theta_{MAP} = \arg\max_\theta \left[\sum_i \log p(y^{(i)}|x^{(i)}, \theta) - \frac{1}{2\tau^2}\|\theta\|_2^2\right]$$

this is exactly $\ell_2$ regularization! the prior $p(\theta)$ acts as regularization.

**practical note:** MAP with gaussian prior = $\ell_2$ regularization. this makes bayesian logistic regression work well even when features >> examples (like in text classification).

## putting it all together

**the workflow:**

1. choose model family (linear, neural net, etc.)
2. choose loss function (MSE, cross-entropy, etc.)
3. add regularization (weight decay, dropout, etc.)
4. use cross-validation to tune hyperparameters ($\lambda$, architecture, etc.)
5. train final model on all data
6. pray your test error is good 

**key insights:**

- bias-variance tradeoff is fundamental
- but double descent shows over-parameterization can be good!
- regularization (explicit and implicit) is crucial
- validation/cross-validation helps us choose the right complexity
- the optimizer itself acts as a regularizer

next up we'll dive into unsupervised learning where we don't even have labels!