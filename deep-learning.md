# Deep Learning

## the problem with linear models

so far we've been doing stuff like:

- linear regression: $h_\theta(x) = \theta^T x$
- linear with features: $h_\theta(x) = \theta^T \phi(x)$

both are **linear in the parameters** $\theta$. but what if we want something more flexible? something that's non-linear in both parameters AND inputs?

enter: **neural networks**

## Supervised Learning with non-linear models

we still have our training set $\{(x^{(i)}, y^{(i)})\}_{i=1}^n$, but now our hypothesis $h_\theta(x)$ can be some crazy non-linear function.

### Regression

where $y^{(i)} \in \mathbb{R}$, we still use mean squared error:

$$J^{(i)}(\theta) = \frac{1}{2}(h_\theta(x^{(i)}) - y^{(i)})^2$$

total cost:
$$J(\theta) = \frac{1}{n}\sum_{i=1}^n J^{(i)}(\theta)$$

### Binary Classification

where $y \in \{0,1\}$, we use a model $\bar{h}_\theta: \mathbb{R}^d \to \mathbb{R}$ that outputs a **logit**.

then we apply sigmoid to get probability:
$$h_\theta(x) = g(\bar{h}_\theta(x)) = \frac{1}{1 + e^{-\bar{h}_\theta(x)}}$$

and use logistic loss:
$$J^{(i)}(\theta) = -\log p(y^{(i)} | x^{(i)}; \theta) = \ell_{\text{logistic}}(\bar{h}_\theta(x^{(i)}), y^{(i)})$$

### Multi-Class Classification

where $y \in \{1, 2, ..., k\}$, our model $\bar{h}_\theta: \mathbb{R}^d \to \mathbb{R}^k$ outputs **logits** for each class.

apply softmax:
$$p(y = j | x; \theta) = \frac{\exp(\bar{h}_\theta(x)_j)}{\sum_{s=1}^k \exp(\bar{h}_\theta(x)_s)}$$

use cross-entropy loss:
$$J^{(i)}(\theta) = \ell_{ce}(\bar{h}_\theta(x^{(i)}), y^{(i)})$$

## Optimizers - `SGD and variants`

we optimize using gradient descent (GD) or stochastic gradient descent (SGD):

**gradient descent:**
$$\theta := \theta - \alpha\nabla_\theta J(\theta)$$

**Stochastic Gradient Descent (SGD):**

```
for i = 1 to n_iter:
    sample j uniformly from {1, ..., n}
    θ := θ - α∇_θ J^(j)(θ)
```

**mini-batch SGD:**

```
for i = 1 to n_iter:
    sample B examples j_1, ..., j_B
    θ := θ - (α/B) Σ_k ∇_θ J^(j_k)(θ)
```

mini-batch is usually fastest because we can parallelize on GPUs!

## Neural Networks - Building Blocks

### Single Neuron

let's start simple. for the housing price prediction problem, if we want to prevent negative prices, we can use:

$$\bar{h}_\theta(x) = \max(wx + b, 0)$$

where $\theta = (w, b)$ and $\max(t, 0)$ is called **ReLU** (Rectified Linear Unit).

for multi-dimensional input $x \in \mathbb{R}^d$:
$$\bar{h}_\theta(x) = \text{ReLU}(w^T x + b)$$

where $w \in \mathbb{R}^d$, $b \in \mathbb{R}$

**key terms:**

- $b$ = bias
- $w$ = weight vector
- this is a 1-layer network

### Stacking Neurons

now let's get fancy. say we have features: size, bedrooms, zip code, wealth.

we might think:

- family size depends on size + bedrooms
- walkability depends on zip code
- school quality depends on zip code + wealth

so we create intermediate variables (hidden units):
$$a_1 = \text{ReLU}(\theta_1 x_1 + \theta_2 x_2 + \theta_3)$$
$$a_2 = \text{ReLU}(\theta_4 x_3 + \theta_5)$$
$$a_3 = \text{ReLU}(\theta_6 x_3 + \theta_7 x_4 + \theta_8)$$

then combine them:
$$\bar{h}_\theta(x) = \theta_9 a_1 + \theta_{10} a_2 + \theta_{11} a_3 + \theta_{12}$$

but this requires domain knowledge. can we be more general?

### 2-layer Fully-Connected Neural Network

let's make each hidden unit depend on ALL inputs:

$$a_1 = \text{ReLU}(w_1^T x + b_1)$$
$$a_2 = \text{ReLU}(w_2^T x + b_2)$$
$$a_3 = \text{ReLU}(w_3^T x + b_3)$$

then:
$$\bar{h}_\theta(x) = \theta_9 a_1 + \theta_{10} a_2 + \theta_{11} a_3 + \theta_{12}$$

this is a **fully-connected neural network** because all $a_i$'s depend on all $x_i$'s.

### Vectorization - crucial for efficiency

instead of for loops, we use matrix operations!

define weight matrix $W^{[1]} \in \mathbb{R}^{m \times d}$:
$$W^{[1]} = \begin{bmatrix} - w_1^T - \\ - w_2^T - \\ \vdots \\ - w_m^T - \end{bmatrix}$$

then:
$$z = W^{[1]} x + b^{[1]}$$
$$a = \text{ReLU}(z)$$
$$\bar{h}_\theta(x) = W^{[2]} a + b^{[2]}$$

where $W^{[2]} \in \mathbb{R}^{1 \times m}$ and $b^{[2]} \in \mathbb{R}$

**why vectorization matters:** GPUs can do matrix operations in parallel, making this MUCH faster than for loops!

### Multi-layer Neural Networks

for an $r$-layer network:
$$a^{[1]} = \text{ReLU}(W^{[1]} x + b^{[1]})$$
$$a^{[2]} = \text{ReLU}(W^{[2]} a^{[1]} + b^{[2]})$$
$$\vdots$$
$$a^{[r-1]} = \text{ReLU}(W^{[r-1]} a^{[r-2]} + b^{[r-1]})$$
$$\bar{h}_\theta(x) = W^{[r]} a^{[r-1]} + b^{[r]}$$

**parameters:**

- total neurons: $m_1 + m_2 + ... + m_r$
- total parameters: $(d+1)m_1 + (m_1+1)m_2 + ... + (m_{r-1}+1)m_r$

note: typically no ReLU on the last layer so we can output negative numbers if needed

### Other Activation Functions

instead of ReLU, we can use:

**Sigmoid:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**tanh:**
$$\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**Leaky ReLU:**
$$\sigma(z) = \max\{z, \gamma z\}, \quad \gamma \in (0,1)$$

**GELU** (used in GPT, BERT):
$$\sigma(z) = \frac{z}{2}\left[1 + \text{erf}\left(\frac{z}{\sqrt{2}}\right)\right]$$

**SoftPlus:**
$$\sigma(z) = \frac{1}{\beta}\log(1 + \exp(\beta z)), \quad \beta > 0$$

**why not identity $\sigma(z) = z$?**

if we used $\sigma(z) = z$, then:
$$\bar{h}_\theta(x) = W^{[2]} a^{[1]} = W^{[2]} W^{[1]} x = \tilde{W} x$$

it collapses to just linear regression! we need non-linearity to capture complex patterns.

## Connection to kernel methods

remember feature maps from kernel methods? we can view neural networks similarly:

let $\beta$ = all parameters except the last layer. then:
$$a^{[r-1]} = \phi_\beta(x)$$

and:
$$\bar{h}_\theta(x) = W^{[r]} \phi_\beta(x) + b^{[r]}$$

so we're learning BOTH:

- a good feature representation $\phi_\beta(x)$ (the penultimate layer)
- a linear model on top of those features

this is why deep learning needs less domain knowledge - it learns good features automatically!

the penultimate layer $a^{[r-1]}$ is often called the **learned features** or **representations**. these can even transfer to other tasks!