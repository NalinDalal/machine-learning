# Generalized Linear Models (GLMs)

This note summarizes how linear regression, logistic regression, and their cousins fit into the unified framework of Generalized Linear Models (GLMs).

## Setup — quick recap

- Regression: $y\mid x;\theta\sim\mathcal{N}(\mu,\sigma^2)$.
- Binary classification: $y\mid x;\theta\sim\text{Bernoulli}(\phi)$.

In both cases the mean/parameter ($\mu$ or $\phi$) is a function of $x$ and $\theta$.

## Exponential family

A distribution belongs to the exponential family if it can be written as
$$
p(y;\eta)=b(y)\exp\big(\eta^T T(y)-a(\eta)\big),
$$
where $\eta$ is the natural (canonical) parameter, $T(y)$ is the sufficient statistic, $a(\eta)$ is the log-partition function, and $b(y)$ is the base measure.

Examples:

- Bernoulli: $p(y;\phi)=\phi^y(1-\phi)^{1-y}=\exp\big(\eta y-a(\eta)\big)$ with $\eta=\log(\phi/(1-\phi))$, $a(\eta)=\log(1+e^{\eta})$ and $\phi=\sigma(\eta)=1/(1+e^{-\eta})$.
- Gaussian ($\sigma^2=1$): $p(y;\mu)=\frac{1}{\sqrt{2\pi}}e^{-y^2/2}\exp(\mu y-\mu^2/2)$ with $\eta=\mu$, $a(\eta)=\eta^2/2$, $T(y)=y$.

Many other distributions belong to the family (multinomial, Poisson, Gamma, etc.).

## The GLM recipe (three assumptions)

1. The conditional distribution $y\mid x$ is from the exponential family.
2. We model the conditional mean: $h(x)=\mathbb{E}[y\mid x]$.
3. The natural parameter is linear in the inputs: $\eta=\theta^T x$ (or $\eta_i=\theta_i^T x$ for vector-valued $\eta$).

These assumptions give a simple mapping from inputs $x$ to the expected response via the canonical response function $g(\eta)=\mathbb{E}[y\mid\eta]$.

## Recovering familiar models

Linear regression (OLS): assume Gaussian noise. The canonical link is identity, so
$$
h_\theta(x)=\mathbb{E}[y\mid x]=\eta=\theta^T x.
$$

Logistic regression: assume Bernoulli outcomes. The canonical response is the sigmoid,
$$
h_\theta(x)=\mathbb{E}[y\mid x]=\sigma(\eta)=\frac{1}{1+e^{-\eta}}=\frac{1}{1+e^{-\theta^T x}}.
$$

Softmax / multinomial (multiclass): for $k$ classes with parameters $\theta_1,\dots,\theta_k$,
$$
p(y=i\mid x)=\frac{e^{\theta_i^T x}}{\sum_{j=1}^k e^{\theta_j^T x}},
$$
which generalizes the sigmoid to multiple classes.

Loss (negative log-likelihood) for softmax:
$$
\ell(\theta)=-\sum_{i=1}^n\log\frac{e^{\theta_{y^{(i)}}^T x^{(i)}}}{\sum_{j=1}^k e^{\theta_j^T x^{(i)}}}.
$$
Gradient for class parameter $\theta_i$:
$$
\frac{\partial\ell}{\partial\theta_i}=\sum_{t=1}^n\big(\mathbb{1}\{y^{(t)}=i\}-\phi_i^{(t)}\big)x^{(t)},
$$
where $\phi_i^{(t)}=\frac{e^{\theta_i^T x^{(t)}}}{\sum_j e^{\theta_j^T x^{(t)}}}$.

## Unified view — quick table

| Problem | Distribution | Response $g(\eta)$ | Algorithm |
|---|---|---|---|
| Continuous prediction | Gaussian | $g(\eta)=\eta$ | Linear regression |
| Binary classification | Bernoulli | $g(\eta)=\sigma(\eta)$ | Logistic regression |
| Multi-class | Multinomial | $g(\eta)=\text{softmax}(\eta)$ | Softmax regression |
| Count data | Poisson | $g(\eta)=e^{\eta}$ | Poisson regression |
| Prediction / Loss |


## Key terminology

- Canonical response function $g(\eta)$: maps natural parameter to expected value.
- Canonical link function: $g^{-1}$, maps mean back to the natural parameter.

For logistic regression the canonical response is the sigmoid and the link is the logit.

## Why this matters

Choices like the sigmoid or linear predictor are not arbitrary: they follow from:

<ol type="I">
<li>selecting a distribution suitable for the data type</li>
<li>using the exponential family structure</li>
<li>modeling the natural parameter as a linear function of inputs</li>
</ol>

This gives a systematic recipe for building models and explains the similarity of update rules across different problems.
