
# Theoretical Statistics — Summary

## 1. Data and Errors

Data can follow many models (for example, linear regression of $y$ on $x$), but real data always contain error. Fisher proposed several desiderata for estimators:

- **Consistency:** estimates converge to the true parameter as $n\to\infty$.
- **Efficiency:** choose estimators that minimize variance (error).
- **Sufficiency:** a statistic should capture all information about the parameter present in the sample.

---

## 2. Measuring Error

### Mean Error (ME)

The Mean Error measures the average signed difference between predicted values $\hat{y}_i$ and actual values $y_i$:

$$
\mathrm{ME} = \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i).
$$

- $n$: number of observations
- $\hat{y}_i$: predicted value for observation $i$
- $y_i$: actual value for observation $i$

Note: ME indicates bias (can be positive or negative).

### Mean Squared Error (MSE)

The Mean Squared Error measures average squared error:

$$
\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2.
$$

MSE penalizes large deviations more than small ones and is a common accuracy metric.

**Comparison:**

| Metric | Formula | Interpretation |
|---|---:|---|
| ME  | $\dfrac{1}{n}\sum(\hat{y}_i - y_i)$ | Average signed error (bias) |
| MSE | $\dfrac{1}{n}\sum(\hat{y}_i - y_i)^2$ | Average squared error (penalizes large deviations) |

---

## 3. Bivariate Normal Distribution

For a bivariate normal vector $(\theta_1,\theta_2)$ with means $\bar{\theta}$, standard deviations $\sigma_1,\sigma_2$ and correlation $r$, the joint density is

$$
f(\theta_1,\theta_2) = \frac{1}{2\pi\sigma_1\sigma_2\sqrt{1-r^2}}\exp\left\{-\frac{1}{2(1-r^2)}\left[\frac{(\theta_1-\bar{\theta})^2}{\sigma_1^2} - \frac{2r(\theta_1-\bar{\theta})(\theta_2-\bar{\theta})}{\sigma_1\sigma_2} + \frac{(\theta_2-\bar{\theta})^2}{\sigma_2^2}\right]\right\}.
$$

The marginal distribution of $\theta_1$ is normal:

$$
f(\theta_1) = \frac{1}{\sigma_1\sqrt{2\pi}}\exp\left(-\frac{(\theta_1-\bar{\theta})^2}{2\sigma_1^2}\right).
$$

The conditional distribution of $\theta_2$ given $\theta_1$ is

$$
\Theta_2\mid\Theta_1=\theta_1 \sim \mathcal{N}\left(\bar{\theta} + r\frac{\sigma_2}{\sigma_1}(\theta_1-\bar{\theta}),\; (1-r^2)\sigma_2^2\right).
$$

Independence occurs when $r=0$ (zero correlation). The factor $r^2$ measures the proportion of variance in one variable explained by the other.

---

## 4. Probability and Maximum Likelihood

For a sample of $n$ Bernoulli trials with $x$ successes and $y$ failures, the likelihood is
$x+y=n$

$$
L(p) = \binom{n}{x} p^x (1-p)^{n-x}.
$$

or

$$
P(\text{sample}) = \frac{n!}{x! y!} p^x (1-p)^y
$$

The maximum likelihood estimator (MLE) for $p$ is

$$
\hat{p} = \frac{x}{n}.
$$

With a flat prior, the posterior (up to normalization) is

$$
\pi(p\mid\text{data}) \propto p^x(1-p)^{n-x}.
$$

### Reparameterization example

Let $\sin\theta = 2p-1$ (so $p=(1+\sin\theta)/2$). Then

$$
L(\theta) = \binom{n}{x}\left(\frac{1+\sin\theta}{2}\right)^x\left(\frac{1-\sin\theta}{2}\right)^{n-x}
$$

and the log-likelihood is

$$
\ell(\theta)=x\log(1+\sin\theta)+(n-x)\log(1-\sin\theta)-n\log 2.
$$

Differentiating gives the usual score equation; 

$$
\frac{x \cos \theta}{1 + \sin \theta} - \frac{y \cos \theta}{1 - \sin \theta} = 0
$$

- Solve:

$$
df = \frac{\cos \theta}{2} d\theta
$$

- **Bayesian view:** Likelihood 
$$ p^x \propto  (1-p)^y $$ maximum at (p = x/n)
- Standard deviation:
solving yields the same MLE $p=x/n$. For large $n$, the sampling standard deviation of $\hat p$ is

$$
\sigma_{\hat p}=\sqrt{\frac{p(1-p)}{n}}.
$$

> Fisher's work on both **discrete** and **continuous** models helped establish modern estimation theory.