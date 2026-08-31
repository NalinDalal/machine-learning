# Unsupervised Learning

No labels, just data. The goal is to find patterns, structure, or relationships in the data without being told what to look for. Think of it as letting the model explore and summarize the data on its own.

## The Core Idea

Given data $\{x^{(i)}\}_{i=1}^n$ (no $y$ labels), we want to learn a model that captures something useful about the data distribution $p(x)$. Common tasks:

- **Clustering**: Group similar data points (e.g., customer segmentation).
- **Dimensionality Reduction**: Compress data while preserving structure (e.g., visualizing high-dimensional data).
- **Density Estimation**: Model the probability distribution of the data (e.g., anomaly detection).

## Key Methods

### 1. Clustering (e.g., K-Means)

**Goal**: Partition data into $k$ groups where points in same cluster/group are similar.

**How K-Means Works**:
Algorithm (K-Means)

1. Initialize $k$ cluster centers (randomly or heuristically).
2. Repeat:
  - Assign each point $x^{(i)}$ to the nearest cluster center (Euclidean distance).
  - Update each cluster center $\mu_j$ to the mean of assigned points.
- Stop when assignments stabilize or after fixed iterations.

**Objective**:
Minimize within-cluster variance:

$$
J(\mu_1, \dots, \mu_k) = \sum_{i=1}^n \min_{j} \| x^{(i)} - \mu_j \|_2^2
$$

where $\mu_j$ is the $j$-th cluster center.

**Pros**:

- Simple, fast, works well for globular clusters.

**Cons**:

- Assumes $k$ is known (use elbow method or silhouette score to guess).
- Sensitive to initialization (run multiple times, pick best).
- Struggles with non-spherical clusters.

**Alternatives**:

- **Hierarchical Clustering**: Builds a tree of clusters (dendrogram).
- **DBSCAN**: Groups based on density, handles outliers.
- **Gaussian Mixture Models (GMM)**: Probabilistic clustering, assumes data comes from a mix of Gaussians.


## 2. Dimensionality Reduction (PCA)

Goal: map $x\in\mathbb{R}^d$ to low-dimensional $z\in\mathbb{R}^k$ ($k\ll d$) while retaining useful information.

PCA (linear):

1. Center data.
2. Compute covariance $\Sigma=\tfrac{1}{n}\sum_i x^{(i)}(x^{(i)})^T$.
3. Take top-$k$ eigenvectors $U\in\mathbb{R}^{d\times k}$ and set $z^{(i)}=U^T x^{(i)}$.

PCA minimizes reconstruction error:
$$
\min_U\sum_{i=1}^n\|x^{(i)}-UU^T x^{(i)}\|_2^2,
$$
subject to $U^T U=I$.

Pros: efficient, interpretable; 
Cons: linearity limitation. 

Nonlinear alternatives: t-SNE, UMAP (visualization), and autoencoders (neural).

## 3. Density Estimation (GMM example)

Goal: estimate $p(x)$.

Gaussian Mixture Model (GMM): assume
$$
p(x)=\sum_{j=1}^k\pi_j\mathcal{N}(x\mid\mu_j,\Sigma_j),
$$
fit via Expectation-Maximization (EM): E-step computes responsibilities, M-step updates parameters to maximize likelihood.

Use cases: anomaly detection (low $p(x)$), clustering, generative modeling. Alternatives: kernel density estimation, variational autoencoders (VAEs), normalizing flows.

## Practical tips

- Preprocess: standardize features (zero mean, unit variance) before PCA or K-Means.
- Choose $k$: elbow plot, silhouette score, or domain knowledge.
- Evaluate without labels: use intrinsic metrics (silhouette, Davies–Bouldin) or evaluate on downstream tasks.
- Scale to large data: mini-batch K-Means, randomized SVD for PCA.

## Why it matters

- Exploratory data analysis: find structure and anomalies.
- Feature learning: create compact, informative representations for supervised models.
- Pretraining and generative modeling: unsupervised methods power modern representation learning.
