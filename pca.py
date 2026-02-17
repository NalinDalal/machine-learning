"""pca.py
Demonstrations of dimension reduction techniques (PCA, IncrementalPCA,
Randomized PCA, KernelPCA, and LLE) with documented examples.

This script is intentionally self-contained and uses the `load_digits`
dataset from scikit-learn for quick demos. It prints summary statistics
and demonstrates transforming and inverse-transforming data where
applicable.

Usage examples:
    python pca.py --demo basic
    python pca.py --demo incremental
    python pca.py --demo randomized
    python pca.py --demo kernel
    python pca.py --demo lle

If matplotlib is available the script will show a small visualization
of original vs reconstructed images for PCA compression.
"""

from __future__ import annotations

import argparse
import sys
import numpy as np

from sklearn.datasets import load_digits

try:
    from sklearn.decomposition import PCA, IncrementalPCA
    from sklearn.manifold import LocallyLinearEmbedding
except Exception as e:  # pragma: no cover - missing sklearn
    print("scikit-learn is required to run this demo: pip install scikit-learn")
    raise


def prepare_data():
    """Load and center the digits dataset.

    Returns
    -------
    X : ndarray, shape (n_samples, n_features)
        Centered feature matrix.
    """
    digits = load_digits()
    X = digits.data.astype(float)
    X_centered = X - X.mean(axis=0)
    return X_centered, digits


def demo_basic_pca(X, digits, n_components=2, show_plot=False):
    """Basic PCA demo using SVD (numpy) and scikit-learn's PCA.

    Demonstrates computing principal components via SVD, projecting
    down to `n_components`, and reconstructing back to original space.
    """
    print("--- Basic PCA demo ---")
    # SVD-based principal components (equivalent to PCA on centered data)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    components = Vt.T[:, :n_components]
    X_proj = X.dot(components)
    print(f"X shape: {X.shape}, projected shape: {X_proj.shape}")

    # scikit-learn PCA for comparison
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print("sklearn PCA explained variance ratio:", pca.explained_variance_ratio_)

    # Reconstruct using inverse transform (approximation)
    X_rec = pca.inverse_transform(X_pca)
    mse = np.mean((X - X_rec) ** 2)
    print(f"Reconstruction MSE (n_components={n_components}): {mse:.6f}")

    if show_plot:
        try:
            import matplotlib.pyplot as plt

            n_show = 6
            fig, axes = plt.subplots(2, n_show, figsize=(8, 3))
            for i in range(n_show):
                axes[0, i].imshow(digits.images[i], cmap="gray")
                axes[0, i].axis("off")
                axes[1, i].imshow(X_rec[i].reshape(digits.images[0].shape), cmap="gray")
                axes[1, i].axis("off")
            axes[0, 0].set_title("Original")
            axes[1, 0].set_title("Reconstructed")
            plt.tight_layout()
            plt.show()
        except Exception:
            print("matplotlib not available; skipping visualization")


def demo_incremental_pca(X, n_components=154, n_batches=10):
    """Demonstrate IncrementalPCA using mini-batches.

    Args:
        X: centered data matrix
        n_components: number of principal components to keep
        n_batches: number of mini-batches to simulate
    """
    print("--- Incremental PCA demo ---")
    inc_pca = IncrementalPCA(n_components=n_components)
    for batch in np.array_split(X, n_batches):
        inc_pca.partial_fit(batch)
    X_reduced = inc_pca.transform(X)
    print(f"Transformed shape: {X_reduced.shape}")


def demo_randomized_pca(X, n_components=50):
    """Demonstrate randomized SVD-based PCA (fast approximation)."""
    print("--- Randomized PCA demo ---")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=42)
    X_reduced = pca.fit_transform(X)
    print(f"Randomized PCA transformed shape: {X_reduced.shape}")
    print("Explained variance ratio (first 10):", pca.explained_variance_ratio_[:10])


def demo_kernel_pca(X, n_components=2, kernel="rbf", gamma=0.04):
    """Kernel PCA demo using scikit-learn's KernelPCA if available."""
    try:
        from sklearn.decomposition import KernelPCA
    except Exception:
        print("KernelPCA not available in this environment.")
        return

    print("--- Kernel PCA demo ---")
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=gamma, fit_inverse_transform=False)
    X_k = kpca.fit_transform(X)
    print(f"KernelPCA transformed shape: {X_k.shape}")


def demo_lle(X, n_components=2, n_neighbors=10):
    """Locally Linear Embedding (LLE) demo."""
    print("--- Locally Linear Embedding (LLE) demo ---")
    lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors)
    X_lle = lle.fit_transform(X)
    print(f"LLE transformed shape: {X_lle.shape}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PCA and manifold learning demos")
    p.add_argument("--demo", choices=["basic", "incremental", "randomized", "kernel", "lle"], default="basic")
    p.add_argument("--show-plot", action="store_true", help="Show reconstruction plot for basic PCA if matplotlib available")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    X, digits = prepare_data()

    if args.demo == "basic":
        demo_basic_pca(X, digits, n_components=2, show_plot=args.show_plot)
    elif args.demo == "incremental":
        demo_incremental_pca(X, n_components=30, n_batches=5)
    elif args.demo == "randomized":
        demo_randomized_pca(X, n_components=30)
    elif args.demo == "kernel":
        demo_kernel_pca(X, n_components=2, kernel="rbf", gamma=0.0433)
    elif args.demo == "lle":
        demo_lle(X, n_components=2, n_neighbors=12)


if __name__ == "__main__":
    main()
