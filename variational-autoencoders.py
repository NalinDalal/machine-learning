# Variational Autoencoder (VAE) for MNIST
# Probabilistic generative autoencoder that learns a latent Gaussian distribution.
# Encoder outputs mean μ and log-variance γ; codings are sampled via the
# reparameterization trick: z = μ + σ·ε, where σ = exp(γ/2), ε ~ N(0,1).

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

# ─── Load Data ───────────────────────────────────────────────
mnist = input_data.read_data_sets("/tmp/data/")

# ─── Hyperparameters ─────────────────────────────────────────
n_inputs = 28 * 28          # MNIST images
n_hidden1 = 500             # encoder layer 1
n_hidden2 = 500             # encoder layer 2
n_hidden3 = 20              # coding layer (latent space)
n_hidden4 = n_hidden2       # decoder layer 1
n_hidden5 = n_hidden1       # decoder layer 2
n_outputs = n_inputs

learning_rate = 0.001
n_epochs = 50
batch_size = 150

# ─── Build the VAE ──────────────────────────────────────────
with tf.contrib.framework.arg_scope(
        [fully_connected],
        activation_fn=tf.nn.elu,
        weights_initializer=tf.contrib.layers.variance_scaling_initializer()):

    X = tf.placeholder(tf.float32, [None, n_inputs])

    # Encoder
    hidden1 = fully_connected(X, n_hidden1)
    hidden2 = fully_connected(hidden1, n_hidden2)

    # Latent distribution parameters (no activation — linear outputs)
    hidden3_mean = fully_connected(hidden2, n_hidden3, activation_fn=None)   # μ
    hidden3_gamma = fully_connected(hidden2, n_hidden3, activation_fn=None)  # log(σ²)

    # Reparameterization trick
    hidden3_sigma = tf.exp(0.5 * hidden3_gamma)                             # σ
    noise = tf.random_normal(tf.shape(hidden3_sigma), dtype=tf.float32)      # ε ~ N(0,1)
    hidden3 = hidden3_mean + hidden3_sigma * noise                           # z = μ + σ·ε

    # Decoder
    hidden4 = fully_connected(hidden3, n_hidden4)
    hidden5 = fully_connected(hidden4, n_hidden5)
    logits = fully_connected(hidden5, n_outputs, activation_fn=None)
    outputs = tf.sigmoid(logits)

# ─── Loss Function ──────────────────────────────────────────
# Reconstruction loss (binary cross-entropy, summed over pixels)
reconstruction_loss = tf.reduce_sum(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits))

# Latent loss (KL divergence between learned latent distribution and N(0,1))
# KL(q(z|x) || p(z)) = -0.5 * Σ(1 + γ - μ² - exp(γ))
latent_loss = 0.5 * tf.reduce_sum(
    tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1 - hidden3_gamma)

cost = reconstruction_loss + latent_loss

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()

# ─── Training ───────────────────────────────────────────────
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        epoch_cost = 0.0
        for iteration in range(n_batches):
            X_batch, _ = mnist.train.next_batch(batch_size)
            _, batch_cost = sess.run([training_op, cost], feed_dict={X: X_batch})
            epoch_cost += batch_cost
        print(f"Epoch {epoch + 1}/{n_epochs}  Cost: {epoch_cost / n_batches:.2f}")

    # ─── Generate New Digits ─────────────────────────────────
    n_digits = 60
    codings_rnd = np.random.normal(size=[n_digits, n_hidden3])
    outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})

    def plot_image(image, shape=[28, 28]):
        plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
        plt.axis("off")

    n_rows, n_cols = 6, 10
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 6))
    for idx in range(n_digits):
        plt.subplot(n_rows, n_cols, idx + 1)
        plot_image(outputs_val[idx])
    plt.suptitle("Generated Digits (sampled from latent space)")
    plt.tight_layout()
    plt.savefig("vae_generated_digits.png", dpi=150)
    plt.show()

    # ─── Visualise Reconstructions ───────────────────────────
    n_test = 5
    X_test = mnist.test.images[:n_test]
    reconstructed = outputs.eval(feed_dict={X: X_test})

    fig, axes = plt.subplots(n_test, 2, figsize=(4, 10))
    for i in range(n_test):
        plt.subplot(n_test, 2, i * 2 + 1)
        plot_image(X_test[i])
        plt.title("Original")
        plt.subplot(n_test, 2, i * 2 + 2)
        plot_image(reconstructed[i])
        plt.title("Reconstructed")
    plt.tight_layout()
    plt.savefig("vae_reconstructions.png", dpi=150)
    plt.show()

    # ─── 2-D Latent Space Visualisation (optional, works best with n_hidden3=2) ─
    # If the latent dimension is 2 you can plot the manifold directly.
    if n_hidden3 == 2:
        X_test_full = mnist.test.images
        y_test_full = mnist.test.labels
        codings_val = hidden3_mean.eval(feed_dict={X: X_test_full})

        plt.figure(figsize=(8, 6))
        plt.scatter(codings_val[:, 0], codings_val[:, 1],
                    c=y_test_full, cmap="tab10", s=2, alpha=0.7)
        plt.colorbar()
        plt.title("2-D Latent Space (coloured by digit label)")
        plt.xlabel("z₁")
        plt.ylabel("z₂")
        plt.tight_layout()
        plt.savefig("vae_latent_space.png", dpi=150)
        plt.show()
