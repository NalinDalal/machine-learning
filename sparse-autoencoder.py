# Sparse Autoencoder for MNIST
# Uses KL divergence to enforce sparsity in the coding layer,
# pushing the autoencoder to have only a small fraction of active neurons.

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

# ─── Load Data ───────────────────────────────────────────────
mnist = input_data.read_data_sets("/tmp/data/")

# ─── Hyperparameters ─────────────────────────────────────────
n_inputs = 28 * 28          # MNIST images
n_hidden1 = 300             # encoder layer
n_hidden2 = 150             # coding layer (sparsity applied here)
n_hidden3 = n_hidden1       # decoder layer
n_outputs = n_inputs

learning_rate = 0.01
sparsity_target = 0.1       # target proportion of active neurons
sparsity_weight = 0.2       # weight of the sparsity penalty

n_epochs = 20
batch_size = 150


# ─── KL Divergence for Sparsity ─────────────────────────────
def kl_divergence(p, q):
    """Kullback-Leibler divergence between target sparsity p and actual mean activation q."""
    return p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q))


# ─── Build the Autoencoder ──────────────────────────────────
X = tf.placeholder(tf.float32, shape=[None, n_inputs])

# Encoder
weights1 = tf.Variable(
    tf.contrib.layers.variance_scaling_initializer()([n_inputs, n_hidden1]),
    dtype=tf.float32, name="weights1")
biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
hidden1 = tf.nn.elu(tf.matmul(X, weights1) + biases1)

weights2 = tf.Variable(
    tf.contrib.layers.variance_scaling_initializer()([n_hidden1, n_hidden2]),
    dtype=tf.float32, name="weights2")
biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
hidden2 = tf.nn.elu(tf.matmul(hidden1, weights2) + biases2)  # coding layer

# Decoder
weights3 = tf.Variable(
    tf.contrib.layers.variance_scaling_initializer()([n_hidden2, n_hidden3]),
    dtype=tf.float32, name="weights3")
biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
hidden3 = tf.nn.elu(tf.matmul(hidden2, weights3) + biases3)

weights4 = tf.Variable(
    tf.contrib.layers.variance_scaling_initializer()([n_hidden3, n_outputs]),
    dtype=tf.float32, name="weights4")
biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")
logits = tf.matmul(hidden3, weights4) + biases4
outputs = tf.nn.sigmoid(logits)


# ─── Loss Function ──────────────────────────────────────────
# Reconstruction loss (cross-entropy works well for MNIST pixel values in [0,1])
reconstruction_loss = tf.reduce_sum(
    tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits))

# Sparsity loss via KL divergence on the coding layer
hidden2_mean = tf.reduce_mean(hidden2, axis=0)   # mean activation per neuron
sparsity_loss = tf.reduce_sum(kl_divergence(sparsity_target, hidden2_mean))

# Total loss
loss = reconstruction_loss + sparsity_weight * sparsity_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()


# ─── Training ───────────────────────────────────────────────
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        epoch_loss = 0.0
        for iteration in range(n_batches):
            X_batch, _ = mnist.train.next_batch(batch_size)
            _, batch_loss = sess.run([training_op, loss], feed_dict={X: X_batch})
            epoch_loss += batch_loss
        print(f"Epoch {epoch + 1}/{n_epochs}  Loss: {epoch_loss / n_batches:.2f}")

    # ─── Visualise Reconstructions ───────────────────────────
    n_test_digits = 5
    X_test = mnist.test.images[:n_test_digits]
    outputs_val = outputs.eval(feed_dict={X: X_test})

    def plot_image(image, shape=[28, 28]):
        plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
        plt.axis("off")

    fig, axes = plt.subplots(n_test_digits, 2, figsize=(4, 10))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.title("Original")
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])
        plt.title("Reconstructed")
    plt.tight_layout()
    plt.savefig("sparse_autoencoder_reconstructions.png", dpi=150)
    plt.show()

    # ─── Visualise Learned Features (first hidden layer) ────
    weights1_val = weights1.eval()
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plot_image(weights1_val.T[i])
    plt.suptitle("First Hidden Layer Features")
    plt.tight_layout()
    plt.savefig("sparse_autoencoder_features.png", dpi=150)
    plt.show()

    # ─── Check Sparsity ────────────────────────────────────
    hidden2_val = hidden2.eval(feed_dict={X: mnist.test.images[:1000]})
    mean_activation = np.mean(hidden2_val, axis=0)
    print(f"\nCoding layer mean activation: {np.mean(mean_activation):.4f} "
          f"(target: {sparsity_target})")
    print(f"Neurons with activation < {sparsity_target}: "
          f"{np.sum(mean_activation < sparsity_target)} / {n_hidden2}")
