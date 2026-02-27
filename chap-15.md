# Autoencoders

1. **They learn without labels (unsupervised learning).**
   Autoencoders train by trying to copy input → output, so they don’t need labeled data.

2. **They compress data into a smaller representation (coding).**
   The network squeezes input into a lower-dimensional form, which helps in **dimensionality reduction** and feature extraction.

3. **Constraints make them useful.**
   By limiting the internal size or adding noise, the model can’t just copy data directly — it must learn meaningful patterns, and it can even generate new similar data (generative model).

# Efficient Data Representations

• 40, 27, 25, 36, 81, 57, 10, 73, 19, 68
• 50, 25, 76, 38, 19, 58, 29, 88, 44, 22, 11, 34, 17, 52, 26, 13, 40, 20

1. At first glance, the shorter number sequence seems easier to memorize.

2. The longer sequence is actually easier if you notice its pattern (hailstone sequence rule: even → half, odd → 3×n + 1).

3. Recognizing patterns makes memorization easier.

4. Humans rely on pattern recognition because memory is limited.

5. Chess experts remember board positions easily only when the positions follow real game patterns.

6. Experts don’t have better memory — they recognize patterns better.

7. Autoencoders work similarly: they learn patterns to store data efficiently.

8. An autoencoder has two parts:
   - **Encoder** → compresses input into a smaller internal representation.
   - **Decoder** → reconstructs the original input from that representation.

9. The output layer has the same number of neurons as the input layer because it tries to copy the input.

10. In an **undercomplete autoencoder** (smaller hidden layer), the model cannot simply copy inputs — it must learn the most important features and ignore unnecessary details.

# [Performing PCA with an Undercomplete Linear Autoencoder](./pca-linear-autoencoder.py)

If the autoencoder uses only linear activations and the cost function is the Mean
Squared Error (MSE), then it ends up performing Principal Component Analysis

```python
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

n_inputs = 3 # 3D inputs
n_hidden = 2 # 2D codings

n_outputs = n_inputs

learning_rate = 0.01

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = fully_connected(X, n_hidden, activation_fn=None)
outputs = fully_connected(hidden, n_outputs, activation_fn=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # MSE

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(reconstruction_loss)

init = tf.global_variables_initializer()
```

things to note:

- The number of outputs is equal to the number of inputs.
- To perform simple PCA, we set activation_fn=None (i.e., all neurons are linear) and the cost function is the MSE.

```python
#applying it
X_train, X_test = [...] # load the dataset

n_iterations = 1000
codings = hidden # the output of the hidden layer provides the codings

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        training_op.run(feed_dict={X: X_train}) # no labels (unsupervised)
    codings_val = codings.eval(feed_dict={X: X_test})
```

# Stacked Autoencoder

autoencoders can have multiple hidden layers. In this case they are called stacked autoencoders (or deep autoencoders).

architecture: typically symmetrical with regards to the central hidden layer (the coding layer).

```mermaid
flowchart TB
    A[Inputs\n784 units] --> B[Hidden 1\n300 units]
    B --> C[Hidden 2 (Codings)\n150 units]
    C --> D[Hidden 3\n300 units]
    D --> E[Outputs\n784 units\n(Reconstructions ≈ Inputs)]
```

**Implementation**

```python
# stacked autoencoder for MNIST,
# using He initialization, the ELU activation function, and ℓ2 regularization.
n_inputs = 28 * 28 # for MNIST
n_hidden1 = 300
n_hidden2 = 150 # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.001

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
with tf.contrib.framework.arg_scope(
[fully_connected],
activation_fn=tf.nn.elu,
weights_initializer=tf.contrib.layers.variance_scaling_initializer(),
weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg)):
hidden1 = fully_connected(X, n_hidden1)
hidden2 = fully_connected(hidden1, n_hidden2) # codings
hidden3 = fully_connected(hidden2, n_hidden3)
outputs = fully_connected(hidden3, n_outputs, activation_fn=None)

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X)) # MSE


reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([reconstruction_loss] + reg_losses)
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()



#trainig the model
n_epochs = 5
batch_size = 150
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_batches = mnist.train.num_examples // batch_size
        for iteration in range(n_batches):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch})
```

**Tying Weights:**

437-453
