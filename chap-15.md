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

437-454
