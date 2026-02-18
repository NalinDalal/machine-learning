"""
tf-gra-des.py
=========================
Linear Regression using Gradient Descent in TensorFlow (TF1-style).

This script demonstrates 3 ways to implement Gradient Descent:

1) Manual gradients (explicit formula)
2) AutoDiff gradients (tf.gradients)
3) Using Optimizers (tf.train.GradientDescentOptimizer / MomentumOptimizer)

It also shows:
- Feeding data using placeholders
- Mini-batch training
- Saving & restoring models
- TensorBoard logging

Author: (your name)
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1) LOAD + PREPARE DATA
# ============================================================

housing = fetch_california_housing()

m, n = housing.data.shape  # m = number of instances, n = number of features

# Scale features (important for Gradient Descent)
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing.data)

# Add bias term x0 = 1 for every instance
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

# Targets must be shaped (m, 1)
y_data = housing.target.reshape(-1, 1)


# ============================================================
# 2) HYPERPARAMETERS
# ============================================================

n_epochs = 1000
learning_rate = 0.01

# Mini-batch toggle
USE_MINIBATCH = False
batch_size = 50

# Choose training method:
# "manual" | "autodiff" | "optimizer" | "momentum"
TRAINING_MODE = "manual"

# TensorBoard logging toggle
USE_TENSORBOARD = True

# Save model toggle
SAVE_MODEL = True
MODEL_PATH = "/tmp/my_model_final.ckpt"


# ============================================================
# 3) HELPER: MINI-BATCH FETCHER
# ============================================================

def fetch_batch(epoch, batch_index, batch_size):
    """
    Returns one mini-batch of data.

    We use a deterministic random seed so results are reproducible.
    """
    np.random.seed(epoch * 1000 + batch_index)
    indices = np.random.randint(m, size=batch_size)
    X_batch = scaled_housing_data_plus_bias[indices]
    y_batch = y_data[indices]
    return X_batch, y_batch


# ============================================================
# 4) BUILD COMPUTATION GRAPH
# ============================================================

# NOTE:
# - If using placeholders, we can feed different data each run (mini-batch).
# - If not, we can use constants for full-batch training.

if USE_MINIBATCH:
    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
else:
    X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(y_data, dtype=tf.float32, name="y")

# Model parameters theta: shape = (n+1, 1)
# Initialized randomly in range [-1, 1]
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")

# Predictions: y_pred = X @ theta
y_pred = tf.matmul(X, theta, name="predictions")


# ------------------------------------------------------------
# LOSS FUNCTION (MSE)
# ------------------------------------------------------------
with tf.name_scope("loss"):
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")


# ============================================================
# 5) TRAINING OP (3 METHODS)
# ============================================================

if TRAINING_MODE == "manual":
    # Manual gradient:
    # gradients = 2/m * X^T (X theta - y)
    # But for mini-batch, use current batch size dynamically.

    with tf.name_scope("manual_gradient_descent"):
        if USE_MINIBATCH:
            batch_m = tf.cast(tf.shape(X)[0], tf.float32)  # dynamic batch size
            gradients = 2 / batch_m * tf.matmul(tf.transpose(X), error)
        else:
            gradients = 2 / m * tf.matmul(tf.transpose(X), error)

        training_op = tf.assign(theta, theta - learning_rate * gradients)

elif TRAINING_MODE == "autodiff":
    # AutoDiff: let TF compute gradient of mse w.r.t theta
    with tf.name_scope("autodiff_gradient_descent"):
        gradients = tf.gradients(mse, [theta])[0]
        training_op = tf.assign(theta, theta - learning_rate * gradients)

elif TRAINING_MODE == "optimizer":
    # Using built-in optimizer
    with tf.name_scope("optimizer_gradient_descent"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(mse)

elif TRAINING_MODE == "momentum":
    # Momentum optimizer
    with tf.name_scope("optimizer_momentum"):
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=0.9
        )
        training_op = optimizer.minimize(mse)

else:
    raise ValueError("Invalid TRAINING_MODE. Choose: manual | autodiff | optimizer | momentum")


# ============================================================
# 6) TENSORBOARD SUMMARIES
# ============================================================

if USE_TENSORBOARD:
    mse_summary = tf.summary.scalar("MSE", mse)
    merged_summaries = tf.summary.merge_all()
    file_writer = tf.summary.FileWriter("tf_logs", tf.get_default_graph())
else:
    file_writer = None


# ============================================================
# 7) SAVE / RESTORE
# ============================================================

if SAVE_MODEL:
    saver = tf.train.Saver()
else:
    saver = None


# ============================================================
# 8) TRAIN
# ============================================================

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    if USE_MINIBATCH:
        n_batches = int(np.ceil(m / batch_size))
    else:
        n_batches = 1

    for epoch in range(n_epochs):

        for batch_index in range(n_batches):

            # Prepare feed dict if placeholders are used
            if USE_MINIBATCH:
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                feed_dict = {X: X_batch, y: y_batch}
            else:
                feed_dict = None

            # Print MSE occasionally
            if epoch % 100 == 0 and batch_index == 0:
                if feed_dict:
                    mse_val = sess.run(mse, feed_dict=feed_dict)
                else:
                    mse_val = sess.run(mse)
                print("Epoch:", epoch, "MSE:", mse_val)

            # TensorBoard logging
            if USE_TENSORBOARD and batch_index % 10 == 0:
                if feed_dict:
                    summary_str = sess.run(merged_summaries, feed_dict=feed_dict)
                else:
                    summary_str = sess.run(merged_summaries)

                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)

            # Run training step
            if feed_dict:
                sess.run(training_op, feed_dict=feed_dict)
            else:
                sess.run(training_op)

        # Save checkpoint occasionally
        if SAVE_MODEL and epoch % 100 == 0:
            saver.save(sess, "/tmp/my_model.ckpt")

    # Final theta
    best_theta = theta.eval()

    print("\nTraining complete.")
    print("Best theta:\n", best_theta)

    # Save final model
    if SAVE_MODEL:
        save_path = saver.save(sess, MODEL_PATH)
        print("\nModel saved to:", save_path)

# Close TensorBoard writer
if file_writer:
    file_writer.close()


# ============================================================
# 9) RESTORE MODEL (EXAMPLE)
# ============================================================

if SAVE_MODEL:
    tf.reset_default_graph()

    # Rebuild theta variable for restoring
    theta_restored = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
    saver_restore = tf.train.Saver()

    with tf.Session() as sess:
        saver_restore.restore(sess, MODEL_PATH)
        restored_theta = theta_restored.eval()
        print("\nRestored theta:\n", restored_theta)


"""
Run TensorBoard:
----------------
source env/bin/activate
tensorboard --logdir tf_logs/

Open:
http://0.0.0.0:6006/
"""
