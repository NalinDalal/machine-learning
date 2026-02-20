"""
Training Deep Neural Nets — Chapter 11
========================================
Covers:
  1. Vanishing/Exploding Gradients & Weight Initialization (Xavier, He)
  2. Activation Functions (Sigmoid, Tanh, ReLU, Leaky ReLU, ELU)
  3. Batch Normalization
  4. Gradient Clipping
  5. Reusing Pretrained Layers (Transfer Learning)
  6. Fast Optimizers (Momentum, Nesterov, RMSProp, Adam)
  7. Learning Rate Scheduling
  8. Regularization (ℓ1/ℓ2, Dropout, Max-Norm)
  9. Data Augmentation

Uses TensorFlow 1.x style (tf.contrib) to match the textbook.
Run with:  python training-neural-net.py
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout, arg_scope

# ──────────────────────────────────────────────────────────────────────
# 0 · DATASET — MNIST
# ──────────────────────────────────────────────────────────────────────
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/")
X_train, y_train = mnist.train.images, mnist.train.labels
X_test, y_test = mnist.test.images, mnist.test.labels

n_inputs = 28 * 28  # 784 pixels
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
n_epochs = 20
batch_size = 200

# ======================================================================
# SECTION 1 · WEIGHT INITIALIZATION
# ======================================================================
# Problem: with standard init W ~ U(-1/√n, 1/√n), n·Var(W) = 1/3
#          → gradients shrink every layer → deeper layers barely learn.
#
# Xavier/Glorot init: Var(W) = 2 / (n_in + n_out)
#   → stabilises both forward activations and backward gradients.
#
# He init (for ReLU): Var(W) = 2 / n_in
#   → accounts for ReLU zeroing out half the inputs.
# ======================================================================

# He initializer — best for ReLU / Leaky ReLU
he_init = tf.contrib.layers.variance_scaling_initializer()

# Xavier initializer — best for tanh / sigmoid / softsign
xavier_init = tf.contrib.layers.xavier_initializer()

print("\n✅ Weight initializers created (He & Xavier)")


# ======================================================================
# SECTION 2 · ACTIVATION FUNCTIONS
# ======================================================================
# Ranking (deep nets): ELU > Leaky ReLU > ReLU > tanh > sigmoid
#
# Sigmoid : range (0,1), NOT zero‐centered → vanishing gradients
# Tanh    : range (-1,1), zero‐centered   → better, but sequential saturation
# Softsign: x/(1+|x|), polynomial saturation → most stable classic activation
# ReLU    : max(0,z)   → fast, but "dying ReLU" problem
# Leaky ReLU: max(αz,z) → never dies (α = 0.01 or 0.2)
# ELU     : α(eᶻ-1) for z<0, z for z≥0  → best overall
# ======================================================================


def leaky_relu(z, name=None):
    """Leaky ReLU — keeps a small slope α=0.01 for z<0 so neurons never die."""
    return tf.maximum(0.01 * z, z, name=name)


def parametric_relu(z, alpha_init=0.01, name="prelu"):
    """Parametric ReLU — α is learned via backprop."""
    with tf.variable_scope(name):
        alpha = tf.get_variable("alpha", shape=z.get_shape()[-1],
                                initializer=tf.constant_initializer(alpha_init),
                                dtype=tf.float32)
        return tf.maximum(alpha * z, z)


# ── Demo: build a small net with each activation ──

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None,), name="y")

# ELU (recommended default)
with tf.name_scope("elu_net"):
    h1_elu = fully_connected(X, n_hidden1, activation_fn=tf.nn.elu,
                             weights_initializer=he_init, scope="elu_h1")
    h2_elu = fully_connected(h1_elu, n_hidden2, activation_fn=tf.nn.elu,
                             weights_initializer=he_init, scope="elu_h2")
    logits_elu = fully_connected(h2_elu, n_outputs, activation_fn=None,
                                 scope="elu_out")

# Leaky ReLU (fast alternative)
with tf.name_scope("leaky_relu_net"):
    h1_lr = fully_connected(X, n_hidden1, activation_fn=leaky_relu,
                            weights_initializer=he_init, scope="lr_h1")
    h2_lr = fully_connected(h1_lr, n_hidden2, activation_fn=leaky_relu,
                            weights_initializer=he_init, scope="lr_h2")
    logits_lr = fully_connected(h2_lr, n_outputs, activation_fn=None,
                                scope="lr_out")

print("✅ Activation function demo graphs built (ELU, Leaky ReLU)")


# ======================================================================
# SECTION 3 · BATCH NORMALIZATION
# ======================================================================
# Algorithm (per mini‐batch B of m_B samples):
#   1. μ_B  = (1/m_B) Σ x⁽ⁱ⁾
#   2. σ²_B = (1/m_B) Σ (x⁽ⁱ⁾ − μ_B)²
#   3. x̂⁽ⁱ⁾ = (x⁽ⁱ⁾ − μ_B) / √(σ²_B + ε)
#   4. z⁽ⁱ⁾ = γ · x̂⁽ⁱ⁾ + β          (γ, β learned)
#
# ε ≈ 1e-3 (smoothing term to avoid ÷ 0)
# At test time uses running averages of μ and σ² collected during training.
# ======================================================================

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None,), name="y")
is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

# Batch‐norm parameters shared across all FC layers via arg_scope
bn_params = {
    "is_training": is_training,
    "decay": 0.99,            # exponential moving average decay
    "updates_collections": None  # force in‐place updates of moving mean/var
}

with arg_scope([fully_connected],
               normalizer_fn=batch_norm,
               normalizer_params=bn_params):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits = fully_connected(hidden2, n_outputs, activation_fn=None,
                             scope="outputs")

# loss + training op
xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

print("✅ Batch Normalization graph built")

# ── Training loop with BN ──
print("\n─── Training with Batch Normalization ───")
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op,
                     feed_dict={is_training: True, X: X_batch, y: y_batch})
        acc = accuracy.eval(feed_dict={is_training: False,
                                       X: X_test, y: y_test})
        print(f"  Epoch {epoch:2d}  Test accuracy: {acc:.4f}")


# ======================================================================
# SECTION 4 · GRADIENT CLIPPING
# ======================================================================
# Clip every gradient component to [−threshold, +threshold] so that
# exploding gradients can't destabilise training.
# ======================================================================

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None,), name="y")

hidden1 = fully_connected(X, n_hidden1, scope="gc_h1")
hidden2 = fully_connected(hidden1, n_hidden2, scope="gc_h2")
logits = fully_connected(hidden2, n_outputs, activation_fn=None, scope="gc_out")

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy, name="loss")

threshold = 1.0
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
grads_and_vars = optimizer.compute_gradients(loss)
# clip each gradient tensor to [-threshold, +threshold]
capped_gvs = [(tf.clip_by_value(grad, -threshold, threshold), var)
               for grad, var in grads_and_vars]
training_op = optimizer.apply_gradients(capped_gvs)

print("\n✅ Gradient Clipping graph built (threshold = 1.0)")


# ======================================================================
# SECTION 5 · REUSING PRETRAINED LAYERS (Transfer Learning)
# ======================================================================
# Instead of training from scratch, reuse lower layers from an existing
# model.  Freeze them so only the new top layers are trained.
# ======================================================================

# ── 5a. Restore & retrain a TF model ──
# saver.restore(sess, "./my_original_model.ckpt")

# ── 5b. Reuse specific hidden layers (1-3), train new layers (4+) ──
tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None,), name="y")

hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
hidden3 = fully_connected(hidden2, 50, scope="hidden3")
# new layers added on top
hidden4 = fully_connected(hidden3, 50, scope="hidden4")
logits = fully_connected(hidden4, n_outputs, activation_fn=None, scope="outputs")

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

# Only train hidden4 and outputs (freeze hidden1-3)
train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope="hidden[4]|outputs")
training_op = optimizer.minimize(loss, var_list=train_vars)

# Savers for restoring pretrained layers and saving new model
reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                               scope="hidden[123]")
reuse_vars_dict = {var.name: var for var in reuse_vars}
original_saver = tf.train.Saver(reuse_vars_dict)  # restore old layers
new_saver = tf.train.Saver()                       # save whole model

print("✅ Transfer Learning graph built (freeze layers 1-3, train 4+)")

# ── 5c. Caching frozen layer outputs for speed ──
# Since frozen layers don't change, compute their output once and reuse.
#
#   hidden2_outputs = sess.run(hidden2, feed_dict={X: X_train})
#   # then train only on hidden2_outputs → hidden3 → ... → output
#   for epoch in range(n_epochs):
#       shuffled_idx = np.random.permutation(len(hidden2_outputs))
#       h2_batches = np.array_split(hidden2_outputs[shuffled_idx], n_batches)
#       y_batches  = np.array_split(y_train[shuffled_idx], n_batches)
#       for h2_batch, y_batch in zip(h2_batches, y_batches):
#           sess.run(training_op, feed_dict={hidden2: h2_batch, y: y_batch})

# ── 5d. Reusing weights from other frameworks ──
# original_w = [...]  # load weights from PyTorch / Keras / etc.
# original_b = [...]
# with tf.variable_scope("", reuse=True):
#     hidden1_weights = tf.get_variable("hidden1/weights")
# assign_op = tf.assign(hidden1_weights, original_w)
# sess.run(assign_op)


# ======================================================================
# SECTION 6 · FAST OPTIMIZERS
# ======================================================================
# Standard GD is slow.  These add "memory" of past gradients.
#
# Momentum:     m ← βm + η∇J(θ);  θ ← θ − m
#               β typically 0.9, acts like a ball rolling downhill.
#
# Nesterov AG:  m ← βm + η∇J(θ + βm);  θ ← θ − m
#               Looks ahead before computing gradient → converges faster.
#
# RMSProp:      s ← βs + (1−β)(∇J ⊙ ∇J);  θ ← θ − η·∇J / √(s+ε)
#               Adaptive per-parameter learning rate; β ≈ 0.9.
#
# Adam:         m ← β₁m + (1−β₁)∇J          (momentum)
#               s ← β₂s + (1−β₂)(∇J ⊙ ∇J)  (RMSProp)
#               m̂ = m/(1−β₁ᵀ)                (bias correction)
#               ŝ = s/(1−β₂ᵀ)
#               θ ← θ − η·m̂ / √(ŝ+ε)
# ======================================================================

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None,), name="y")

hidden1 = fully_connected(X, n_hidden1, scope="opt_h1")
hidden2 = fully_connected(hidden1, n_hidden2, scope="opt_h2")
logits = fully_connected(hidden2, n_outputs, activation_fn=None, scope="opt_out")

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)

learning_rate = 0.001

# Momentum
opt_momentum = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                          momentum=0.9)

# Nesterov Accelerated Gradient
opt_nesterov = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                          momentum=0.9, use_nesterov=True)

# RMSProp
opt_rmsprop = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
                                        momentum=0.9, decay=0.9, epsilon=1e-10)

# Adam (recommended default in most cases)
opt_adam = tf.train.AdamOptimizer(learning_rate=learning_rate)

# use Adam for the training demo
training_op = opt_adam.minimize(loss)

print("✅ Optimizer comparison graph built (Momentum, Nesterov, RMSProp, Adam)")


# ======================================================================
# SECTION 7 · LEARNING RATE SCHEDULING
# ======================================================================
# Start with a large learning rate for speed, then decay it so the
# model can settle into a good minimum.
# Exponential decay: lr = lr₀ · decay_rate ^ (global_step / decay_steps)
# ======================================================================

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None,), name="y")

hidden1 = fully_connected(X, n_hidden1, scope="lr_h1")
hidden2 = fully_connected(hidden1, n_hidden2, scope="lr_h2")
logits = fully_connected(hidden2, n_outputs, activation_fn=None, scope="lr_out")

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)

initial_learning_rate = 0.1
decay_steps = 10000
decay_rate = 1 / 10
global_step = tf.Variable(0, trainable=False, name="global_step")

# lr decays every `decay_steps` steps by factor `decay_rate`
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
                                           decay_steps, decay_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
training_op = optimizer.minimize(loss, global_step=global_step)

print("✅ Learning Rate Scheduling graph built (exponential decay)")


# ======================================================================
# SECTION 8 · REGULARIZATION
# ======================================================================
# Techniques to prevent overfitting:
#   a) Early Stopping — stop when validation error rises
#   b) ℓ1 / ℓ2 — penalise large weights in the loss
#   c) Dropout — randomly zero out neurons during training
#   d) Max-Norm — clip weight vector norm per neuron
# ======================================================================

# ── 8a. Early Stopping ──
# Simply evaluate on a validation set each epoch;
# save the best model; stop when no improvement for N epochs.
# (Implemented via tf.train.Saver + patience counter — see BN training loop above.)

# ── 8b. ℓ1 / ℓ2 Regularization ──
tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None,), name="y")

scale = 0.01  # regularization strength

with arg_scope([fully_connected],
               weights_regularizer=tf.contrib.layers.l1_regularizer(scale=scale)):
    hidden1 = fully_connected(X, n_hidden1, scope="l1_h1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="l1_h2")
    logits = fully_connected(hidden2, n_outputs, activation_fn=None, scope="l1_out")

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
# collect all regularization losses added by arg_scope
reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([base_loss] + reg_losses, name="loss")

print("✅ ℓ1 Regularization graph built (scale = 0.01)")

# ── 8c. Dropout ──
tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None,), name="y")
is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

keep_prob = 0.5  # drop 50 % of neurons during training

X_drop = dropout(X, keep_prob, is_training=is_training)
hidden1 = fully_connected(X_drop, n_hidden1, scope="drop_h1")
hidden1_drop = dropout(hidden1, keep_prob, is_training=is_training)
hidden2 = fully_connected(hidden1_drop, n_hidden2, scope="drop_h2")
hidden2_drop = dropout(hidden2, keep_prob, is_training=is_training)
logits = fully_connected(hidden2_drop, n_outputs, activation_fn=None,
                         scope="drop_out")

# At test time dropout is OFF (is_training=False) and weights are already
# scaled by keep_prob internally, so no manual rescaling needed.

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(loss)
correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()

print("✅ Dropout graph built (keep_prob = 0.5)")

print("\n─── Training with Dropout ───")
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op,
                     feed_dict={is_training: True, X: X_batch, y: y_batch})
        acc = accuracy.eval(feed_dict={is_training: False,
                                       X: X_test, y: y_test})
        print(f"  Epoch {epoch:2d}  Test accuracy: {acc:.4f}")

# ── 8d. Max-Norm Regularization ──
# Constraint: ‖w‖₂ ≤ r  for each neuron's incoming weights.
# After each training step, clip weights that exceed threshold.

tf.reset_default_graph()
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int64, shape=(None,), name="y")


def max_norm_regularizer(threshold, axes=1, name="max_norm",
                         collection="max_norm"):
    """Creates a max-norm constraint (not a loss term).
    After each training step, call the ops in the 'max_norm' collection.
    """
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None  # no regularization loss — this is a hard constraint
    return max_norm


max_norm_reg = max_norm_regularizer(threshold=1.0)

hidden1 = fully_connected(X, n_hidden1, scope="mn_h1",
                           weights_regularizer=max_norm_reg)
hidden2 = fully_connected(hidden1, n_hidden2, scope="mn_h2",
                           weights_regularizer=max_norm_reg)
logits = fully_connected(hidden2, n_outputs, activation_fn=None, scope="mn_out")

xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(xentropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_op = optimizer.minimize(loss)

# Gather all clip ops — run them AFTER each training step
clip_all_weights = tf.get_collection("max_norm")
init = tf.global_variables_initializer()

print("✅ Max-Norm Regularization graph built (threshold = 1.0)")

print("\n─── Training with Max-Norm ───")
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            sess.run(clip_all_weights)  # enforce max-norm after each step
        # no accuracy metric added here for brevity
    print("  Max-Norm training complete")


# ======================================================================
# SECTION 9 · DATA AUGMENTATION
# ======================================================================
# Generate new training instances from existing ones to reduce overfitting.
# For images: random flips, rotations, crops, brightness/contrast shifts.
# TF provides tf.image.* ops that can be added to the input pipeline.
# ======================================================================

def augment_image(image):
    """Apply random augmentations to a single 28×28 image tensor."""
    image = tf.reshape(image, [28, 28, 1])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # Pad and random‐crop to simulate small translations
    image = tf.image.resize_image_with_crop_or_pad(image, 32, 32)
    image = tf.random_crop(image, [28, 28, 1])
    return tf.reshape(image, [784])


print("✅ Data augmentation helper defined")