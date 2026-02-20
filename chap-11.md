# Training Deep Neural Nets

3 problems:
- vanishing gradients problem{makes low layers of network hard to train}
- slow training due to large network
- chance of overfitting

## Vanishing/Exploding Gradients Problems
as backpropogation iterates, gradients gets smaller and smaller
```python
he_init=tf.contrib.layers.variance_scaling_initializer()
hidden1=fully_connected(X,n_hidden1,weights_initializer=he_init,scope='h1')
```
hence lower layer conn wt lefts out, no good solution: vanishing gradients problem
or gradients can increase: exploding gradients problem

optimized feedforward neural networks with one to five hidden layers, with one thousand hidden units per layer, and with a softmax logistic regression for the output layer

cost: -logP(y|x) where (x, y) is the (input image, target class) pair. 
batch size 10

### Activation Functions & Their Effect

**Sigmoid** — range (0,1), not zero-centered
- top layer saturates to 0 quickly → derivative ≈ 0 → vanishing gradient
- lower layers stop learning, training gets stuck
- sigmoid + random init = dangerous in deep nets
- unsupervised pretraining helps by giving meaningful initial weights

**Tanh** — range (-1,1), zero-centered, symmetric
- better than sigmoid since near-zero region has good gradients
- but with standard init: layers saturate sequentially (layer 1 → 2 → 3...)
- still struggles in deep nets

**Softsign** — $f(x) = \frac{x}{1 + |x|}$
- polynomial (slower) saturation vs tanh's exponential
- layers saturate together, activations stay near "knees" (nonlinear but not flat)
- gradients still flow → more stable than tanh

> Training deep networks is mostly about
> controlling signal and gradient variance across layers.

### Cost Function

| Loss | Behavior |
|------|----------|
| Quadratic | flat plateaus, harder optimization |
| Cross-entropy (log-likelihood + softmax) | better gradients, faster convergence |

→ always use cross-entropy for classification

### Gradient Propagation at Init

gradient variance ∝ product of (layer_size × weight_variance)
- $n \cdot Var(W) < 1$ → gradients shrink (vanishing)
- $n \cdot Var(W) > 1$ → gradients explode

### Xavier/Glorot Initialization

we want $Var[z_i]$ and $Var[\text{gradient}]$ constant across layers

two constraints: $n_i \cdot Var(W_i) = 1$ and $n_{i+1} \cdot Var(W_i) = 1$

compromise → $Var(W) = \frac{2}{n_i + n_{i+1}}$

uniform version: $W \sim U\left(-\frac{\sqrt{6}}{\sqrt{n_i+n_{i+1}}},\ \frac{\sqrt{6}}{\sqrt{n_i+n_{i+1}}}\right)$

**Standard init** $W \sim U(-1/\sqrt{n},\ 1/\sqrt{n})$ gives $n \cdot Var(W) = 1/3$ → gradients shrink each layer → deeper layers learn slower

```python
he_init = tf.contrib.layers.variance_scaling_initializer()
hidden1 = fully_connected(X, n_hidden1, weights_initializer=he_init, scope="h1")
```

| Config | Result |
|--------|--------|
| Sigmoid + standard init | slow convergence, poor minima |
| Tanh + standard init | better than sigmoid, still unstable |
| Tanh + Xavier init | much better performance |
| Softsign (any init) | most robust, gentler saturation, stable |

**Key takeaways:**
1. deep nets fail because gradient variance multiplies across layers
2. activation saturation kills learning (sigmoid worst, softsign best)
3. zero-centered activations help (tanh > sigmoid)
4. Xavier init stabilizes both forward & backward signal flow
5. cross-entropy > quadratic loss for classification

### Non Saturating Activation Functions

sigmoid was inspired by biology but turns out other functions work much better in deep nets

**ReLU** — $f(z) = \max(0, z)$
- doesn't saturate for positive values, fast to compute
- but suffers from **dying ReLUs**: if weighted sum goes negative → output stuck at 0 → gradient is 0 → neuron never recovers
- large learning rate makes it worse (up to half the neurons can die)

**Leaky ReLU** — $\text{LeakyReLU}_\alpha(z) = \max(\alpha z,\ z)$
- small slope $\alpha$ for $z < 0$ (typically 0.01, but $\alpha = 0.2$ often works better)
- never fully dies — can go into "coma" but eventually wake up
- always outperforms strict ReLU

| Variant | How $\alpha$ works |
|---------|-------------------|
| Leaky ReLU | fixed hyperparameter (e.g. 0.01 or 0.2) |
| RReLU (Randomized) | $\alpha$ random during training, fixed avg at test — acts as regularizer |
| PReLU (Parametric) | $\alpha$ learned via backprop — best on large datasets, risk of overfitting on small ones |

**ELU (Exponential Linear Unit)** — outperformed all ReLU variants

$$
\text{ELU}_\alpha(z) = \begin{cases} \alpha(e^z - 1) & \text{if } z < 0 \\ z & \text{if } z \geq 0 \end{cases}
$$

why ELU is better:
1. takes negative values → avg output closer to 0 → reduces vanishing gradients
2. nonzero gradient for $z < 0$ → no dying units
3. smooth everywhere (including $z = 0$) → gradient descent doesn't bounce

drawback: slower to compute (exponential), but faster convergence compensates during training

**Ranking:** ELU > leaky ReLU (& variants) > ReLU > tanh > sigmoid
- care about speed? → leaky ReLU
- don't want to tune $\alpha$? → use defaults (0.01 for leaky ReLU, 1 for ELU)
- overfitting? → try RReLU
- huge training set? → try PReLU

```python
# ELU in TensorFlow
hidden1 = fully_connected(X, n_hidden1, activation_fn=tf.nn.elu)

# Leaky ReLU (custom)
def leaky_relu(z, name=None):
    return tf.maximum(0.01 * z, z, name=name)

hidden1 = fully_connected(X, n_hidden1, activation_fn=leaky_relu)
```

### Batch Normalization
to solve vanishing/exploding gradients problems
operation lets the model learn the optimal scale and mean of the inputs for each layer.

**Algorithm** — for each mini-batch $B$ with $m_B$ instances:

1. $\mu_B = \frac{1}{m_B} \sum_{i=1}^{m_B} \mathbf{x}^{(i)}$
2. $\sigma_B^2 = \frac{1}{m_B} \sum_{i=1}^{m_B} (\mathbf{x}^{(i)} - \mu_B)^2$
3. $\hat{\mathbf{x}}^{(i)} = \frac{\mathbf{x}^{(i)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$
4. $\mathbf{z}^{(i)} = \gamma\, \hat{\mathbf{x}}^{(i)} + \beta$

where:
- $\mu_B$ — empirical mean over the mini-batch
- $\sigma_B$ — empirical std dev over the mini-batch
- $m_B$ — number of instances in the mini-batch
- $\hat{\mathbf{x}}^{(i)}$ — zero-centered and normalized input
- $\gamma$ — scaling parameter (learned)
- $\beta$ — shifting/offset parameter (learned)
- $\epsilon$ — tiny number to avoid division by zero (typically $10^{-3}$), called *smoothing term*
- $\mathbf{z}^{(i)}$ — BN output: scaled and shifted version of normalized input

```python
import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
bn_params = {
'is_training': is_training,
'decay': 0.99,
'updates_collections': None
}
hidden1 = fully_connected(X, n_hidden1, scope="hidden1",
normalizer_fn=batch_norm, normalizer_params=bn_params)
hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2",
normalizer_fn=batch_norm, normalizer_params=bn_params)
logits = fully_connected(hidden2, n_outputs, activation_fn=None,scope="outputs",
normalizer_fn=batch_norm, normalizer_params=bn_params)


with tf.contrib.framework.arg_scope(
[fully_connected],
normalizer_fn=batch_norm,
normalizer_params=bn_params):
hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
logits = fully_connected(hidden2, n_outputs, scope="outputs",
activation_fn=None)

with tf.Session() as sess:
sess.run(init)
for epoch in range(n_epochs):
[...]
for X_batch, y_batch in zip(X_batches, y_batches):
sess.run(training_op,
feed_dict={is_training: True, X: X_batch, y: y_batch})
accuracy_score = accuracy.eval(
feed_dict={is_training: False, X: X_test_scaled, y: y_test}))
print(accuracy_score)
```

### Gradient Clipping
clip the gradients during backpropagation so that they never exceed some threshold
```python
threshold = 1.0
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(loss)
capped_gvs = [(tf.clip_by_value(grad,
-threshold, threshold), var)
for grad, var in grads_and_vars]
training_op = optimizer.apply_gradients(capped_gvs)
```

## Reusing Pretrained Layers
don't do training yourself, you should use pre trained neural network{transfer learning}
**Reusing a TensorFlow Model**
```python
[...] # construct the original model
with tf.Session() as sess:
    saver.restore(sess, "./my_original_model.ckpt")
    [...] # Train it on your new task

[...] # build new model with the same definition as before for hidden layers 1-3
init = tf.global_variables_initializer()
reuse_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
scope="hidden[123]")
reuse_vars_dict = dict([(var.name, var.name) for var in reuse_vars])
original_saver = tf.Saver(reuse_vars_dict) # saver to restore the original model
new_saver = tf.Saver() # saver to save the new model
with tf.Session() as sess:
    sess.run(init)
    original_saver.restore("./my_original_model.ckpt") # restore layers 1 to 3
    [...] # train the new model
    new_saver.save("./my_new_model.ckpt") # save the whole model
```

**Reusing Models from Other Frameworks**
```python
original_w = [...] # Load the weights from the other framework
original_b = [...] # Load the biases from the other framework

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
[...] # # Build the rest of the model

# Get a handle on the variables created by fully_connected()
with tf.variable_scope("", default_name="", reuse=True): # root scope
    hidden1_weights = tf.get_variable("hidden1/weights")
    hidden1_biases = tf.get_variable("hidden1/biases")

# Create nodes to assign arbitrary values to the weights and biases
original_weights = tf.placeholder(tf.float32, shape=(n_inputs, n_hidden1))
original_biases = tf.placeholder(tf.float32, shape=(n_hidden1))
assign_hidden1_weights = tf.assign(hidden1_weights, original_weights)
assign_hidden1_biases = tf.assign(hidden1_biases, original_biases)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    sess.run(assign_hidden1_weights, feed_dict={original_weights: original_w})
    sess.run(assign_hidden1_biases, feed_dict={original_biases: original_b})
    [...] # Train the model on your new task
```

**Freezing the Lower Layers**
good idea to “freeze” their weights
when training the new DNN: if the lower-layer weights are fixed, then the higher-
layer weights will be easier to train
```python
train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="hidden[34]|outputs")
training_op = optimizer.minimize(loss, var_list=train_vars)
```

**Caching the Frozen Layers**
Since training goes through the whole dataset
many times, this will give you a huge speed boost as you will only need to go through
the frozen layers once per training instance

```python
hidden2_outputs = sess.run(hidden2, feed_dict={X: X_train})

#build batches of outputs from hidden layer 2 and feed them to the training operation
import numpy as np
n_epochs = 100
n_batches = 500
for epoch in range(n_epochs):
    shuffled_idx = rnd.permutation(len(hidden2_outputs))
    hidden2_batches = np.array_split(hidden2_outputs[shuffled_idx], n_batches)
    y_batches = np.array_split(y_train[shuffled_idx], n_batches)
    for hidden2_batch, y_batch in zip(hidden2_batches, y_batches):
        sess.run(training_op, feed_dict={hidden2: hidden2_batch, y: y_batch})
```

## Fast Optimisers

use fast optimiser; most popular ones: Momentum optimization, Nesterov Accelerated Gradient, AdaGrad, RMSProp, and finally Adam optimization.

**Momentum Optimisation:**
ball rolling down: start slowly, speed increases, we will reach terminal velocity at some point of time
cares a great deal about what previous gradients were: at
each iteration, it adds the local gradient to the momentum vector m (multiplied by the
learning rate η), and it updates the weights by simply subtracting this momentum vector

**Momentum algorithm:**
1. $\mathbf{m} \leftarrow \beta\mathbf{m} + \eta\nabla_\theta J(\theta)$
2. $\theta \leftarrow \theta - \mathbf{m}$

where $\beta$ is the momentum (typically 0.9), $\eta$ is the learning rate, $\nabla_\theta J(\theta)$ is the gradient of the cost function

```python
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9)
```
**Nesterov Accelerated Gradient**

1. $\mathbf{m} \leftarrow \beta\mathbf{m} + \eta\nabla_\theta J(\theta + \beta\mathbf{m})$
2. $\theta \leftarrow \theta - \mathbf{m}$

measures gradient not at current position $\theta$ but slightly ahead at $\theta + \beta\mathbf{m}$ — generally converges faster than standard momentum

```python
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
momentum=0.9, use_nesterov=True)
```

**RMSProp:**
accumulating only the gradients from the most recent iterations

1. $\mathbf{s} \leftarrow \beta\mathbf{s} + (1 - \beta)\nabla_\theta J(\theta) \otimes \nabla_\theta J(\theta)$
2. $\theta \leftarrow \theta - \eta\, \nabla_\theta J(\theta) \oslash \sqrt{\mathbf{s} + \epsilon}$

where $\beta$ is the decay rate (typically 0.9), $\otimes$ is element-wise multiply, $\oslash$ is element-wise division

```python
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
momentum=0.9, decay=0.9, epsilon=1e-10)
```

**Adam Optimization:**
momentum + RMSProp

1. $\mathbf{m} \leftarrow \beta_1\mathbf{m} + (1 - \beta_1)\nabla_\theta J(\theta)$
2. $\mathbf{s} \leftarrow \beta_2\mathbf{s} + (1 - \beta_2)\nabla_\theta J(\theta) \otimes \nabla_\theta J(\theta)$
3. $\mathbf{m} \leftarrow \frac{\mathbf{m}}{1 - \beta_1^T}$
4. $\mathbf{s} \leftarrow \frac{\mathbf{s}}{1 - \beta_2^T}$
5. $\theta \leftarrow \theta - \eta\, \mathbf{m} \oslash \sqrt{\mathbf{s} + \epsilon}$

where $T$ is the iteration number, steps 3–4 are bias correction (since $\mathbf{m}$ and $\mathbf{s}$ are initialized at 0)

```python
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
```

**learning rate scheduling:**
```python
initial_learning_rate = 0.1
decay_steps = 10000
decay_rate = 1/10
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
decay_steps, decay_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
training_op = optimizer.minimize(loss, global_step=global_step)
```

## Avoiding Overfitting Through Regularization

**Early Stopping**: just interrupt training when its performance on the validation set starts dropping.

**l1 and l2 regularisation**: 
```python
[...] # construct the neural network
base_loss = tf.reduce_mean(xentropy, name="avg_xentropy")
reg_losses = tf.reduce_sum(tf.abs(weights1)) + tf.reduce_sum(tf.abs(weights2))
loss = tf.add(base_loss, scale * reg_losses, name="loss")

with arg_scope(
[fully_connected],weights_regularizer=tf.contrib.layers.l1_regularizer(scale=0.01)):
    hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")
    logits = fully_connected(hidden2, n_outputs, activation_fn=None,scope="out")

reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
loss = tf.add_n([base_loss] + reg_losses, name="loss")
```

**DropOut**: Dropout neurons, so their neightbouring will work more
multiply each input connection weight by the keep probability (1 – p) after training

```python
from tensorflow.contrib.layers import dropout
[...]
is_training = tf.placeholder(tf.bool, shape=(), name='is_training')
keep_prob = 0.5
X_drop = dropout(X, keep_prob, is_training=is_training)
hidden1 = fully_connected(X_drop, n_hidden1, scope="hidden1")
hidden1_drop = dropout(hidden1, keep_prob, is_training=is_training)
hidden2 = fully_connected(hidden1_drop, n_hidden2, scope="hidden2")
hidden2_drop = dropout(hidden2, keep_prob, is_training=is_training)
logits = fully_connected(hidden2_drop, n_outputs, activation_fn=None,
scope="outputs")
```

**Max-Norm Regularization**: for each neuron, constrains incoming connection weights $\mathbf{w}$ such that $\|\mathbf{w}\|_2 \leq r$, where $r$ is the max-norm hyperparameter and $\|\cdot\|_2$ is the $\ell_2$ norm.

```python
threshold = 1.0
clipped_weights = tf.clip_by_norm(weights, clip_norm=threshold, axes=1)
clip_weights = tf.assign(weights, clipped_weights)

with tf.Session() as sess:
    [...]
    for epoch in range(n_epochs):
        [...]
        for X_batch, y_batch in zip(X_batches, y_batches):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            clip_weights.eval()

hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
with tf.variable_scope("hidden1", reuse=True):
    weights1 = tf.get_variable("weights")

hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
hidden2 = fully_connected(hidden1, n_hidden2, scope="hidden2")

with tf.variable_scope("", default_name="", reuse=True): # root scope
    weights1 = tf.get_variable("hidden1/weights")
    weights2 = tf.get_variable("hidden2/weights")

for variable in tf.global_variables():
    print(variable.name)
```

more cleaner solution:
```python
def max_norm_regularizer(threshold, axes=1, name="max_norm",
collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights, clip_norm=threshold, axes=axes)
        clip_weights = tf.assign(weights, clipped, name=name)
        tf.add_to_collection(collection, clip_weights)
        return None # there is no regularization loss term
    return max_norm

max_norm_reg = max_norm_regularizer(threshold=1.0)
hidden1 = fully_connected(X, n_hidden1, scope="hidden1",weights_regularizer=max_norm_reg)
```

```python
clip_all_weights = tf.get_collection("max_norm")
with tf.Session() as sess:
    [...]
    for epoch in range(n_epochs):
        [...]
        for X_batch, y_batch in zip(X_batches, y_batches):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            sess.run(clip_all_weights)
```


**Data Augmentation**: generating new training instances from existing ones