"""
Neural Network Policy for CartPole — TensorFlow
=================================================
Chapter 16: Reinforcement Learning

Builds a simple neural network policy:
- Input: 4 observation floats (cart position, velocity, pole angle, angular velocity)
- Hidden: 4 neurons with ELU activation
- Output: 1 neuron with sigmoid → probability of going left
- Action selection: multinomial sampling from [p(left), 1-p(left)]

This demonstrates the architecture only. For full training with
policy gradients, see policy-gradient-cartpole.py.
"""

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import gym
import numpy as np

# --- 1. Define neural network architecture ---
n_inputs = 4       # == env.observation_space.shape[0]
n_hidden = 4       # simple task, 4 hidden neurons suffice
n_outputs = 1      # outputs probability of accelerating left

initializer = tf.contrib.layers.variance_scaling_initializer()

# --- 2. Build the neural network ---
X = tf.placeholder(tf.float32, shape=[None, n_inputs])

hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.elu,
                         weights_initializer=initializer)

logits = fully_connected(hidden, n_outputs, activation_fn=None,
                         weights_initializer=initializer)

outputs = tf.nn.sigmoid(logits)  # probability of going left

# --- 3. Select action randomly based on estimated probabilities ---
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

init = tf.global_variables_initializer()

# --- 4. Test the (untrained) policy ---
env = gym.make("CartPole-v0")

n_episodes = 50
totals = []

with tf.Session() as sess:
    init.run()

    for episode in range(n_episodes):
        obs = env.reset()
        episode_reward = 0

        for step in range(1000):
            action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
            obs, reward, done, info = env.step(action_val[0][0])
            episode_reward += reward
            if done:
                break

        totals.append(episode_reward)

print(f"\n--- Untrained NN Policy ({n_episodes} episodes) ---")
print(f"  Mean reward: {np.mean(totals):.2f}")
print(f"  Std:         {np.std(totals):.2f}")
print(f"  Min:         {np.min(totals):.0f}")
print(f"  Max:         {np.max(totals):.0f}")
print("  (Random-ish since weights are untrained)")

env.close()
