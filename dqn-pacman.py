"""
Deep Q-Network (DQN) — Ms. Pac-Man
====================================
Chapter 16: Reinforcement Learning

Full Deep Q-Learning implementation for Atari Ms. Pac-Man:

Architecture:
  - Preprocessing: crop, downsize to 88×80, grayscale, normalize [-1, 1]
  - DQN: 3 conv layers (32-64-64 maps) + 2 FC layers (512 hidden → 9 actions)
  - Two networks: Actor (plays) & Critic (learns)
  - Critic is periodically copied to Actor

Key innovations:
  - Replay memory: stores experiences, samples random batches (breaks correlation)
  - ε-greedy exploration: ε decays from 1.0 to 0.05 over 50k steps
  - Actor-Critic: actor plays, critic learns from actor's experience

Training loop:
  1. Actor plays with ε-greedy policy
  2. Store (state, action, reward, next_state, continue) in replay memory
  3. Sample random batch from replay memory
  4. Compute target: y = r + γ·max Q_actor(s', a')
  5. Train critic to predict target Q-values (MSE loss)
  6. Periodically copy critic → actor

Cost function:
  J(θ_critic) = (1/m) Σ (y_i - Q(s_i, a_i, θ_critic))²

Dependencies: pip3 install --upgrade 'gym[all]'
"""

import os
import numpy as np
import numpy.random as rnd
import tensorflow as tf
from tensorflow.contrib.layers import convolution2d, fully_connected
from collections import deque
import gym

# =============================================
# 1. ENVIRONMENT SETUP & PREPROCESSING
# =============================================

env = gym.make("MsPacman-v0")
print(f"Observation shape: {env.observation_space.shape}")  # (210, 160, 3)
print(f"Action space: {env.action_space}")                  # Discrete(9)

mspacman_color = np.array([210, 164, 74]).mean()

def preprocess_observation(obs):
    """Crop, downsize to 88x80, grayscale, improve contrast, normalize."""
    img = obs[1:176:2, ::2]         # crop and downsize
    img = img.mean(axis=2)           # to greyscale
    img[img == mspacman_color] = 0   # improve Ms. Pac-Man contrast
    img = (img - 128) / 128 - 1     # normalize from -1.0 to 1.0
    return img.reshape(88, 80, 1)


# =============================================
# 2. DQN ARCHITECTURE
# =============================================

input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8, 8), (4, 4), (3, 3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 10     # conv3 has 64 maps of 11x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n  # 9 discrete actions
learning_rate = 0.001
initializer = tf.contrib.layers.variance_scaling_initializer()


def q_network(X_state, scope):
    """Build a DQN: 3 conv layers + 2 FC layers.
    
    Returns:
        outputs: Q-value estimates for each action
        trainable_vars_by_name: dict of trainable variables (for copying)
    """
    prev_layer = X_state
    with tf.variable_scope(scope) as scope:
        for n_maps, kernel_size, stride, padding, activation in zip(
                conv_n_maps, conv_kernel_sizes, conv_strides,
                conv_paddings, conv_activation):
            prev_layer = convolution2d(
                prev_layer, num_outputs=n_maps, kernel_size=kernel_size,
                stride=stride, padding=padding, activation_fn=activation,
                weights_initializer=initializer)

        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = fully_connected(
            last_conv_layer_flat, n_hidden,
            activation_fn=hidden_activation,
            weights_initializer=initializer)
        outputs = fully_connected(
            hidden, n_outputs, activation_fn=None,
            weights_initializer=initializer)

        trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        trainable_vars_by_name = {
            var.name[len(scope.name):]: var for var in trainable_vars}

    return outputs, trainable_vars_by_name


# =============================================
# 3. BUILD TWO DQNs (ACTOR & CRITIC)
# =============================================

X_state = tf.placeholder(tf.float32,
                          shape=[None, input_height, input_width, input_channels])

actor_q_values, actor_vars = q_network(X_state, scope="q_networks/actor")
critic_q_values, critic_vars = q_network(X_state, scope="q_networks/critic")

# Operation to copy critic weights → actor
copy_ops = [actor_var.assign(critic_vars[var_name])
            for var_name, actor_var in actor_vars.items()]
copy_critic_to_actor = tf.group(*copy_ops)

# Critic training operations
X_action = tf.placeholder(tf.int32, shape=[None])
q_value = tf.reduce_sum(
    critic_q_values * tf.one_hot(X_action, n_outputs),
    axis=1, keep_dims=True)

y = tf.placeholder(tf.float32, shape=[None, 1])
cost = tf.reduce_mean(tf.square(y - q_value))
global_step = tf.Variable(0, trainable=False, name='global_step')
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(cost, global_step=global_step)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


# =============================================
# 4. REPLAY MEMORY
# =============================================

replay_memory_size = 10000
replay_memory = deque([], maxlen=replay_memory_size)


def sample_memories(batch_size):
    """Sample a random batch of experiences from replay memory."""
    indices = rnd.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1),
            cols[3], cols[4].reshape(-1, 1))


# =============================================
# 5. ε-GREEDY EXPLORATION
# =============================================

eps_min = 0.05
eps_max = 1.0
eps_decay_steps = 50000


def epsilon_greedy(q_values, step):
    """ε-greedy action selection: random with prob ε, greedy with 1-ε."""
    epsilon = max(eps_min, eps_max - (eps_max - eps_min) * step / eps_decay_steps)
    if rnd.rand() < epsilon:
        return rnd.randint(n_outputs)      # random action
    else:
        return np.argmax(q_values)          # optimal action


# =============================================
# 6. TRAINING LOOP
# =============================================

n_steps = 100000           # total training steps
training_start = 1000      # warmup: no training for first 1000 game iterations
training_interval = 3      # train every 3 game iterations
save_steps = 50            # save model every 50 training steps
copy_steps = 25            # copy critic → actor every 25 training steps
discount_rate = 0.95
skip_start = 90            # skip start of each game (just waiting time)
batch_size = 50
iteration = 0
checkpoint_path = "./my_dqn.ckpt"
done = True                # env needs to be reset

print("\n--- Starting Deep Q-Learning Training ---")
print(f"Total training steps: {n_steps}")
print(f"Replay memory size: {replay_memory_size}")
print(f"ε decay: {eps_max} → {eps_min} over {eps_decay_steps} steps\n")

with tf.Session() as sess:
    if os.path.isfile(checkpoint_path + ".index"):
        saver.restore(sess, checkpoint_path)
        print("Restored model from checkpoint.")
    else:
        init.run()
        print("Initialized fresh model.")

    while True:
        step = global_step.eval()
        if step >= n_steps:
            break
        iteration += 1

        if done:  # game over, start again
            obs = env.reset()
            for skip in range(skip_start):  # skip boring start
                obs, reward, done, info = env.step(0)
            state = preprocess_observation(obs)

        # Actor evaluates what to do
        q_values = actor_q_values.eval(feed_dict={X_state: [state]})
        action = epsilon_greedy(q_values, step)

        # Actor plays
        obs, reward, done, info = env.step(action)
        next_state = preprocess_observation(obs)

        # Memorize experience
        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state

        if iteration < training_start or iteration % training_interval != 0:
            continue

        # --- Critic learns ---
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (
            sample_memories(batch_size))

        # Actor estimates future Q-values
        next_q_values = actor_q_values.eval(
            feed_dict={X_state: X_next_state_val})
        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)

        # Target Q-values: y = r + γ·max Q(s', a')
        y_val = rewards + continues * discount_rate * max_next_q_values

        # Train critic
        training_op.run(feed_dict={
            X_state: X_state_val,
            X_action: X_action_val,
            y: y_val})

        # Periodically copy critic → actor
        if step % copy_steps == 0:
            copy_critic_to_actor.run()

        # Save checkpoint
        if step % save_steps == 0:
            saver.save(sess, checkpoint_path)

        # Log progress
        if step % 1000 == 0:
            epsilon = max(eps_min,
                          eps_max - (eps_max - eps_min) * step / eps_decay_steps)
            print(f"Step {step:6d} | Iteration {iteration:6d} | "
                  f"ε = {epsilon:.3f} | Replay size = {len(replay_memory)}")

    saver.save(sess, checkpoint_path)
    print("\nTraining complete! Final model saved.")

env.close()
