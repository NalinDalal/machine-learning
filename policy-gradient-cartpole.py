"""
Policy Gradient (REINFORCE) — CartPole Training
=================================================
Chapter 16: Reinforcement Learning

Full implementation of the REINFORCE algorithm:
1. Play game several times, compute gradients at each step (don't apply yet)
2. Compute discounted & normalized action scores
3. Multiply each gradient by its action score (reinforce good, discourage bad)
4. Compute mean gradient, perform one gradient descent step

Key concepts:
- Credit assignment: discounted rewards with γ = 0.95
- Reward normalization: subtract mean, divide by std
- Gradient manipulation: compute_gradients() → tweak → apply_gradients()

After ~250 iterations, the policy learns to balance the pole quite well.
"""

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import gym

# =============================================
# 1. CONSTRUCTION PHASE
# =============================================

n_inputs = 4
n_hidden = 4
n_outputs = 1
learning_rate = 0.01
initializer = tf.contrib.layers.variance_scaling_initializer()

# Neural network policy
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.elu,
                         weights_initializer=initializer)
logits = fully_connected(hidden, n_outputs, activation_fn=None,
                         weights_initializer=initializer)
outputs = tf.nn.sigmoid(logits)
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

# Target: pretend chosen action was the best
y = 1. - tf.to_float(action)

# Cost function and gradient computation (don't apply yet!)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]

# Placeholders for tweaked gradients
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))

training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


# =============================================
# 2. HELPER FUNCTIONS — Discounted Rewards
# =============================================

def discount_rewards(rewards, discount_rate):
    """Compute discounted cumulative rewards (backwards)."""
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, discount_rate):
    """Discount and normalize rewards across all episodes.
    
    Positive scores → good actions, Negative scores → bad actions.
    """
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]


# Quick sanity check
print("Discount test:", discount_rewards([10, 0, -50], discount_rate=0.8))
# Expected: [-22., -40., -50.]


# =============================================
# 3. EXECUTION PHASE — Train with REINFORCE
# =============================================

env = gym.make("CartPole-v0")

n_iterations = 250          # training iterations
n_max_steps = 1000          # max steps per episode
n_games_per_update = 10     # train policy every 10 episodes
save_iterations = 10        # save model every 10 training iterations
discount_rate = 0.95

with tf.Session() as sess:
    init.run()

    for iteration in range(n_iterations):
        all_rewards = []      # raw rewards for each episode
        all_gradients = []    # gradients at each step of each episode

        # --- Play n_games_per_update episodes ---
        for game in range(n_games_per_update):
            current_rewards = []
            current_gradients = []
            obs = env.reset()

            for step in range(n_max_steps):
                action_val, gradients_val = sess.run(
                    [action, gradients],
                    feed_dict={X: obs.reshape(1, n_inputs)})
                obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(reward)
                current_gradients.append(gradients_val)
                if done:
                    break

            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        # --- Policy update using REINFORCE ---
        # Discount and normalize action scores
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)

        feed_dict = {}
        for var_index, grad_placeholder in enumerate(gradient_placeholders):
            # Multiply gradients by action scores, compute mean
            mean_gradients = np.mean(
                [reward * all_gradients[game_index][step][var_index]
                 for game_index, rewards in enumerate(all_rewards)
                 for step, reward in enumerate(rewards)],
                axis=0)
            feed_dict[grad_placeholder] = mean_gradients

        sess.run(training_op, feed_dict=feed_dict)

        if iteration % save_iterations == 0:
            saver.save(sess, "./my_policy_net_pg.ckpt")

        # --- Periodic evaluation ---
        if iteration % 25 == 0:
            eval_rewards = []
            for _ in range(20):
                obs = env.reset()
                ep_reward = 0
                for step in range(n_max_steps):
                    action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
                    obs, reward, done, info = env.step(action_val[0][0])
                    ep_reward += reward
                    if done:
                        break
                eval_rewards.append(ep_reward)
            print(f"Iteration {iteration:3d} | Mean reward: {np.mean(eval_rewards):.1f}")

    saver.save(sess, "./my_policy_net_pg.ckpt")
    print("\nTraining complete! Model saved.")

env.close()
