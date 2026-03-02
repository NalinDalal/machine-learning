"""
CartPole Basic Environment — OpenAI Gym Introduction
=====================================================
Chapter 16: Reinforcement Learning

Demonstrates:
- Creating a CartPole-v0 environment
- Resetting and rendering the environment
- Understanding observations (cart position, velocity, pole angle, angular velocity)
- Taking actions (0=left, 1=right) via env.step()
- Running a simple hardcoded policy (accelerate toward pole lean direction)
- Evaluating the basic policy over 500 episodes

The basic policy achieves ~42 mean reward (max ~68) — not great.
The cart oscillates left and right until the pole tilts too much.
"""

import gym
import numpy as np

# --- 1. Create and explore the CartPole environment ---
env = gym.make("CartPole-v0")
obs = env.reset()
print("Initial observation:", obs)
# obs = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

print("Action space:", env.action_space)  # Discrete(2): 0=left, 1=right
print("Observation space:", env.observation_space)

# --- 2. Take a single step ---
action = 1  # accelerate right
obs, reward, done, info = env.step(action)
print("\nAfter action=1 (right):")
print("  Observation:", obs)
print("  Reward:", reward)       # 1.0 at every step
print("  Done:", done)           # True if pole tilts too much
print("  Info:", info)

# --- 3. Hardcoded basic policy ---
# Accelerate left when pole leans left, right when it leans right
def basic_policy(obs):
    """Simple policy: follow the pole's lean direction."""
    angle = obs[2]
    return 0 if angle < 0 else 1  # 0=left, 1=right

# --- 4. Evaluate the basic policy over 500 episodes ---
totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(1000):  # max 1000 steps per episode
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)

print("\n--- Basic Policy Results (500 episodes) ---")
print(f"  Mean: {np.mean(totals):.2f}")
print(f"  Std:  {np.std(totals):.2f}")
print(f"  Min:  {np.min(totals):.0f}")
print(f"  Max:  {np.max(totals):.0f}")
# Typical: mean ~42, max ~68 — the cart oscillates until pole falls

env.close()
