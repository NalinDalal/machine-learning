"""
Q-Learning — Model-Free Reinforcement Learning
================================================
Chapter 16: Reinforcement Learning

Q-Learning adapts Q-Value Iteration for when the agent does NOT know
transition probabilities T(s,a,s') or rewards R(s,a,s') in advance.
It learns from experience using a running average update:

  Q(s,a) ← (1-α)·Q(s,a) + α·(r + γ·max_{a'} Q(s',a'))

This is an OFF-POLICY algorithm: the policy being trained (greedy on Q)
is not the one being executed (random exploration).

Key concepts demonstrated:
  - Learning rate decay for convergence
  - Random exploration policy
  - Comparison with Q-Value Iteration results
"""

import numpy as np
import numpy.random as rnd

# --- MDP definition (same as q-value-iteration.py) ---
nan = np.nan

T = np.array([
    [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
    [[0.0, 1.0, 0.0], [nan, nan, nan], [0.0, 0.0, 1.0]],
    [[nan, nan, nan], [0.8, 0.1, 0.1], [nan, nan, nan]],
])

R = np.array([
    [[10., 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[10., 0.0, 0.0], [nan, nan, nan], [0.0, 0.0, -50.]],
    [[nan, nan, nan], [40., 0.0, 0.0], [nan, nan, nan]],
])

possible_actions = [[0, 1, 2], [0, 2], [1]]

# --- Q-Learning ---
learning_rate0 = 0.05
learning_rate_decay = 0.1
n_iterations = 20000
discount_rate = 0.95
s = 0  # start in state 0

Q = np.full((3, 3), -np.inf)
for state, actions in enumerate(possible_actions):
    Q[state, actions] = 0.0

print(f"Running Q-Learning ({n_iterations} iterations, γ={discount_rate})...\n")

for iteration in range(n_iterations):
    a = rnd.choice(possible_actions[s])           # choose random action
    sp = rnd.choice(range(3), p=T[s, a])          # sample next state from T
    reward = R[s, a, sp]

    # Decaying learning rate for convergence
    learning_rate = learning_rate0 / (1 + iteration * learning_rate_decay)

    # Q-Learning update rule
    Q[s, a] = learning_rate * Q[s, a] + (1 - learning_rate) * (
        reward + discount_rate * np.max(Q[sp])
    )

    s = sp  # move to next state

# --- Results ---
print("Learned Q-Values:")
print(Q)
print()

action_names = {0: "a0", 1: "a1", 2: "a2"}
optimal_actions = np.argmax(Q, axis=1)
print("Learned optimal policy:")
for s in range(3):
    a = optimal_actions[s]
    print(f"  State s{s} → {action_names[a]} (Q = {Q[s, a]:.2f})")

print("\nExpected (from Q-Value Iteration):")
print("  s0 → a0, s1 → a2 (go through fire!), s2 → a1")
print("  Q-Learning converges to the same result despite random exploration!")


# --- Bonus: ε-greedy exploration ---
print("\n--- Q-Learning with ε-greedy exploration ---")

Q_eg = np.full((3, 3), -np.inf)
for state, actions in enumerate(possible_actions):
    Q_eg[state, actions] = 0.0

s = 0
eps_max = 1.0
eps_min = 0.05
eps_decay_steps = 10000

for iteration in range(n_iterations):
    # ε-greedy: explore randomly with probability ε, exploit with 1-ε
    epsilon = max(eps_min, eps_max - (eps_max - eps_min) * iteration / eps_decay_steps)

    if rnd.rand() < epsilon:
        a = rnd.choice(possible_actions[s])  # random action (explore)
    else:
        # Greedy: pick best known action
        q_vals = Q_eg[s]
        valid_mask = np.array([i in possible_actions[s] for i in range(3)])
        masked_q = np.where(valid_mask, q_vals, -np.inf)
        a = np.argmax(masked_q)

    sp = rnd.choice(range(3), p=T[s, a])
    reward = R[s, a, sp]
    learning_rate = learning_rate0 / (1 + iteration * learning_rate_decay)
    Q_eg[s, a] = learning_rate * Q_eg[s, a] + (1 - learning_rate) * (
        reward + discount_rate * np.max(Q_eg[sp])
    )
    s = sp

print("ε-greedy Q-Values:")
print(Q_eg)

optimal_eg = np.argmax(Q_eg, axis=1)
print("\nε-greedy optimal policy:")
for s in range(3):
    a = optimal_eg[s]
    print(f"  State s{s} → {action_names[a]} (Q = {Q_eg[s, a]:.2f})")
