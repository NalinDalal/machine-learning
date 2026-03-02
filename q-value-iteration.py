"""
Q-Value Iteration — Markov Decision Process
=============================================
Chapter 16: Reinforcement Learning

Demonstrates Q-Value Iteration on a simple 3-state MDP where
transition probabilities and rewards are fully known.

MDP Structure:
  - 3 states: s0, s1, s2
  - s0 has actions: a0 (70% stay +10 reward, 30% → s1), a1 (stay), a2 (80% stay, 20% → s1)
  - s1 has actions: a0 (stay in s1), a2 (go to s2, -50 reward)
  - s2 has actions: a1 (80% → s0 +40 reward, 10% → s1, 10% → s2)

Algorithm:
  Q_{k+1}(s,a) = Σ_{s'} T(s,a,s') * [R(s,a,s') + γ * max_{a'} Q_k(s',a')]

Results with γ=0.95:
  - s0 → a0 (try for +10 reward)
  - s1 → a2 (go through the fire to reach s2!)
  - s2 → a1 (only option, leads back to s0 with +40)

With γ=0.9: s1 → a0 (stay put — future rewards discounted too heavily)
"""

import numpy as np

# --- Define the MDP ---
nan = np.nan  # represents impossible actions

# Transition probabilities: T[s, a, s']
T = np.array([
    # State s0: actions a0, a1, a2
    [[0.7, 0.3, 0.0],   # a0: 70% stay in s0, 30% go to s1
     [1.0, 0.0, 0.0],   # a1: 100% stay in s0
     [0.8, 0.2, 0.0]],  # a2: 80% stay in s0, 20% go to s1
    # State s1: actions a0, a1 (impossible), a2
    [[0.0, 1.0, 0.0],   # a0: 100% stay in s1
     [nan, nan, nan],    # a1: impossible
     [0.0, 0.0, 1.0]],  # a2: 100% go to s2
    # State s2: actions a0 (impossible), a1, a2 (impossible)
    [[nan, nan, nan],    # a0: impossible
     [0.8, 0.1, 0.1],   # a1: 80% → s0, 10% → s1, 10% → s2
     [nan, nan, nan]],   # a2: impossible
])

# Rewards: R[s, a, s']
R = np.array([
    # State s0
    [[10., 0.0, 0.0],   # a0: +10 if stay in s0
     [0.0, 0.0, 0.0],   # a1: no reward
     [0.0, 0.0, 0.0]],  # a2: no reward
    # State s1
    [[10., 0.0, 0.0],   # a0: +10 (but stays in s1, rare scenario)
     [nan, nan, nan],    # a1: impossible
     [0.0, 0.0, -50.]], # a2: -50 if go to s2
    # State s2
    [[nan, nan, nan],    # a0: impossible
     [40., 0.0, 0.0],   # a1: +40 if go to s0
     [nan, nan, nan]],   # a2: impossible
])

possible_actions = [[0, 1, 2], [0, 2], [1]]

# --- Q-Value Iteration ---
Q = np.full((3, 3), -np.inf)  # -inf for impossible actions
for state, actions in enumerate(possible_actions):
    Q[state, actions] = 0.0    # initial Q-value = 0 for possible actions

discount_rate = 0.95
n_iterations = 100

print(f"Running Q-Value Iteration (γ={discount_rate}, {n_iterations} iterations)...\n")

for iteration in range(n_iterations):
    Q_prev = Q.copy()
    for s in range(3):
        for a in possible_actions[s]:
            Q[s, a] = np.sum([
                T[s, a, sp] * (R[s, a, sp] + discount_rate * np.max(Q_prev[sp]))
                for sp in range(3)
            ])

# --- Results ---
print("Q-Values:")
print(Q)
print()

optimal_actions = np.argmax(Q, axis=1)
action_names = {0: "a0", 1: "a1", 2: "a2"}
print("Optimal policy:")
for s in range(3):
    a = optimal_actions[s]
    print(f"  State s{s} → {action_names[a]} (Q = {Q[s, a]:.2f})")

# --- Compare: with lower discount rate ---
print("\n--- With discount_rate = 0.9 ---")
Q2 = np.full((3, 3), -np.inf)
for state, actions in enumerate(possible_actions):
    Q2[state, actions] = 0.0

for iteration in range(n_iterations):
    Q2_prev = Q2.copy()
    for s in range(3):
        for a in possible_actions[s]:
            Q2[s, a] = np.sum([
                T[s, a, sp] * (R[s, a, sp] + 0.9 * np.max(Q2_prev[sp]))
                for sp in range(3)
            ])

optimal_actions_2 = np.argmax(Q2, axis=1)
print("Optimal policy (γ=0.9):")
for s in range(3):
    a = optimal_actions_2[s]
    print(f"  State s{s} → {action_names[a]} (Q = {Q2[s, a]:.2f})")
print("Note: with γ=0.9, s1 → a0 (stay put) instead of a2 (fire) — future less valued!")
