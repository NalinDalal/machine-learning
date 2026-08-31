# Reinforcement Learning

Reinforcement learning (RL) studies agents that learn by interacting with an environment to maximize cumulative reward. There are no labeled training pairs — the agent discovers good behavior through trial and error.

Now we’re in a world where the model learns by **doing**.

## Core idea

An **agent** takes **actions** in an environment, observes states and rewards, and tries to learn a policy that maximizes expected cumulative return.

Key concepts:

- State ($s$): environment description (e.g., agent position in game).
- Action ($a$): what the agent does (e.g., move left).
- Reward ($r$): scalar feedback (e.g., +1 for success, -1 for failure).
- Policy ($\pi$): maps states to actions; can be deterministic $\pi(s)=a$ or stochastic $\pi(a|s)$.
- Value function: expected cumulative reward from a state (or state-action pair).

Objective (expected discounted return):
$$
J(\pi)=\mathbb{E}\left[\sum_{t=0}^\infty\gamma^t r_t\right],
$$
with discount factor $\gamma\in[0,1)$.

## Formal setup: MDP

**Markov Decision Process (MDP)**: tuple $(\mathcal{S},\mathcal{A},p,r,\gamma)$ with states $\mathcal{S}$, actions $\mathcal{A}$, transition kernel $p(s'\mid s,a)$ and reward function $r(s,a)$.

Two broad paradigms:

- Model-based RL: learn a model $p(s'\mid s,a)$ and plan with it.
- Model-free RL: learn policy/value directly from experience.

## Key Algorithm families

### 1. Value-Based Methods (Q-Learning)

Idea: learn an action-value function $Q(s,a)$ and act greedily.

Action-value under policy $\pi$:
$$
Q^\pi(s,a)=\mathbb{E}\left[\sum_{k=0}^\infty\gamma^k r_{t+k}\mid s_t=s,a_t=a\right].
$$

**Q-Learning (off-policy)**:
$$
Q(s,a)\leftarrow Q(s,a)+\alpha\big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\big].
$$

$\alpha$ is learning rate.

Pros: simple, effective for small discrete problems. 
Cons: scales poorly to large/continuous spaces (addressed by DQN — deep Q-networks — using neural nets, replay buffers, target networks).

### 2) Policy-based methods (REINFORCE)

Idea: parameterize policy $\pi(a|s;\theta)$ and optimize $J(\theta)$ directly.

Policy gradient (REINFORCE) estimator:
$$
\nabla J(\theta)\approx\sum_t\nabla_\theta\log\pi(a_t|s_t;\theta)\,G_t,
$$
where $G_t=\sum_{k=t}^\infty\gamma^{k-t}r_k$ is the return.

Pros: handles continuous actions and stochastic policies. Cons: high variance; often improved with baselines (actor-critic).

### 3) Actor–Critic methods

Combine policy (actor) and value (critic). The critic estimates a value (or advantage) to reduce variance of policy updates.

Advantage function: $A(s,a)=Q(s,a)-V(s)$.
Actor Update: $\nabla_\theta \log \pi(a|s; \theta) \cdot A(s, a)$
Actor update uses $A(s,a)$ as the learning signal; critic is trained to minimize TD error.

Actor–critic methods (A2C, PPO, A3C, etc.) strike a practical balance between stability and performance.

## Exploration vs exploitation

Agents must explore to discover rewards but exploit known good actions. Common strategies:

- $\epsilon$-greedy: with probability $\epsilon$ take a random action.
- Softmax / Boltzmann: sample actions proportionally to preferences. based on $Q%-value or policy probabilities.
- Upper Confidence Bound (UCB): Prioritise actions with high uncertaininty;balance mean and uncertainty in action selection.
- Thompson Sampling: sample from posterior over models/policies.

## Practical challenges

- Sparse Rewards — Hard to learn when rewards are rare.
  **Fix**: use Reward shaping or intrinsic curiosity.
- Sample Efficiency — Need tons of interactions.
  **Fix**: use Replay buffers, off-policy methods, or model-based approaches.
- Non-stationarity — Environment changes during learning
  **Fix**: stabilize with target networks, Slower updates.
- Credit Assignment — Which action cause the reward?
  **Fix**: handled via discounting, eligibility traces, or advantage estimators.

## Practical Tips

- Start with simple environments (e.g. CartPole, MountainCar, GridWorld).
- Normalize inputs where appropriate state/rewards for stability.
- Carefully tune hyperparameters: learning rate, discount factor $\gamma$, exploration rate/schedule $epsilon$.
- Use established libraries: Stable-Baselines3, RLlib, Gym / Gymnasium.
- Monitor metrics: Track cumulative rewards and episodic return, loss curves, policy entropy, value estimates.

## Why it matters

- Applications: robotics, games, autonomous systems, resource allocation.
- RL addresses sequential decision-making beyond supervised learning.
- Learning from interaction is a core ingredient for more general intelligence.