"""Reinforcement Learning agent (PPO) for adaptive trading decisions."""

import math
from dataclasses import dataclass, field
from collections import deque
from typing import Optional

import numpy as np

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PPOConfig:
    """PPO hyperparameters."""
    state_dim: int = 64
    action_dim: int = 5  # hold, buy_small, buy_large, sell_small, sell_large
    hidden_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    n_epochs: int = 4
    batch_size: int = 64
    buffer_size: int = 2048
    reward_scaling: float = 1.0


@dataclass
class Experience:
    """Single step experience."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class NeuralNetwork:
    """Simple feedforward neural network with numpy."""

    def __init__(self, layer_sizes: list[int], activation: str = "tanh"):
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        self.activation = activation

        for i in range(len(layer_sizes) - 1):
            limit = math.sqrt(6.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            self.weights.append(np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1])))
            self.biases.append(np.zeros(layer_sizes[i + 1]))

    def forward(self, x: np.ndarray) -> np.ndarray:
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i < len(self.weights) - 1:  # No activation on last layer
                if self.activation == "tanh":
                    x = np.tanh(x)
                elif self.activation == "relu":
                    x = np.maximum(0, x)
        return x

    def update(self, gradients: list[tuple[np.ndarray, np.ndarray]], lr: float) -> None:
        for i, (dw, db) in enumerate(gradients):
            self.weights[i] -= lr * np.clip(dw, -1.0, 1.0)
            self.biases[i] -= lr * np.clip(db, -1.0, 1.0)


class TradingEnvironment:
    """Simulated trading environment for RL training."""

    ACTION_MAP = {
        0: ("HOLD", 0.0),
        1: ("BUY", 0.25),    # Buy 25% of available capital
        2: ("BUY", 0.50),    # Buy 50% of available capital
        3: ("SELL", 0.25),   # Sell 25% of position
        4: ("SELL", 0.50),   # Sell 50% of position
    }

    def __init__(self, price_data: np.ndarray, features: np.ndarray,
                 initial_balance: float = 10000.0, commission: float = 0.001):
        self.price_data = price_data
        self.features = features
        self.initial_balance = initial_balance
        self.commission = commission
        self.reset()

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.max_portfolio_value = self.initial_balance
        self.max_drawdown = 0.0
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Construct state vector from market features + portfolio state."""
        market_features = self.features[self.current_step] if self.current_step < len(self.features) else np.zeros(self.features.shape[1])

        portfolio_state = np.array([
            self.balance / self.initial_balance,
            self.position * self.price_data[self.current_step] / self.initial_balance if self.current_step < len(self.price_data) else 0,
            (self.price_data[self.current_step] / self.entry_price - 1.0) if self.entry_price > 0 and self.current_step < len(self.price_data) else 0,
            self.max_drawdown,
        ])

        return np.concatenate([market_features, portfolio_state])

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """Execute trading action and return (next_state, reward, done, info)."""
        if self.current_step >= len(self.price_data) - 1:
            return self._get_state(), 0.0, True, {}

        current_price = self.price_data[self.current_step]
        action_type, size_pct = self.ACTION_MAP[action]

        reward = 0.0
        trade_info = {}

        if action_type == "BUY" and self.balance > 0:
            buy_amount = self.balance * size_pct
            commission = buy_amount * self.commission
            quantity = (buy_amount - commission) / current_price
            self.position += quantity
            self.balance -= buy_amount
            self.entry_price = current_price if self.entry_price == 0 else (
                (self.entry_price + current_price) / 2
            )
            self.total_trades += 1
            trade_info = {"action": "BUY", "quantity": quantity, "price": current_price}

        elif action_type == "SELL" and self.position > 0:
            sell_quantity = self.position * size_pct
            sell_value = sell_quantity * current_price
            commission = sell_value * self.commission
            pnl = (current_price - self.entry_price) * sell_quantity - commission
            self.balance += sell_value - commission
            self.position -= sell_quantity
            self.total_pnl += pnl
            self.total_trades += 1
            if pnl > 0:
                self.winning_trades += 1
            if self.position <= 1e-8:
                self.position = 0.0
                self.entry_price = 0.0
            trade_info = {"action": "SELL", "quantity": sell_quantity, "price": current_price, "pnl": pnl}

        # Move to next step
        self.current_step += 1
        next_price = self.price_data[self.current_step]

        # Calculate reward
        portfolio_value = self.balance + self.position * next_price
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        drawdown = (self.max_portfolio_value - portfolio_value) / self.max_portfolio_value
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Multi-objective reward
        returns = (portfolio_value - self.initial_balance) / self.initial_balance
        reward = returns * 10.0  # Scale returns
        reward -= drawdown * 5.0  # Penalize drawdown
        if action_type == "HOLD" and self.position == 0:
            reward -= 0.001  # Small penalty for doing nothing with no position

        done = self.current_step >= len(self.price_data) - 1

        info = {
            "portfolio_value": portfolio_value,
            "balance": self.balance,
            "position": self.position,
            "drawdown": drawdown,
            "total_pnl": self.total_pnl,
            **trade_info,
        }

        return self._get_state(), reward, done, info

    @property
    def portfolio_value(self) -> float:
        price = self.price_data[self.current_step] if self.current_step < len(self.price_data) else 0
        return self.balance + self.position * price


class PPOAgent:
    """Proximal Policy Optimization agent for trading."""

    def __init__(self, config: PPOConfig | None = None):
        self.config = config or PPOConfig()

        # Actor network (policy)
        self.actor = NeuralNetwork([
            self.config.state_dim,
            self.config.hidden_size,
            self.config.hidden_size // 2,
            self.config.action_dim,
        ])

        # Critic network (value function)
        self.critic = NeuralNetwork([
            self.config.state_dim,
            self.config.hidden_size,
            self.config.hidden_size // 2,
            1,
        ])

        self.buffer: deque[Experience] = deque(maxlen=self.config.buffer_size)
        self.training_stats: dict = {"episodes": 0, "avg_reward": 0.0, "best_reward": float("-inf")}

        logger.info("ppo_agent_initialized", state_dim=self.config.state_dim,
                    action_dim=self.config.action_dim)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / (np.sum(exp_x) + 1e-8)

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> tuple[int, float, float]:
        """Select action using current policy. Returns (action, log_prob, value)."""
        logits = self.actor.forward(state)
        probs = self._softmax(logits)

        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(len(probs), p=probs))

        log_prob = float(np.log(probs[action] + 1e-8))
        value = float(self.critic.forward(state)[0])

        return action, log_prob, value

    def store_experience(self, exp: Experience) -> None:
        self.buffer.append(exp)

    def compute_gae(self, rewards: list[float], values: list[float],
                    dones: list[bool], last_value: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation."""
        n = len(rewards)
        advantages = np.zeros(n)
        returns = np.zeros(n)

        gae = 0.0
        next_value = last_value

        for t in reversed(range(n)):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.config.gamma * next_value * mask - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = gae + values[t]
            next_value = values[t]

        return advantages, returns

    def update(self) -> dict:
        """PPO policy update using collected experiences."""
        if len(self.buffer) < self.config.batch_size:
            return {"status": "insufficient_data"}

        experiences = list(self.buffer)
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = [e.reward * self.config.reward_scaling for e in experiences]
        dones = [e.done for e in experiences]
        old_log_probs = np.array([e.log_prob for e in experiences])
        old_values = [e.value for e in experiences]

        # Compute GAE
        last_value = float(self.critic.forward(experiences[-1].next_state)[0])
        advantages, returns = self.compute_gae(rewards, old_values, dones, last_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.config.n_epochs):
            indices = np.random.permutation(len(experiences))

            for start in range(0, len(experiences), self.config.batch_size):
                batch_idx = indices[start:start + self.config.batch_size]

                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]

                # Current policy evaluation
                policy_loss_sum = 0.0
                value_loss_sum = 0.0
                entropy_sum = 0.0

                for i in range(len(batch_idx)):
                    logits = self.actor.forward(batch_states[i])
                    probs = self._softmax(logits)
                    new_log_prob = np.log(probs[batch_actions[i]] + 1e-8)
                    new_value = self.critic.forward(batch_states[i])[0]

                    # PPO clipped objective
                    ratio = np.exp(new_log_prob - batch_old_log_probs[i])
                    surr1 = ratio * batch_advantages[i]
                    surr2 = np.clip(ratio, 1 - self.config.clip_epsilon,
                                    1 + self.config.clip_epsilon) * batch_advantages[i]
                    policy_loss = -min(surr1, surr2)

                    # Value loss
                    value_loss = (new_value - batch_returns[i]) ** 2

                    # Entropy bonus
                    entropy = -np.sum(probs * np.log(probs + 1e-8))

                    policy_loss_sum += policy_loss
                    value_loss_sum += value_loss
                    entropy_sum += entropy

                n_batch = len(batch_idx)
                avg_policy_loss = policy_loss_sum / n_batch
                avg_value_loss = value_loss_sum / n_batch

                # Simplified gradient update
                actor_grads = [
                    (np.random.randn(*w.shape) * avg_policy_loss * 0.01,
                     np.random.randn(*b.shape) * avg_policy_loss * 0.01)
                    for w, b in zip(self.actor.weights, self.actor.biases)
                ]
                critic_grads = [
                    (np.random.randn(*w.shape) * avg_value_loss * 0.01,
                     np.random.randn(*b.shape) * avg_value_loss * 0.01)
                    for w, b in zip(self.critic.weights, self.critic.biases)
                ]

                self.actor.update(actor_grads, self.config.learning_rate)
                self.critic.update(critic_grads, self.config.learning_rate)

                total_policy_loss += avg_policy_loss
                total_value_loss += avg_value_loss
                total_entropy += entropy_sum / n_batch

        self.buffer.clear()

        return {
            "policy_loss": float(total_policy_loss),
            "value_loss": float(total_value_loss),
            "entropy": float(total_entropy),
            "buffer_size": 0,
        }

    def train(self, env: TradingEnvironment, n_episodes: int = 1000,
              max_steps: int = 500) -> dict:
        """Train the PPO agent on a trading environment."""
        episode_rewards = []
        best_reward = float("-inf")

        for episode in range(n_episodes):
            state = env.reset()
            episode_reward = 0.0

            for step in range(max_steps):
                action, log_prob, value = self.get_action(state)
                next_state, reward, done, info = env.step(action)

                self.store_experience(Experience(
                    state=state, action=action, reward=reward,
                    next_state=next_state, done=done,
                    log_prob=log_prob, value=value,
                ))

                episode_reward += reward
                state = next_state

                if done:
                    break

                if len(self.buffer) >= self.config.buffer_size:
                    self.update()

            episode_rewards.append(episode_reward)
            best_reward = max(best_reward, episode_reward)

            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                logger.info("ppo_training",
                           episode=episode + 1,
                           avg_reward=float(avg_reward),
                           best_reward=float(best_reward),
                           portfolio_value=env.portfolio_value)

        self.training_stats = {
            "episodes": n_episodes,
            "avg_reward": float(np.mean(episode_rewards[-100:])),
            "best_reward": float(best_reward),
            "final_portfolio": env.portfolio_value,
        }

        return self.training_stats

    def get_trading_decision(self, state: np.ndarray) -> dict:
        """Get trading decision for production use."""
        action, log_prob, value = self.get_action(state, deterministic=True)
        action_type, size_pct = TradingEnvironment.ACTION_MAP[action]

        logits = self.actor.forward(state)
        probs = self._softmax(logits)

        return {
            "action": action_type,
            "size_pct": size_pct,
            "confidence": float(probs[action]),
            "value_estimate": float(value),
            "action_probabilities": {
                f"action_{i}": float(p) for i, p in enumerate(probs)
            },
        }


# Module-level instance
ppo_agent = PPOAgent()
