"""Deep Q-Network agent for trading strategy selection."""

import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from app.core.logging import get_logger
from app.services.ml.rl.environment import TradingEnvironment, TradingState

logger = get_logger(__name__)


class DQNetwork(nn.Module):
    """Dueling DQN with advantage separation."""

    def __init__(self, state_dim: int, n_actions: int = 4, hidden: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        value = self.value(feat)
        adv = self.advantage(feat)
        # Dueling: Q = V + (A - mean(A))
        return value + adv - adv.mean(dim=1, keepdim=True)


@dataclass
class Transition:
    """Experience replay memory entry."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class DQNAgent:
    """DQN agent that learns to select optimal trading strategies.

    Uses:
    - Dueling DQN architecture
    - Experience replay
    - Target network (soft update)
    - Epsilon-greedy exploration
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 5000,
        buffer_size: int = 50000,
        batch_size: int = 64,
        target_update_tau: float = 0.005,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_tau = target_update_tau

        self._device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Networks
        self.policy_net = DQNetwork(state_dim, n_actions).to(self._device)
        self.target_net = DQNetwork(state_dim, n_actions).to(self._device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=buffer_size)
        self._step_count = 0

    def _state_to_tensor(self, state: TradingState) -> np.ndarray:
        """Flatten TradingState into a feature vector."""
        return np.concatenate([
            state.features,
            [state.position, state.unrealized_pnl, state.equity_ratio, state.drawdown],
        ])

    def select_action(self, state: TradingState) -> int:
        """Epsilon-greedy action selection."""
        self.epsilon = self.epsilon_end + (1.0 - self.epsilon_end) * np.exp(
            -self._step_count / self.epsilon_decay
        )
        self._step_count += 1

        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)

        state_vec = self._state_to_tensor(state)
        with torch.no_grad():
            x = torch.FloatTensor(state_vec).unsqueeze(0).to(self._device)
            q_values = self.policy_net(x)
            return int(q_values.argmax(dim=1).item())

    def predict_action(self, state: TradingState) -> int:
        """Greedy action selection (no exploration)."""
        state_vec = self._state_to_tensor(state)
        with torch.no_grad():
            x = torch.FloatTensor(state_vec).unsqueeze(0).to(self._device)
            q_values = self.policy_net(x)
            return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: TradingState,
        action: int,
        reward: float,
        next_state: TradingState,
        done: bool,
    ) -> None:
        self.memory.append(
            Transition(
                state=self._state_to_tensor(state),
                action=action,
                reward=reward,
                next_state=self._state_to_tensor(next_state),
                done=done,
            )
        )

    def train_step(self) -> float:
        """Single training step from replay buffer."""
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(list(self.memory), self.batch_size)

        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self._device)
        actions = torch.LongTensor([t.action for t in batch]).to(self._device)
        rewards = torch.FloatTensor([t.reward for t in batch]).to(self._device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(
            self._device
        )
        dones = torch.FloatTensor([t.done for t in batch]).to(self._device)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Soft update target network
        for tp, pp in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            tp.data.copy_(
                self.target_update_tau * pp.data
                + (1 - self.target_update_tau) * tp.data
            )

        return loss.item()

    def train_on_env(
        self,
        env: TradingEnvironment,
        episodes: int = 100,
    ) -> dict:
        """Train agent on a trading environment."""
        total_rewards = []

        for ep in range(episodes):
            state = env.reset()
            episode_reward = 0

            while True:
                action = self.select_action(state)
                result = env.step(action)

                self.store_transition(
                    state, action, result.reward, result.state, result.done
                )
                loss = self.train_step()

                episode_reward += result.reward
                state = result.state

                if result.done:
                    break

            total_rewards.append(episode_reward)

            if (ep + 1) % 10 == 0:
                avg_reward = np.mean(total_rewards[-10:])
                logger.info(
                    "dqn_episode",
                    episode=ep + 1,
                    avg_reward=round(float(avg_reward), 4),
                    epsilon=round(self.epsilon, 4),
                )

        return {
            "mean_reward": float(np.mean(total_rewards)),
            "best_reward": float(np.max(total_rewards)),
            "final_epsilon": self.epsilon,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
            },
            path,
        )
        logger.info("dqn_saved", path=str(path))

    def load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self._device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.policy_net.eval()
        logger.info("dqn_loaded", path=str(path))
