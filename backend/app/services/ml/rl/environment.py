"""Trading environment for reinforcement learning agents."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TradingState:
    """Observation for the RL agent."""

    features: np.ndarray  # Market features (normalized)
    position: int  # 0=flat, 1=long, -1=short
    unrealized_pnl: float
    equity_ratio: float  # current_equity / initial_equity
    drawdown: float


@dataclass
class StepResult:
    """Result of a single environment step."""

    state: TradingState
    reward: float
    done: bool
    info: dict = field(default_factory=dict)


class TradingEnvironment:
    """Simplified trading environment for DQN strategy selection.

    Actions:
        0: Hold (do nothing)
        1: Buy / Go Long
        2: Sell / Go Short
        3: Close position

    State: feature vector + position info
    Reward: risk-adjusted PnL change
    """

    N_ACTIONS = 4

    def __init__(
        self,
        prices: np.ndarray,
        features: np.ndarray,
        initial_capital: float = 10000,
        commission: float = 0.001,
        max_position_pct: float = 0.3,
    ):
        self.prices = prices
        self.features = features
        self.initial_capital = initial_capital
        self.commission = commission
        self.max_position_pct = max_position_pct

        self._step = 0
        self._equity = initial_capital
        self._peak_equity = initial_capital
        self._position = 0  # 0=flat, 1=long, -1=short
        self._entry_price = 0.0
        self._position_size = 0.0

    @property
    def state_dim(self) -> int:
        return self.features.shape[1] + 4  # features + position info

    def reset(self) -> TradingState:
        """Reset environment to initial state."""
        self._step = 0
        self._equity = self.initial_capital
        self._peak_equity = self.initial_capital
        self._position = 0
        self._entry_price = 0.0
        self._position_size = 0.0
        return self._get_state()

    def step(self, action: int) -> StepResult:
        """Execute one step."""
        prev_equity = self._equity
        price = self.prices[self._step]

        # Execute action
        if action == 1 and self._position == 0:
            # Open long
            self._position = 1
            self._entry_price = price
            self._position_size = (
                self._equity * self.max_position_pct / price
            )
            self._equity -= self._position_size * price * self.commission

        elif action == 2 and self._position == 0:
            # Open short
            self._position = -1
            self._entry_price = price
            self._position_size = (
                self._equity * self.max_position_pct / price
            )
            self._equity -= self._position_size * price * self.commission

        elif action == 3 and self._position != 0:
            # Close position
            pnl = self._position * (price - self._entry_price) * self._position_size
            self._equity += pnl - abs(self._position_size * price * self.commission)
            self._position = 0
            self._position_size = 0.0

        # Update unrealized P&L for open positions
        unrealized = 0.0
        if self._position != 0:
            unrealized = (
                self._position
                * (price - self._entry_price)
                * self._position_size
            )

        self._peak_equity = max(self._peak_equity, self._equity + unrealized)
        self._step += 1

        done = self._step >= len(self.prices) - 1

        # Reward: risk-adjusted equity change
        equity_change = (self._equity + unrealized - prev_equity) / self.initial_capital
        drawdown = (
            (self._peak_equity - self._equity - unrealized) / self._peak_equity
            if self._peak_equity > 0
            else 0
        )
        drawdown_penalty = -drawdown * 0.5 if drawdown > 0.02 else 0
        reward = equity_change + drawdown_penalty

        state = self._get_state()
        return StepResult(
            state=state,
            reward=reward,
            done=done,
            info={"equity": self._equity, "unrealized": unrealized},
        )

    def _get_state(self) -> TradingState:
        features = self.features[min(self._step, len(self.features) - 1)]
        unrealized = 0.0
        if self._position != 0 and self._step < len(self.prices):
            price = self.prices[self._step]
            unrealized = (
                self._position
                * (price - self._entry_price)
                * self._position_size
            )

        return TradingState(
            features=features,
            position=self._position,
            unrealized_pnl=unrealized / self.initial_capital,
            equity_ratio=self._equity / self.initial_capital,
            drawdown=(
                (self._peak_equity - self._equity) / self._peak_equity
                if self._peak_equity > 0
                else 0
            ),
        )
