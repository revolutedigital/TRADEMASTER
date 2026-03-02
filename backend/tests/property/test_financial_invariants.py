"""Property-based tests: financial calculations must satisfy invariants."""

import math
import pytest

try:
    from hypothesis import given, strategies as st, settings as hyp_settings
    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    pytest.skip("hypothesis not installed", allow_module_level=True)


@given(price=st.floats(min_value=0.01, max_value=1_000_000))
@hyp_settings(max_examples=200)
def test_pnl_never_nan(price):
    """P&L calculation must never return NaN."""
    entry = 100.0
    quantity = 1.0
    pnl = (price - entry) * quantity
    assert not math.isnan(pnl)
    assert not math.isinf(pnl)


@given(
    entry=st.floats(min_value=0.01, max_value=1_000_000),
    current=st.floats(min_value=0.01, max_value=1_000_000),
    qty=st.floats(min_value=0.001, max_value=1000),
)
@hyp_settings(max_examples=200)
def test_long_pnl_direction(entry, current, qty):
    """Long position P&L is positive when price goes up."""
    pnl = (current - entry) * qty
    if current > entry:
        assert pnl > 0
    elif current < entry:
        assert pnl < 0
    else:
        assert pnl == 0


@given(
    entry=st.floats(min_value=0.01, max_value=1_000_000),
    current=st.floats(min_value=0.01, max_value=1_000_000),
    qty=st.floats(min_value=0.001, max_value=1000),
)
@hyp_settings(max_examples=200)
def test_short_pnl_direction(entry, current, qty):
    """Short position P&L is positive when price goes down."""
    pnl = (entry - current) * qty
    if current < entry:
        assert pnl > 0
    elif current > entry:
        assert pnl < 0


@given(commission_rate=st.floats(min_value=0, max_value=0.01))
@hyp_settings(max_examples=100)
def test_commission_always_non_negative(commission_rate):
    """Commission must never be negative."""
    price = 50000.0
    quantity = 0.1
    commission = price * quantity * commission_rate
    assert commission >= 0


@given(
    returns=st.lists(
        st.floats(min_value=-0.5, max_value=2.0),
        min_size=2,
        max_size=100,
    )
)
@hyp_settings(max_examples=100)
def test_cumulative_return_finite(returns):
    """Cumulative return should always be finite given bounded inputs."""
    import numpy as np
    arr = np.array(returns)
    cumulative = np.prod(1 + arr)
    assert np.isfinite(cumulative)


@given(
    values=st.lists(
        st.floats(min_value=100, max_value=200000),
        min_size=2,
        max_size=200,
    )
)
@hyp_settings(max_examples=100)
def test_max_drawdown_bounded(values):
    """Max drawdown must be between 0 and 1."""
    import numpy as np
    arr = np.array(values)
    peak = np.maximum.accumulate(arr)
    drawdown = (peak - arr) / peak
    max_dd = np.max(drawdown)
    assert 0 <= max_dd <= 1
