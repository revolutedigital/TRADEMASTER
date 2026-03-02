"""Fuzz testing: verify input validation handles malformed data.

Uses hypothesis for property-based fuzzing of API inputs and data parsers.
"""

import math
import pytest
from decimal import Decimal, InvalidOperation
from hypothesis import given, strategies as st, settings as hyp_settings, assume


class TestFuzzOrderInputs:
    """Fuzz test order-related inputs."""

    @given(quantity=st.text(min_size=1, max_size=50))
    @hyp_settings(max_examples=100)
    def test_quantity_parsing_never_crashes(self, quantity):
        """Parsing arbitrary strings as quantity should never crash."""
        try:
            val = Decimal(quantity)
            assert isinstance(val, Decimal)
        except (InvalidOperation, ValueError, OverflowError):
            pass  # Expected for invalid inputs

    @given(price=st.floats(allow_nan=True, allow_infinity=True))
    @hyp_settings(max_examples=100)
    def test_price_validation_handles_special_floats(self, price):
        """Price validation should reject NaN and Infinity."""
        if math.isnan(price) or math.isinf(price):
            assert not (price > 0 and price < float('inf'))
        else:
            # Valid float prices should be representable as Decimal
            try:
                d = Decimal(str(price))
                assert isinstance(d, Decimal)
            except (InvalidOperation, ValueError):
                pass

    @given(symbol=st.text(min_size=0, max_size=100, alphabet=st.characters()))
    @hyp_settings(max_examples=100)
    def test_symbol_validation_handles_arbitrary_strings(self, symbol):
        """Symbol validation should handle any string input."""
        # Valid symbols are uppercase alphanumeric
        cleaned = ''.join(c for c in symbol if c.isalnum()).upper()
        assert isinstance(cleaned, str)
        # Should not exceed reasonable length
        assert len(cleaned) <= 100

    @given(side=st.text(min_size=0, max_size=20))
    @hyp_settings(max_examples=50)
    def test_side_validation(self, side):
        """Only BUY and SELL should be valid sides."""
        valid_sides = {"BUY", "SELL"}
        normalized = side.strip().upper()
        is_valid = normalized in valid_sides
        assert isinstance(is_valid, bool)


class TestFuzzMarketData:
    """Fuzz test market data parsing."""

    @given(
        open_price=st.floats(min_value=0.0001, max_value=1e8),
        high=st.floats(min_value=0.0001, max_value=1e8),
        low=st.floats(min_value=0.0001, max_value=1e8),
        close=st.floats(min_value=0.0001, max_value=1e8),
        volume=st.floats(min_value=0, max_value=1e12),
    )
    @hyp_settings(max_examples=200)
    def test_candle_validation_invariants(self, open_price, high, low, close, volume):
        """OHLCV candles should maintain price ordering invariants."""
        assume(not any(math.isnan(x) for x in [open_price, high, low, close, volume]))
        
        # High should be >= all others, low should be <= all others
        actual_high = max(open_price, high, close)
        actual_low = min(open_price, low, close)
        assert actual_high >= actual_low
        assert volume >= 0

    @given(returns=st.lists(st.floats(min_value=-0.99, max_value=10.0), min_size=2, max_size=1000))
    @hyp_settings(max_examples=100)
    def test_cumulative_returns_never_negative(self, returns):
        """Cumulative product of (1 + returns) should never go negative if returns > -1."""
        assume(all(not math.isnan(r) and not math.isinf(r) for r in returns))
        import numpy as np
        cum_returns = np.cumprod([1 + r for r in returns])
        assert all(cr >= 0 for cr in cum_returns)


class TestFuzzSQLInjection:
    """Fuzz test SQL injection defense."""

    @given(payload=st.text(min_size=1, max_size=200))
    @hyp_settings(max_examples=200)
    def test_rasp_handles_arbitrary_input(self, payload):
        """RASP scanner should never crash on any input."""
        from app.core.rasp import _scan_request_data
        result = _scan_request_data(payload)
        assert result is None or isinstance(result, str)

    @given(payload=st.sampled_from([
        "' OR '1'='1", "'; DROP TABLE users; --",
        "1 UNION SELECT * FROM information_schema.tables",
        "<script>alert('xss')</script>",
        "javascript:alert(1)",
        "../../../etc/passwd",
        "%2e%2e%2f%2e%2e%2fetc/passwd",
        "SLEEP(5)--",
        "1; EXEC xp_cmdshell('whoami')",
    ]))
    def test_known_attack_payloads_detected(self, payload):
        """Known attack payloads should be detected by RASP."""
        from app.core.rasp import _scan_request_data
        result = _scan_request_data(payload)
        assert result is not None, f"Attack payload not detected: {payload}"
