"""Small compiled helpers shared by the strategies: fixed-size rings kept inside the flat state.

A strategy keeps its whole memory in one float64 array so that the lab and the live runner can
snapshot, reset and replay it. A ring occupies `RING_HEADER + capacity` consecutive slots: the
write position, the number of values held, and the values. The values are always at indices
`[0, count)` of the ring's payload, in no particular order, which is all a mean, a standard
deviation or a median needs.
"""

from __future__ import annotations

import numpy as np
from numba import njit

RING_HEADER = 2
FAR_TARGET = 1.0e6  # a profit target that no market move reaches, for strategies that exit by time


def ring_slots(capacity: int) -> int:
    """How many state slots a ring of this capacity needs."""
    return RING_HEADER + capacity


@njit(cache=True)
def ring_push(state, offset, capacity, value):
    head = int(state[offset])
    state[offset + RING_HEADER + head] = value
    state[offset] = (head + 1) % capacity
    if state[offset + 1] < capacity:
        state[offset + 1] += 1.0


@njit(cache=True)
def ring_count(state, offset):
    return int(state[offset + 1])


@njit(cache=True)
def ring_mean(state, offset):
    count = int(state[offset + 1])
    total = 0.0
    for i in range(count):
        total += state[offset + RING_HEADER + i]
    return total / count if count > 0 else 0.0


@njit(cache=True)
def ring_std(state, offset):
    """Population standard deviation of the values held."""
    count = int(state[offset + 1])
    if count == 0:
        return 0.0
    mean = ring_mean(state, offset)
    squares = 0.0
    for i in range(count):
        deviation = state[offset + RING_HEADER + i] - mean
        squares += deviation * deviation
    return np.sqrt(squares / count)


@njit(cache=True)
def ring_median(state, offset):
    count = int(state[offset + 1])
    if count == 0:
        return 0.0
    values = np.sort(state[offset + RING_HEADER : offset + RING_HEADER + count])
    middle = count // 2
    if count % 2 == 1:
        return values[middle]
    return 0.5 * (values[middle - 1] + values[middle])


@njit(cache=True)
def true_range(high, low, previous_close, has_previous):
    """Wilder's true range; the plain range on the first bar, when there is no previous close."""
    result = high - low
    if has_previous:
        result = max(result, abs(high - previous_close), abs(low - previous_close))
    return result
