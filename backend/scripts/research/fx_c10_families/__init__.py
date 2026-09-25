"""Day-trade signal families of round C10 (docs/forex/c10-preregistration.md). Each module exposes `PARAMS`, the fixed
parameter sets (each names its bar `timeframe`), and `signals(ctx, params) -> (side, stop_ref)`: side in {-1, 0, +1}
decided at the close of each bar with only what is known then, and a structural stop level or NaN (the family then
trades only with the grid's stops in daily ATRs). Exits always come from the management grid of fx_c10_core."""

from scripts.research.fx_c10_families import (
    asian, bollinger, burst, donchian, gap, momentum, nr, orb, pdh, rsi2, soup, williams,
)

FAMILIES = {module.__name__.rsplit(".", 1)[-1]: module for module in
            (orb, momentum, rsi2, bollinger, donchian, williams, gap, nr, burst, asian, pdh, soup)}
