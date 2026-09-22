# Microstructure v1 — economic edge budget

Source: verified Binance Spot BTCUSDT 1-second bars, 2026-09-13 through 2026-09-20

> The oracle selects direction after observing the path. It is only an upper-bound feasibility diagnostic and cannot approve a signal or authorize execution.

| Horizon | Paths | Expected cost | Stress cost | MFE p50 | MFE p90 | MFE p99 | Clears expected | Clears stress | Stress p99 net | Feasible |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 5s | 138,239 | 12.00 bps | 24.00 bps | 0.00 | 1.91 | 5.85 | 0.1% | 0.0% | -18.15 bps | NO |
| 15s | 138,237 | 12.00 bps | 24.00 bps | 0.79 | 3.94 | 10.15 | 0.6% | 0.1% | -13.85 bps | NO |
| 30s | 138,234 | 12.00 bps | 24.00 bps | 1.71 | 5.95 | 14.39 | 1.7% | 0.2% | -9.61 bps | NO |
| 120s | 138,216 | 12.00 bps | 24.00 bps | 4.78 | 12.64 | 30.78 | 11.2% | 2.0% | 6.78 bps | YES |
| 300s | 138,180 | 12.00 bps | 24.00 bps | 8.28 | 21.09 | 50.35 | 29.9% | 7.3% | 26.35 bps | YES |

Passing this table only means that enough price movement exists in hindsight. It does not demonstrate that any causal signal can choose the direction or capture it.
