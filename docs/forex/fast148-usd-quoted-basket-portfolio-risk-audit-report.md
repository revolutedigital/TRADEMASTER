# Fast148 — USD-quoted basket portfolio risk audit

Fast148 auditou a candidata Fast147 pelo ângulo de risco de portfólio:

> A cesta é operável quando olhamos drawdown, anos ruins e posições
> simultâneas, ou só parece boa na média?

Status:

`FAST148_USD_QUOTED_BASKET_REQUIRES_DRAWDOWN_AND_EXPOSURE_CONTROLS`

## Resposta curta

A cesta continua estatisticamente interessante, mas **não pode virar shadow/Testnet
sem controles de drawdown e exposição**.

O problema principal não é a média. A média é boa. O problema é o caminho:

- pré-2019 teve drawdown histórico de `-2230.5874` pips;
- pré-2019 teve dois anos ruins consecutivos (`2017`, `2018`);
- 2019–2026 chegou a `10` trades simultâneos na cesta.

Isso não mata a hipótese, mas muda a implementação necessária: qualquer versão
operável precisa de limite de posições simultâneas, risco por trade pequeno,
circuit breaker de drawdown e regra de pausa.

## Política auditada

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only`

Pares:

- `AUDUSD`;
- `EURUSD`;
- `GBPUSD`;
- `NZDUSD`.

Regra:

- entrada: `d1_donchian_fade_ch100_hold20`;
- risco: `stop2_fixed_1r`;
- lado: `SHORT`;
- gestão: `fixed_1r`.

## Resultado principal

| Amostra | Trades | Total stress | Média stress | LCB stress | Max drawdown | Max concorrência | Dois anos negativos? |
|---|---:|---:|---:|---:|---:|---:|---|
| Pré-2019 | `89` | `+3366.6223` | `+37.8272` | `+9.7967` | `-2230.5874` | `6` | sim |
| 2019–2026 | `143` | `+6053.7689` | `+42.3340` | `+24.6986` | `-1125.7067` | `10` | não |

## Anos do pré-2019

| Ano | Trades | Total stress | Média stress |
|---|---:|---:|---:|
| `2011` | `1` | `+229.4603` | `+229.4603` |
| `2012` | `5` | `+568.3625` | `+113.6725` |
| `2013` | `22` | `+583.9196` | `+26.5418` |
| `2014` | `11` | `+851.1768` | `+77.3797` |
| `2015` | `5` | `+1481.3116` | `+296.2623` |
| `2016` | `14` | `+1640.3620` | `+117.1687` |
| `2017` | `24` | `-1313.8204` | `-54.7425` |
| `2018` | `7` | `-674.1500` | `-96.3071` |

## Pares em 2019–2026

| Par | Trades | Total stress | Média stress |
|---|---:|---:|---:|
| `AUDUSD` | `42` | `+2191.9455` | `+52.1892` |
| `EURUSD` | `35` | `+699.1564` | `+19.9759` |
| `GBPUSD` | `37` | `+1599.5029` | `+43.2298` |
| `NZDUSD` | `29` | `+1563.1641` | `+53.9022` |

## Interpretação

Fast146 respondeu “existe edge de cesta?”. Sim.

Fast148 respondeu “dá para operar cru?”. Não.

A candidata precisa ser tratada como sistema de portfólio, não como sinal
isolado. O risco vem de concentração temporal: quando o regime fica ruim, vários
pares USD-quoted podem sofrer juntos.

Controles mínimos antes de qualquer shadow offline:

- cap de posições simultâneas;
- limite por par;
- risco por trade pequeno;
- stop de portfólio por drawdown;
- pausa após sequência/ano ruim;
- relatório separado testando esses controles sem alterar a entrada congelada.

## Segurança

Fast148 é pesquisa offline:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`.

Nenhum trade foi enviado. Nenhum ledger foi alterado. O resultado não autoriza
Demo, Testnet nem LIVE.

## Artefatos

- runner:
  `backend/scripts/research/run_fast148_usd_quoted_basket_portfolio_risk_audit.py`;
- teste:
  `backend/tests/unit/test_fast148_usd_quoted_basket_portfolio_risk_audit.py`;
- report JSON:
  `backend/data/lab_fast148/usd-quoted-basket-portfolio-risk-audit/report.json`;
- trades pré:
  `backend/data/lab_fast148/usd-quoted-basket-portfolio-risk-audit/pre-basket-trades.csv`;
- trades current:
  `backend/data/lab_fast148/usd-quoted-basket-portfolio-risk-audit/current-basket-trades.csv`;
- equity pré:
  `backend/data/lab_fast148/usd-quoted-basket-portfolio-risk-audit/pre-equity-curve.csv`;
- equity current:
  `backend/data/lab_fast148/usd-quoted-basket-portfolio-risk-audit/current-equity-curve.csv`;
- resumo:
  `backend/data/lab_fast148/usd-quoted-basket-portfolio-risk-audit/portfolio-risk-summary.json`.

Hashes:

- report:
  `5cc1cf9eeba6a0191e0465f8089192d585f37965edb5c72405d1190f82383b60`;
- trades pré:
  `e0fb91d6dffc20bc0d425300c2082c80c1ec4572d771361df25b2c5b307da76a`;
- trades current:
  `6e24782e8d5932dd1b31498ac64238c39d9ccf52d664f95afc1ffd531f5fe1e2`;
- equity pré:
  `26a357a26e6b847cf3ce2f717330bed730af65de4ecee6d240a870026466156f`;
- equity current:
  `2d8df4f585c7db51b7b093258b2a1e4c54d64aabe714535e020dec177aea1a2c`;
- resumo:
  `ccbe57ff6d11e3cd47384829c9e756b22d7b6ae561f270df3bba1f025dd671c6`.
