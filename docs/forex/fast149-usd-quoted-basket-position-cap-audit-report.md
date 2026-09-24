# Fast149 — USD-quoted basket position-cap audit

Fast149 testou controles simples de exposição na cesta Fast147:

> Dá para reduzir drawdown e simultaneidade sem destruir o edge?

Status:

`FAST149_POSITION_CAP_REDUCES_DRAWDOWN_BUT_REQUIRES_CIRCUIT_BREAKER`

## Resposta curta

Sim, um controle simples ajuda bastante:

`max_per_pair_1`

Regra do overlay:

- manter a entrada congelada;
- manter `SHORT`;
- manter `stop2_fixed_1r`;
- **pular novo trade se já houver uma posição aberta no mesmo par**;
- não alterar canal, hold, alvo, stop, custo ou pares.

Esse overlay reduziu o drawdown sem matar o LCB.

## Comparação

| Amostra | Versão | Trades | Média stress | LCB stress | Total stress | Max drawdown | Max concorrência |
|---|---|---:|---:|---:|---:|---:|---:|
| Pré-2019 | sem cap | `89` | `+37.8272` | `+9.7967` | `+3366.6223` | `-2230.5874` | `6` |
| Pré-2019 | `max_per_pair_1` | `58` | `+52.0126` | `+18.1781` | `+3016.7295` | `-1193.7464` | `4` |
| 2019–2026 | sem cap | `143` | `+42.3340` | `+24.6986` | `+6053.7689` | `-1125.7067` | `10` |
| 2019–2026 | `max_per_pair_1` | `86` | `+33.6173` | `+9.6335` | `+2891.0872` | `-616.6221` | `4` |

Trades pulados:

- pré-2019: `31`;
- 2019–2026: `57`.

## Interpretação

Esse overlay é um avanço operacional real:

- reduz drawdown pré-2019 em aproximadamente `46.5%`;
- reduz drawdown 2019–2026 em aproximadamente `45.2%`;
- reduz a simultaneidade atual de `10` para `4`;
- mantém LCB positivo no pré e no atual.

Mas ainda não resolve tudo:

- pré-2019 continua com dois anos negativos consecutivos (`2017`, `2018`);
- portanto ainda precisa de circuit breaker / regra de pausa;
- continua sendo pesquisa histórica, não autorização de execução.

Leitura honesta:

`max_per_pair_1` deve virar parte da candidata operacional. A próxima peça
necessária é testar circuit breaker causal de portfólio, sem mudar entrada.

## Segurança

Fast149 é pesquisa offline:

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
  `backend/scripts/research/run_fast149_usd_quoted_basket_position_cap_audit.py`;
- teste:
  `backend/tests/unit/test_fast149_usd_quoted_basket_position_cap_audit.py`;
- report JSON:
  `backend/data/lab_fast149/usd-quoted-basket-position-cap-audit/report.json`;
- métricas:
  `backend/data/lab_fast149/usd-quoted-basket-position-cap-audit/position-cap-metrics.csv`;
- trades pré selecionados:
  `backend/data/lab_fast149/usd-quoted-basket-position-cap-audit/selected-overlay-pre-trades.csv`;
- trades current selecionados:
  `backend/data/lab_fast149/usd-quoted-basket-position-cap-audit/selected-overlay-current-trades.csv`;
- equity pré:
  `backend/data/lab_fast149/usd-quoted-basket-position-cap-audit/selected-overlay-pre-equity-curve.csv`;
- equity current:
  `backend/data/lab_fast149/usd-quoted-basket-position-cap-audit/selected-overlay-current-equity-curve.csv`;
- resumo:
  `backend/data/lab_fast149/usd-quoted-basket-position-cap-audit/position-cap-summary.json`.

Hashes:

- report:
  `f7891e0c275e2b15484402bcbb7666d6298a39b1d2c7ffe7a8bc7c415339178b`;
- métricas:
  `fb7a9db0e7efce88d5b91f498b2d3a359a0d5c459c7c69f2784f1d2051eb2f7f`;
- trades pré:
  `115e145f31c29031f0256c1cb8767569feec03714bfc250771f50350190bb0a5`;
- trades current:
  `437af9f105fa677ae8caea6ed9088666f9fbc64a14d15459f61716659efd85a0`;
- equity pré:
  `f8c49a44bb8fe5ddbd7228ab7f02dd9529f7ff422a2c61be0ad4fec59dd37d69`;
- equity current:
  `33e87507767c6ad322aa5a618cc8c27d8de07fd78fbe91deb5d7ec10190a3387`;
- resumo:
  `ae766b38437c3664560315d3de100c0250d14c653f0a247392bcb08a23847412`.
