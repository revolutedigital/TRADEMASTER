# Fast161 — Fast151 sub-basket robustness audit

Fast161 auditou se a política operacional Fast151 melhora ao trocar a cesta de
pares, mantendo fixos:

- entrada `d1_donchian_fade_ch100_hold20`;
- risco `stop2_fixed_1r`;
- lado `SHORT`;
- overlay `max_per_pair_1`;
- circuit breaker `dd800_cooldown180d`.

Status:

`FAST161_STRICT_GATE_KEEP_CURRENT_BASKET_NO_SUBBASKET_REPLACEMENT`

## Resposta curta

Não encontramos uma sub-cesta que substitua a cesta atual sem enfraquecer o
gate estatístico.

A cesta completa continua sendo a única que passa o gate estrito:

`AUDUSD+EURUSD+GBPUSD+NZDUSD`

Isso é importante porque uma checagem ingênua, filtrando apenas o CSV final do
circuit breaker original, fazia `AUDUSD+GBPUSD+NZDUSD` parecer melhor. O
Fast161 recalculou corretamente:

1. filtra a sub-cesta desde os trades base;
2. reaplica `max_per_pair_1`;
3. reaplica o circuit breaker causal `dd800_cooldown180d`;
4. só então mede edge, drawdown e swap break-even.

Com esse recálculo, a sub-cesta sem EURUSD melhora alguns números current, mas
falha no pré por amostra curta e dois anos negativos consecutivos.

## Principais cestas

| Cesta | Gate estrito | Pré trades | Pré LCB | Pré DD | Pré falhas | Current trades | Current LCB | Current DD | Current swap BE |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| AUD+EUR+GBP+NZD | `true` | `51` | `+32.2878` | `-900.2085` | — | `86` | `+9.6335` | `-616.6221` | `0.9589` |
| AUD+GBP+NZD | `false` | `39` | `+33.9482` | `-556.1660` | `trade_count_below_minimum`, `two_consecutive_negative_years` | `66` | `+10.0207` | `-585.4722` | `1.0412` |
| EUR+GBP+NZD | `false` | `50` | `+13.3583` | `-1079.3781` | `drawdown_exceeds_limit`, `two_consecutive_negative_years` | `61` | `+2.9574` | `-585.4722` | `0.2854` |
| AUD+EUR+GBP | `false` | `41` | `-3.0820` | `-996.6750` | `trade_count_below_minimum`, `lcb_non_positive`, `two_consecutive_negative_years` | `70` | `+5.2489` | `-542.4954` | `0.5424` |

## Interpretação

Esse audit evita uma falsa otimização.

Tirar EURUSD parece promissor se a gente apenas filtra o resultado final do
breaker original. Mas isso é errado, porque o breaker é dependente da sequência
de trades fechados. Ao remover EURUSD, as pausas e os trades aceitos mudam.

Depois de recalcular corretamente:

- a cesta completa é a única que passa todos os gates estritos;
- `AUDUSD+GBPUSD+NZDUSD` é interessante, mas não substitui a candidata porque:
  - pré tem só `39` trades;
  - pré tem dois anos negativos consecutivos;
  - é uma seleção pós-hoc;
- não devemos trocar a política operacional Fast151 por sub-cesta sem novo
  preregistro e observação prospectiva.

Leitura de decisão:

- manter a cesta atual de 4 pares como candidata primária;
- não perseguir “tirar EURUSD” como melhoria aprovada;
- se quisermos estudar sub-cesta no futuro, ela precisa virar uma hipótese nova,
  com preregistro próprio.

## Segurança

Fast161 é auditoria histórica offline:

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
  `backend/scripts/research/run_fast161_fast151_subbasket_robustness_audit.py`;
- teste:
  `backend/tests/unit/test_fast161_fast151_subbasket_robustness_audit.py`;
- report JSON:
  `backend/data/lab_fast161/fast151-subbasket-robustness-audit/report.json`;
- métricas:
  `backend/data/lab_fast161/fast151-subbasket-robustness-audit/subbasket-robustness-metrics.csv`;
- resumo:
  `backend/data/lab_fast161/fast151-subbasket-robustness-audit/subbasket-robustness-summary.json`.

Hashes:

- report:
  `1cf79997542ca7ef22d5e015a02a28728a07a6bd24a4ca3beb8661749b680c87`;
- runner:
  `04b25c40baabc818207ab096e5591dbb5bc1cb313d589f7854fc86aea4d5a89b`;
- teste:
  `b836470e9a0c460ed2038d544a907e5d2c0cf8ad094915c8e1527cd45b55780a`;
- métricas:
  `20baa79fd7c9e041b9b5ef0b6795a59dc57adb7b76c518e61fa2cf2fd50c1c4b`;
- resumo:
  `ce9ae96045ac8f7ecd73d213f4db15ab46425f8c459bbc5f0666861fdb84f200`.
