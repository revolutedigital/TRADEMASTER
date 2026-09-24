# Fast137 — FX daily Donchian fade temporal-reverse audit

Fast137 testou uma pergunta mais dura que o Fast133/Fast135: se a família D1
Donchian fade fosse escolhida apenas no histórico pré-2019, ela continuaria
funcionando no período principal 2019–2026?

Status:

`FAST137_FX_DAILY_DONCHIAN_FADE_TEMPORAL_REVERSE_FAILED_CURRENT_AUDIT`

## Protocolo

Universo:

- pares: `AUDUSD`, `EURGBP`, `EURJPY`, `EURUSD`, `GBPJPY`, `GBPUSD`,
  `NZDUSD`, `USDCAD`, `USDCHF`, `USDJPY`;
- sinal: D1 Donchian fade;
- execução: M1 bid/ask;
- risco: ATR(14) diário causal, deslocado um D1 antes do sinal;
- custos: `FUSION_ZERO` e `STRESS`.

Grade:

- canais Donchian: `20`, `55`, `100` dias;
- holds máximos: `5`, `10`, `20` dias;
- stops: `1.0`, `1.5`, `2.0` ATR;
- gestões:
  - `stop_only`;
  - `fixed_1r`;
  - `fixed_2r`;
  - `trail_1r_1r`.

Split temporal:

- seleção: `backend/data/raw/fx_m1_hd_pre`, pré-2019 local
  (`2011-01` a `2018-11`);
- audit: `backend/data/raw/fx_m1_hd`, atual (`2019-01` até
  `2026-09-01`).

Gates:

- seleção precisa de pelo menos `30` trades;
- seleção precisa cobrir pelo menos `3` anos;
- seleção precisa média base > `0` e média stress > `0`;
- seleção precisa pelo menos `50%` dos anos positivos em stress;
- no máximo `100` finalistas auditados;
- audit precisa de pelo menos `12` trades;
- audit precisa cobrir pelo menos `2` anos;
- audit precisa média base/stress positiva;
- audit precisa LCB base e LCB stress positivos com Bonferroni
  (`0.05 / 100 = 0.0005`);
- audit precisa pelo menos `2/3` dos anos positivos em stress.

## Resultado

Números:

- candidatos testados: `1,080`;
- trades simulados no pré-2019: `106,224`;
- candidatos com métricas de seleção: `1,080`;
- finalistas elegíveis no pré-2019: `100`;
- trades simulados no audit current para os finalistas: `10,211`;
- candidatos aprovados no audit: `0`.

Finalistas por par:

| Par | Finalistas pré-2019 |
|---|---:|
| `EURGBP` | `49` |
| `GBPUSD` | `14` |
| `NZDUSD` | `13` |
| `USDCAD` | `2` |
| `USDCHF` | `22` |

Resumo do audit current:

- `85/100` finalistas tiveram média stress positiva em 2019–2026;
- `27/100` passaram o critério de anos positivos (`>= 2/3`);
- `0/100` tiveram LCB stress positivo;
- `0/100` passaram o gate completo.

Melhor finalista por seleção pré-2019:

`USDCHF:d1_donchian_fade_ch100_hold20:stop2_stop_only`

Seleção pré-2019:

- trades: `69`;
- média stress: `+77.0959` pips/trade;
- LCB stress seleção: `+14.8038`;
- anos positivos: `7/8`.

Audit 2019–2026:

- trades: `63`;
- média stress: `+22.7567` pips/trade;
- LCB stress Bonferroni: `-34.7708`;
- anos positivos: `7/8`;
- `pilot_gate=false`.

Melhor candidato por LCB current:

`NZDUSD:d1_donchian_fade_ch100_hold20:stop1.5_fixed_1r`

- média stress current: `+36.1633`;
- LCB stress current: `-2.0150`;
- anos positivos current: `7/8`;
- `pilot_gate=false`.

Observação importante: a candidata exata do Fast134,
`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r`, não entrou entre os
`100` finalistas quando a seleção foi feita apenas no pré-2019.

## Interpretação

Esse resultado não mata completamente o Fast134, mas reduz a confiança de que a
família Donchian fade seja uma anomalia universal e estável. A leitura honesta é:

- existe força em alguns recortes atuais;
- existe média positiva em vários finalistas;
- mas a variância é grande demais;
- quando corrigimos pela busca de `100` finalistas, nenhum candidato tem LCB
  positivo no current;
- trailing stop também não resolveu o problema: os candidatos com
  `trail_1r_1r` continuam falhando no LCB current.

Portanto, Fast137 reforça a tese de dependência de regime. A melhor pista
continua sendo o Fast134 prospectivo, mas ela precisa provar comportamento em
dado novo pós-pré-registro antes de qualquer shadow/Testnet.

## Segurança

Fast137 é pesquisa offline:

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
  `backend/scripts/research/run_fast137_fx_daily_donchian_fade_temporal_reverse.py`;
- teste:
  `backend/tests/unit/test_fast137_fx_daily_donchian_fade_temporal_reverse.py`;
- report JSON:
  `backend/data/lab_fast137/fx-daily-donchian-fade-temporal-reverse/report.json`;
- trades:
  `backend/data/lab_fast137/fx-daily-donchian-fade-temporal-reverse/temporal-reverse-trades.parquet`;
- seleção:
  `backend/data/lab_fast137/fx-daily-donchian-fade-temporal-reverse/pre2019-selection-metrics.csv`;
- finalistas:
  `backend/data/lab_fast137/fx-daily-donchian-fade-temporal-reverse/pre2019-selected-finalists.csv`;
- audit:
  `backend/data/lab_fast137/fx-daily-donchian-fade-temporal-reverse/current-audited-finalists.csv`;
- contagem de sinais:
  `backend/data/lab_fast137/fx-daily-donchian-fade-temporal-reverse/signal-counts.json`;
- qualidade de pares:
  `backend/data/lab_fast137/fx-daily-donchian-fade-temporal-reverse/pair-quality.json`.

Hashes:

- report:
  `7f99d2c512246e4928734100988010c7d82dc352a2fb2965d6906ecbb04f1fc5`;
- trades:
  `2ea0b3b650035529b0601f3f2d0559e25a1a1554a2da2bef98f28ac80b423bdb`;
- seleção:
  `06e623ea27ef7e8b0ec340b6e119d67478c740cee61e8ec24c4415c4a8cde119`;
- finalistas:
  `d3e06af574427dd76c074c6cd1fd894a40f44de4813192d40dbfc0861c3b35c5`;
- audit:
  `4b97d6af50154d17815c2ff529f060e8684f7d7f90d4ac79407c4d34425e043d`;
- contagem de sinais:
  `000d693a2ed91994ba1f6334007a6b7a07b01b130570182a57a8a7c7da9c2a0c`;
- pair quality:
  `fff0967b9f89c578e4bd3a59046ac270c7c1803f577db511db44b458065011c6`.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Continuar observando o Fast134 prospectivo e abrir nova frente de pesquisa com
famílias menos dependentes de regime do que Donchian fade diário.
