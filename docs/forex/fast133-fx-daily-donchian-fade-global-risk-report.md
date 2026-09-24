# Fast133 — FX daily Donchian fade global risk audit

Fast133 ampliou o Fast130. Em vez de auditar stop/target só no único finalista
do Fast129, ele testou gestão de risco em toda a família D1 Donchian fade:
todos os pares, canais, holds e políticas de risco.

Status:

`FAST133_FX_DAILY_DONCHIAN_FADE_GLOBAL_RISK_RESEARCH_CANDIDATE_REQUIRES_PREREGISTRATION`

## Protocolo

Universo:

- pares: `AUDUSD`, `EURGBP`, `EURJPY`, `EURUSD`, `GBPJPY`, `GBPUSD`,
  `NZDUSD`, `USDCAD`, `USDCHF`, `USDJPY`;
- dados: M1 local `backend/data/raw/fx_m1_hd`;
- sinal: D1 Donchian fade;
- execução: M1 bid/ask;
- risco: ATR(14) diário causal, deslocado um D1 antes do sinal.

Grade:

- canais Donchian: `20`, `55`, `100` dias;
- holds máximos: `5`, `10`, `20` dias;
- stops: `1.0`, `1.5`, `2.0` ATR;
- gestões:
  - `stop_only`;
  - `fixed_1r`;
  - `fixed_2r`;
  - `trail_1r_1r`.

Split:

- seleção: antes de `2024-01-01`;
- audit: `2024-01-01` até `2026-09-01`.

Gates:

- mínimo seleção: `30` trades;
- seleção precisa média base > `0` e média stress > `0`;
- no máximo `100` finalistas auditados;
- mínimo audit: `12` trades;
- mínimo audit years: `2`;
- audit precisa média base/stress positiva;
- LCB stress precisa ser positivo com Bonferroni;
- anos positivos em stress precisam ser pelo menos `2/3`.

## Resultado

Números:

- sinais reconstruídos: `13,659`;
- candidatos: `1,080`;
- trades simulados: `99,768`;
- finalistas elegíveis na seleção: `100`;
- candidatos aprovados no audit: `1`.

Único aprovado:

`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r`

Seleção:

- trades: `44`;
- média base: `+38.9073` pips/trade;
- média stress: `+34.5342` pips/trade;
- LCB stress seleção: `+1.3261`;
- anos positivos: `4/5`;
- risco médio: `132.9373` pips.

Audit:

- trades: `21`;
- média base: `+68.6031` pips/trade;
- média stress: `+67.7507` pips/trade;
- LCB stress Bonferroni: `+20.0959`;
- anos positivos: `3/3`;
- `pilot_gate=true`.

## O que mudou em relação ao Fast130

Fast130 tinha escolhido como preferencial por ranking:

`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_2r`

No Fast133, com a busca global e Bonferroni mais duro (`0.05/100 = 0.0005`),
essa variante não passou o LCB:

- `fixed_2r`: audit stress `+58.9756`, mas LCB `-3.0684`;
- `fixed_1r`: audit stress `+67.7507`, LCB `+20.0959`.

Conclusão: o melhor congelamento agora é `fixed_1r`, não `fixed_2r`. Isso não
invalida o Fast131; ele continua como observação de uma variante. Mas a próxima
observação prospectiva deve congelar `fixed_1r`.

## Interpretação

Este é o melhor resultado da linha FX até agora:

- sobreviveu seleção;
- sobreviveu audit independente 2024–2026;
- passou com hard stop;
- passou em custo stress;
- passou com Bonferroni sobre `100` finalistas;
- foi positivo em `3/3` anos no audit.

Ainda assim, não é estratégia aprovada porque foi descoberto por busca
histórica. Precisa de pré-registro prospectivo e dado novo posterior ao
pré-registro.

## Segurança

Fast133 é pesquisa offline:

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
  `backend/scripts/research/run_fast133_fx_daily_donchian_fade_global_risk.py`;
- teste:
  `backend/tests/unit/test_fast133_fx_daily_donchian_fade_global_risk.py`;
- report JSON:
  `backend/data/lab_fast133/fx-daily-donchian-fade-global-risk/report.json`;
- trades:
  `backend/data/lab_fast133/fx-daily-donchian-fade-global-risk/daily-donchian-fade-global-risk-trades.parquet`;
- contagem de sinais:
  `backend/data/lab_fast133/fx-daily-donchian-fade-global-risk/signal-counts.json`;
- seleção:
  `backend/data/lab_fast133/fx-daily-donchian-fade-global-risk/selection-metrics.csv`;
- audit:
  `backend/data/lab_fast133/fx-daily-donchian-fade-global-risk/audited-finalists.csv`;
- qualidade de pares:
  `backend/data/lab_fast133/fx-daily-donchian-fade-global-risk/pair-quality.json`.

Hashes:

- report:
  `9d482ba974a6061f26a8b1a1e6f5dc702d93094f7424496af88871e7f18ca6fd`;
- trades:
  `13bad134cd1da482eb69e4ad081daa255f5e0d2a685b94589a0fa1e5750d41da`;
- contagem de sinais:
  `59b55e287b175263504ed9060cd9c81e7562b075973aa44318986f3240ead9b5`;
- seleção:
  `dea15246a4f1d0a6c365cd7730ddfae133c981f592817e72ce4fbca90458f2b8`;
- audit:
  `a53f99e81ecdce2eba6cc438b3fb293c75381e234a06244ece3f36e2544c5d2d`;
- pair quality:
  `1dc11165fc3ea86d9e930dc51b14ec02df56075b62e7e5e973059c34582dd119`.

## Próximo passo

Pré-registrar observação prospectiva de:

`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r`

Até haver dados prospectivos pós-pré-registro, o status correto continua sendo:
pesquisa candidata, não estratégia aprovada.
