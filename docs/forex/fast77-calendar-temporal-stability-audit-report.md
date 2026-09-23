# Fast77 — auditoria temporal dos weak survivors do calendário FX

Resposta objetiva: ainda não temos estratégia.

Fast77 pegou apenas os weak survivors do Fast76 e fez a pergunta que mais
importa antes de qualquer pré-registro: o resultado positivo da auditoria
2024-2026 aparece ano a ano, ou foi carregado por um único regime?

## Safety boundary

Fast77 mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Fonte

Fast77 usa:

- trades Fast13: `backend/data/lab_fast13/calendar-m1-trades.parquet`;
- weak survivors Fast76 de par+política;
- weak survivors Fast76 de par+evento+política.

Hash do parquet Fast13:

`81f8ac1fedfe24b7404c967ce019ec3feaea32ea8616ab45be8e7af85f8dce29`

Janela:

- seleção: eventos antes de `2024-01-01`;
- auditoria: `2024-01-01` até `2026-09-01`;
- mínimo por ano de auditoria: `4` trades.

## Resultado por escopo

### Par + política

- weak survivors de entrada vindos do Fast76: `6`;
- candidatos positivos em todos os anos de auditoria: `0`;
- candidatos confirmados: `0`;
- candidatos com falha no ano mais recente: `6`.

Interpretação: o melhor agregado do Fast76 em `USDJPY` quebrou em 2026. Não é
estratégia.

### Par + evento + política

- weak survivors de entrada vindos do Fast76: `23`;
- candidatos positivos em todos os anos de auditoria: `1`;
- candidatos confirmados: `0`;
- candidatos com falha no ano mais recente: `20`.

O único sobrevivente temporal fraco:

- par: `USDJPY`;
- evento: `FOMC rate decision`;
- política: `drift_thr2_d5_h120_trail_1r_0_75r`;
- seleção: `31` trades, stress médio `+0.031920612903021145` pips;
- seleção por ano positiva em stress: `2/5` anos;
- auditoria: `20` trades, stress médio `+4.4120315000005235` pips;
- stress total na auditoria: `+88.24063000001047` pips;
- lower stress sem correção: `-1.9468284046199846` pips;
- lower stress Bonferroni: `-7.488511433014544` pips.

Auditoria ano a ano:

- 2024: `8` trades, stress médio `+1.9450002500009873` pips, total
  `+15.560002000007898` pips;
- 2025: `8` trades, stress médio `+4.281719000001332` pips, total
  `+34.25375200001066` pips;
- 2026: `4` trades, stress médio `+9.60671899999798` pips, total
  `+38.42687599999192` pips.

Interpretação: existe uma pista real para observar, mas não para operar. A
seleção histórica é quase zero no agregado, negativa em `3/5` anos, e os limites
inferiores continuam negativos.

## Decisão

Status:

`FAST77_CALENDAR_TEMPORAL_STABILITY_WEAK_SURVIVOR_REQUIRES_PREREGISTRATION`

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O que mudou com o Fast77: a família `USDJPY` pós-CPI perdeu força quando olhada
por ano; a única hipótese que sobreviveu ao corte temporal foi `USDJPY` pós-FOMC
com drift e trailing. Isso deve ir para observação prospectiva congelada, não
para execução.

## Arquivos

- script: `backend/scripts/research/run_fast77_calendar_temporal_stability_audit.py`;
- relatório JSON: `backend/data/lab_fast13/fast77-calendar-temporal-stability-audit/report.json`;
- candidatos temporais par+política:
  `backend/data/lab_fast13/fast77-calendar-temporal-stability-audit/pair_policy_temporal_candidates.json`;
- candidatos temporais par+evento+política:
  `backend/data/lab_fast13/fast77-calendar-temporal-stability-audit/pair_event_policy_temporal_candidates.json`.

Hashes:

- report: `8d1d8a157d495bd8016860b18cba185e007a526d5eb98b955b29fb0604a86aee`;
- pair policy temporal candidates:
  `7e766cddd1f5224b7d83dfe58ece98b9edcec643582c23c82497924eacadad9e`;
- pair event policy temporal candidates:
  `d1098155729c3a36bf68761a1b92bb38b10347642ebf0f0cdcf0441be3ee576d`.
