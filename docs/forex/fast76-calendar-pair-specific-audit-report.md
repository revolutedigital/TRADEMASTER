# Fast76 — auditoria calendário FX por par específico

Resposta objetiva: ainda não temos estratégia.

Fast13 rejeitou a família de eventos macro oficiais agregando os 10 pares FX.
Fast76 reabre uma pergunta menor: será que havia edge escondido em um par
específico, ou em par + evento específico, que a agregação diluiu?

## Safety boundary

Fast76 mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Fonte

Fast76 usa somente o artefato já fechado do Fast13:

`backend/data/lab_fast13/calendar-m1-trades.parquet`

Hash:

`81f8ac1fedfe24b7404c967ce019ec3feaea32ea8616ab45be8e7af85f8dce29`

Split:

- seleção: eventos antes de `2024-01-01`;
- auditoria: `2024-01-01` até `2026-09-01`.

## Escopos auditados

### 1. Par + política

- grupos elegíveis: `972`;
- candidatos positivos na seleção: `79`;
- weak survivors com média positiva na auditoria: `6`;
- candidatos aprovados por limite inferior corrigido: `0`;
- Bonferroni alpha: `0.0006329113924050633`.

O melhor weak survivor:

- par: `USDJPY`;
- política: `drift_thr2_d10_h120_trail_1r_0_75r`;
- seleção: `127` trades, stress médio `+1.3299965590552871` pips;
- auditoria: `77` trades, stress médio `+1.1057449740261995` pips;
- stress total na auditoria: `+85.14236300001737` pips;
- lower stress sem correção: `-5.451433465622784` pips;
- lower stress Bonferroni: `-12.08085060573451` pips.

Interpretação: média positiva, incerteza enorme. Não aprova.

### 2. Par + evento + política

- grupos elegíveis: `2916`;
- candidatos positivos na seleção: `314`;
- weak survivors com média positiva na auditoria: `23`;
- candidatos aprovados por limite inferior corrigido: `0`;
- Bonferroni alpha: `0.00015923566878980894`.

O melhor weak survivor:

- par: `USDJPY`;
- evento: `CPI US`;
- política: `drift_thr2_d10_h30_trail_1r_0_75r`;
- seleção: `45` trades, stress médio `+3.081302333333328` pips;
- auditoria: `28` trades, stress médio `+5.679933285715188` pips;
- stress total na auditoria: `+159.03813200002526` pips;
- lower stress sem correção: `-6.247993341226963` pips;
- lower stress Bonferroni: `-23.20065963439446` pips.

Interpretação: USDJPY depois de CPI/FOMC/NFP merece atenção como pista, mas a
amostra é pequena e a dispersão é grande demais para virar estratégia.

## Arquivos

- script: `backend/scripts/research/run_fast76_calendar_pair_specific_audit.py`;
- relatório JSON: `backend/data/lab_fast13/fast76-calendar-pair-specific-audit/report.json`;
- candidatos par+política: `backend/data/lab_fast13/fast76-calendar-pair-specific-audit/pair_policy_candidates.json`;
- candidatos par+evento+política: `backend/data/lab_fast13/fast76-calendar-pair-specific-audit/pair_event_policy_candidates.json`.

Hashes:

- report: `89a9c63c48ed249a4c811dba05679efbbc1dc182751e0b105832731ecd3a99ca`;
- pair policy: `3f296d0827c219e14dfec23ea1573005e02a1c2aecf54f0a84b11cb5769add61`;
- pair event policy: `730c514b9cd4d96de590f7a3069f76dee4d5bf4070223f3dc3e4ca68053731a7`.

## Decisão

Status:

`FAST76_CALENDAR_PAIR_SPECIFIC_WEAK_SURVIVORS_FAILED_SIGNIFICANCE`

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O aprendizado útil: a primeira pista não completamente morta apareceu em
`USDJPY` pós-eventos macro oficiais, principalmente `CPI US` com drift e
trailing. Mas isso ainda é só hipótese para coleta prospectiva/pré-registro,
porque nenhum candidato teve limite inferior positivo.
