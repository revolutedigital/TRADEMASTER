# Fast78 — runner de observação USDJPY/FOMC

Resposta objetiva: ainda não temos estratégia, e o runner confirma que ainda não
existe evento prospectivo elegível.

Fast78 implementa o pré-registro de observação do único weak survivor temporal
do Fast77. Ele filtra somente:

- par: `USDJPY`;
- evento: `FOMC rate decision`;
- política: `drift_thr2_d5_h120_trail_1r_0_75r`;
- janela: `event_timestamp > 2026-09-23T18:26:11Z`, o timestamp UTC do commit
  de pré-registro `97d7b900a9fbe016dc37a120374f54d8c07c915b`.

## Safety boundary

O runner mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`;
- `testnet_allowed=false`;
- `live_allowed=false`.

## Resultado da execução inicial

Usando o parquet fechado do Fast13:

`backend/data/lab_fast13/calendar-m1-trades.parquet`

Hash:

`81f8ac1fedfe24b7404c967ce019ec3feaea32ea8616ab45be8e7af85f8dce29`

Resultado:

- status: `FAST78_OBSERVATION_NO_ELIGIBLE_EVENTS_YET`;
- eventos elegíveis: `0`;
- observações gravadas: `0`;
- ordens enviadas: `0`;
- gate de falha precoce: não acionado;
- gate mínimo de shadow offline: não passou;
- Testnet: não permitido;
- LIVE: não permitido.

Isso é o comportamento esperado: o dado atual termina antes do pré-registro.

## Gates implementados

### Falha precoce

A hipótese falha se:

- após os primeiros `8` eventos prospectivos, `total_stress_pips <= -20`; ou
- houver dois anos-calendário prospectivos consecutivos com stress médio
  negativo.

### Mínimo para candidato de shadow offline

Mesmo que pareça bom, só vira candidato para revisão de shadow offline se:

- houver pelo menos `20` eventos prospectivos;
- stress médio > `+2.0` pips/trade;
- base médio > `0`;
- lower confidence bound unilateral de stress, sem correção, > `0`;
- pelo menos `60%` dos anos prospectivos tiverem stress médio > `0`;
- nenhuma regra tiver sido alterada.

Passar esse gate ainda não libera Testnet.

## Arquivos

- runner: `backend/scripts/research/run_fast78_usdjpy_fomc_observation.py`;
- teste: `backend/tests/unit/test_fast78_usdjpy_fomc_observation.py`;
- relatório JSON:
  `backend/data/lab_fast13/fast78-usdjpy-fomc-observation/report.json`;
- observações JSONL:
  `backend/data/lab_fast13/fast78-usdjpy-fomc-observation/observations.jsonl`.

Hashes:

- report: `1a8ca043b3780d25c3c50a0949bbf4ff887bc75171c5efaeebd0def511ea90df`;
- observations: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Decisão

Status:

`FAST78_OBSERVATION_NO_ELIGIBLE_EVENTS_YET`

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O sistema agora está preparado para avaliar a hipótese congelada quando houver
dados prospectivos pós-pré-registro, sem recalibrar e sem executar ordens.
