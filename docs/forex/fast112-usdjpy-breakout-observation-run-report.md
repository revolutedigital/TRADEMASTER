# Fast112 — runner de observação USDJPY London breakout

Resposta objetiva: ainda não temos estratégia, e o runner confirma que ainda não
existe trade prospectivo elegível.

Fast112 implementa o pré-registro da melhor pista do Fast111:

- par: `USDJPY`;
- configuração: `F1a.rr2.5.w0.6-2.0.se11:00.x16:30`;
- filtro: `london_entry_hour=8 && stop_pips>94.88491666666664`;
- janela: `entry_time > 2026-09-24T02:59:14Z`, timestamp UTC do commit de
  pré-registro `e9c77709d66136a82ec875acdb63872c558ea102`.

## Safety boundary

O runner mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`;
- sem Demo;
- sem Testnet;
- sem LIVE.

## Resultado da execução inicial

Fonte atual:

`backend/data/lab_fast111/usdjpy-breakout-regime-scout/trade_rows.parquet`

Hash:

`4093661a12ce29ddd84c50b5ab24c7cd8f78bcf35127966edfbf1c11e256380f`

Resultado:

- status: `FAST112_OBSERVATION_NO_ELIGIBLE_TRADES_YET`;
- observações elegíveis: `0`;
- ordens enviadas: `0`;
- gate de falha precoce: não acionado;
- gate mínimo de shadow offline: não passou;
- Testnet: não permitido;
- LIVE: não permitido.

Isso é esperado: o dado materializado no Fast111 termina antes do
pré-registro Fast112.

## Gates implementados

### Falha precoce

A hipótese falha se:

- após os primeiros `12` trades prospectivos, `total_stress_r <= -3.0`; ou
- houver dois anos-calendário prospectivos consecutivos com stress médio
  negativo.

### Mínimo para candidato de shadow offline

Mesmo que pareça bom, só vira candidato para revisão de shadow offline se:

- houver pelo menos `50` trades prospectivos;
- stress médio `> +0.05 R/trade`;
- base médio `> 0`;
- lower confidence bound unilateral de stress, sem correção, `> 0`;
- pelo menos `60%` dos anos prospectivos tiverem stress médio `> 0`;
- nenhuma regra tiver sido alterada.

Passar esse gate ainda não libera Testnet.

## Arquivos

- runner:
  `backend/scripts/research/run_fast112_usdjpy_breakout_observation.py`;
- teste:
  `backend/tests/unit/test_fast112_usdjpy_breakout_observation.py`;
- relatório JSON:
  `backend/data/lab_fast112/usdjpy-breakout-observation/report.json`;
- observações JSONL:
  `backend/data/lab_fast112/usdjpy-breakout-observation/observations.jsonl`.

Hashes:

- report:
  `59e61ac776ff4d3c2b49efa221ff6529bc858b84a5e1d77a4c7f03acba22cec1`;
- observations:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- preregistration:
  `c9bcf46b615ac2572efd806c44a342b9976fc60613706a97f2d238a2fd5fd0f4`.

## Decisão

Status:

`FAST112_OBSERVATION_NO_ELIGIBLE_TRADES_YET`

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O sistema agora está preparado para avaliar a hipótese congelada quando houver
dados prospectivos pós-pré-registro, sem recalibrar e sem executar ordens.
