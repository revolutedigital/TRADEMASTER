# Fast80 — smoke de captura causal

Resposta objetiva: ainda não temos estratégia, mas o pipeline futuro do Fast80
agora está tecnicamente scoreável.

Este smoke não usa dado antigo como prospectivo. Ele apenas reexecuta Fast60 e
Fast61 em output separado para verificar se as decisões e settlements passam a
carregar as features exigidas pelo pré-registro Fast80:

- `realized_vol_30s_bps`;
- `flow_imbalance_5s`.

## Safety boundary

O smoke mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`;
- `testnet_allowed=false`;
- `live_allowed=false`.

## Resultado

### Fast60 smoke

Output:

`backend/data/microstructure_v1/reports/fast80-causal-feature-capture-smoke/fast60/report.json`

Resumo:

- raw top-p: `550`;
- gated signals: `11`;
- runs com gated signals: `8`;
- status: `FAST60_GATED_SHADOW_SIGNALS_RESEARCH_ONLY`;
- decisões gated agora incluem `causal_feature_values`,
  `realized_vol_30s_bps` e `flow_imbalance_5s`.

### Fast61 smoke

Output:

`backend/data/microstructure_v1/reports/fast80-causal-feature-capture-smoke/fast61/report.json`

Resumo:

- sinais: `11`;
- outcomes completos: `8`;
- outcomes incompletos: `3`;
- one-position outcomes: `8`;
- stress total one-position: `-40.62937570991199` bps;
- status: `FAST61_SHADOW_SETTLEMENT_FAILED`.

Esse resultado continua negativo, como no Fast61 original. O objetivo aqui não
era mudar PnL, era verificar se o settlement preserva as features causais.

### Fast80 smoke

Input:

`backend/data/microstructure_v1/reports/fast80-causal-feature-capture-smoke/fast61/one_position_settlements.jsonl`

Resultado:

- status: `FAST80_OBSERVATION_NO_PROSPECTIVE_RECORDS_YET`;
- input records: `8`;
- `scoreable=true`;
- missing `realized_vol_30s_bps`: `0/8`;
- missing `flow_imbalance_5s`: `0/8`;
- registros pós-pré-registro: `0`;
- observações elegíveis: `0`.

Isso é o comportamento correto: o dado é anterior ao pré-registro Fast80, mas o
formato agora é scoreável.

## Arquivos e hashes

- Fast60 report:
  `b95f74fb0daab7a5ff96b47b1fdd4f3cf4a7d38cecaedd213e5f32945e29d813`;
- Fast60 gated decisions:
  `924b0af970721290ba40b8db4f915a00edbfa8a40b1b82ccc4d46be0589e9c46`;
- Fast61 report:
  `5a732c0afe47ea121ab6f02e084141c467b731d65a92e38f6699123791ffb0a5`;
- Fast61 one-position settlements:
  `a8ad4737f9ba6b640f5ebff690f211ecbfe9c7067d65f741d94ca7efd1aed41e`;
- Fast80 smoke report:
  `e6d1f25e5e25bc630ba29ca46c785f1d02e0c83b348f00e4551edeba77c7ad0b`.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O avanço é técnico: a próxima captura prospectiva pode ser avaliada pelo Fast80
sem proxy de feature e sem recalibrar a regra.
