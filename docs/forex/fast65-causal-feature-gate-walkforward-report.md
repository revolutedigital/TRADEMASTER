# Fast65 — walk-forward de gate causal por feature

Fast64 fechou a porta de sessão/lado/probabilidade. Fast65 testa uma família
mais forte, ainda causal e interpretável:

> escolher um único threshold de feature de microestrutura no treino e aplicar
> esse gate congelado ao bloco holdout.

Resultado: melhorou muito o agregado, mas não passou robustez por bloco.

## Escopo

- fonte: trades congelados Fast43–Fast53;
- features: `research_rows.parquet`, somente `horizon_seconds=300`;
- merge: `decision_time_ms + side + horizon_seconds`;
- filtros permitidos: threshold de uma feature causal pré-entrada;
- filtros proibidos: MFE, MAE, terminal, exit reason, stress futuro, qualquer
  campo pós-entrada;
- validação: leave-one-block-out;
- execução avaliada: one-position;
- custo: `stress_net_bps` já materializado nos trades congelados.

## Safety boundary

O Fast65 mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Arquivos

- relatório JSON: `backend/data/microstructure_v1/reports/fast65-causal-feature-gate-walkforward/report.json`;
- holdouts: `backend/data/microstructure_v1/reports/fast65-causal-feature-gate-walkforward/holdouts.json`;
- dataset mergeado: `backend/data/microstructure_v1/reports/fast65-causal-feature-gate-walkforward/selected_trades_with_causal_features.parquet`.

Hashes:

- report: `dd9b69362804512994c0cab98aaeb21d6a4b1315ad3911052e6588d438eeca16`;
- holdouts: `682e869bc83d4643f29e38c26f5b08a5ab0cf7eb3f7ad19b5e7d62f5e1c7dd09`;
- merged selected trades: `f4979be9a9329d748b4b1bb33f137dc223a051cef3257a45ea83387756e68adb`.

## Resultado

Status:

`FAST65_CAUSAL_FEATURE_GATE_FAILED_WALKFORWARD`

Resumo:

- trades originais avaliados: 2058;
- features causais candidatas: 34;
- trades pós-gate em one-position: 474;
- holdouts positivos em stress: `6/11`;
- stress total: `54.73242289284304` bps;
- expected total: `528.7324228928431` bps;
- pior holdout: `-53.19844210611288` bps.

Por holdout:

| Holdout | Gate congelado | Trades | Stress total | Passou |
| --- | --- | ---: | ---: | --- |
| Fast43 | `quote_volume_30s>=q0.80:102346114.9` | 18 | `27.510157047122725` | sim |
| Fast44 | `quote_volume_30s>=q0.80:101812649.4` | 19 | `31.589235297923235` | sim |
| Fast45 | `quote_volume_30s>=q0.80:101513763` | 8 | `36.19543849255784` | sim |
| Fast46 | `flow_imbalance_60s<=q0.33:-0.21563823` | 26 | `-28.28018733680814` | não |
| Fast47 | `quote_volume_30s>=q0.80:102281544.4` | 2 | `31.70727239686306` | não, amostra |
| Fast48 | `realized_vol_30s_bps>=q0.80:44.71952291` | 8 | `-5.952400383304395` | não |
| Fast49 | `quote_volume_30s>=q0.80:99765883.15` | 199 | `-51.9484281634785` | não |
| Fast50 | `realized_vol_30s_bps>=q0.80:44.04709431` | 57 | `26.77601161030352` | sim |
| Fast51 | `quote_volume_30s>=q0.80:99979658.23` | 37 | `24.362118269641677` | sim |
| Fast52 | `quote_volume_30s>=q0.80:101186722.5` | 40 | `15.9716477681349` | sim |
| Fast53 | `realized_vol_30s_bps>=q0.67:24.1181594` | 60 | `-53.19844210611288` | não |

## Interpretação

Essa é uma pista melhor do que Fast64:

- o agregado saiu de negativo para positivo;
- os gates escolhidos convergem para liquidez/atividade recente
  (`quote_volume_30s`) e volatilidade (`realized_vol_30s_bps`);
- isso sugere que a entrada só faz sentido em regime de microestrutura mais
  ativo.

Mas ainda não é estratégia:

- só `6/11` holdouts passaram;
- Fast49 e Fast53 continuam negativos com amostra relevante;
- gate de uma feature só não capturou a quebra de regime.

## Conclusão

Não há estratégia aprovada. Mas Fast65 mudou a próxima direção: abandonar
filtros genéricos de horário/lado e focar em regime de atividade/liquidez. O
próximo teste correto é pré-registrar uma família menor e mais estável baseada
em `quote_volume_30s`/`realized_vol_30s_bps`, com critério de reprovação por
holdout, antes de qualquer shadow prospective.
