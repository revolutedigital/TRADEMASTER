# Fast56 — holdout novo do gate `5min≤0`

Data: 2026-09-23  
Mercado: BTCUSDT futures, Binance public `aggTrades`  
Holdout novo: 2026-09-21 a 2026-09-22  
Resultado: passou no stress, mas ainda é curto demais para aprovar.

## Por que esta rodada existe

Fast55 encontrou o melhor diagnóstico até agora:

`top-p 1% + micro_8_3 + gate 5min≤0`

O problema: esse gate foi escolhido depois de ver o holdout estendido. Fast56
baixa e materializa dias novos que não participaram da escolha do gate.

## Dados novos

Fonte pública Binance Vision, sem credenciais.

| Data | Rows | Gaps | Status | Source SHA-256 | Normalized SHA-256 |
| --- | ---: | ---: | --- | --- | --- |
| 2026-09-21 | 2.495.145 | 0 | VALID | `4ec6dc278b293522f4f7278781d8a1534ac295b1a3e424a0e8fd4f685b1821e8` | `22492d80844f9cff4b5bd0870310f9e7c6e07e49e73d09e7f117e406f7e6046a` |
| 2026-09-22 | 1.606.942 | 0 | VALID | `d024d1955d2385eec53e0b8cec22192132d902ad8046dc21e17f68f06ec3a6a0` | `daad553e4433abbfe580bb765f978b99293e263a2cce0c9f8e8e8b616cbf5c83` |

## Candidato testado

- modelo: artefato congelado Fast43;
- model hash: `9878346a6d225cbc29a5cd17c70e039ebeea0e4831e93111243eded42122ba68`;
- target: `latency_paid_stress_before_stop`;
- horizonte: 300 s;
- features: `flow_price_session`;
- top-p: 1%;
- threshold: `0,90195311025804`;
- trailing: `micro_8_3`;
- gate: só entrar se houve `0` sinais nos últimos `300s`;
- custo esperado/stress: 1/2 bps;
- latência: 100 ms.

## Resultado

### Sem gate

Fonte:

`backend/data/microstructure_v1/reports/latency-adjusted-frozen-top-p-fast56-new-holdout-sep21-sep22-100ms/report.json`

Hash:

`3c4052e3f12c74e98965a3d040a70fa3e0584c1be579ec29abca5356179e1e0d`

| Métrica | Valor |
| --- | ---: |
| Sinais selecionados | 538 |
| Trades | 121 |
| Média 1 bps | +0,8962 |
| Média 2 bps | -0,1038 |
| Total 2 bps | -12,5609 |
| Max DD | 123,5421 |

### Com gate `5min≤0`

Fonte:

`backend/data/microstructure_v1/reports/latency-adjusted-fixed-density-gate-fast56-new-holdout-sep21-sep22-100ms/report.json`

Hash:

`43b4e7e178eb513438ed3aa2030b8e5c9d1e420acc6e24841e9ef58ad944c7fc`

Trades:

`backend/data/microstructure_v1/reports/latency-adjusted-fixed-density-gate-fast56-new-holdout-sep21-sep22-100ms/fixed_density_gate_trades.parquet`

Hash:

`44435be1cb2b9c556c1aa524313b033a4bb68d32a352f042ac0eab073909b04f`

| Métrica | Valor |
| --- | ---: |
| Sinais antes do gate | 538 |
| Sinais depois do gate | 28 |
| Trades | 28 |
| Dias | 2 |
| Média 1 bps | +3,4290 |
| Média 2 bps | +2,4290 |
| Total 2 bps | +68,0117 |
| Win rate 1 bps | 71,43% |
| Max DD | 31,2116 |

## Decisão

Temos um candidato mais forte:

`top-p 1% + micro_8_3 + gate 5min≤0`

Ele passou em dois dias novos que não entraram na escolha do gate e melhorou
claramente o mesmo candidato sem gate.

Ainda não temos estratégia aprovada:

- só são 2 dias novos;
- só 28 trades;
- ainda é `aggTrades`, não execução top-of-book/fila;
- custo 1–2 bps continua premissa;
- falta repetir em novos dias prospectivos e/ou book replay.

Próximo gate:

1. continuar baixando novos dias conforme ficam disponíveis;
2. acumular pelo menos 10 dias novos e 100+ trades com esse gate congelado;
3. exigir stress positivo e consistência diária;
4. depois repetir com book/quote ou shadow prospectivo.

Sem Testnet/LIVE.

