# Fast54 — gate de densidade walk-forward

Data: 2026-09-23  
Mercado: BTCUSDT futures, `aggTrades` normalizado  
Modelo base: top-p low-cost micro congelado em Fast43  
Resultado: melhorou muito o baseline, mas falhou como estratégia.

## Hipótese

Fast53 mostrou que o candidato quebrava quando a quantidade de sinais disparava.
A correlação diária entre sinais selecionados e PnL stress foi negativa:

- correlação `selected_signals` vs `stress_total`: -0,6438;
- correlação `trades` vs `stress_total`: -0,6462.

Fast54 testa um gate causal:

> antes de entrar, contar quantos sinais do próprio artefato congelado ocorreram
> em uma janela passada; só operar se essa contagem estiver abaixo de um limite.

O gate é escolhido em walk-forward:

1. usa apenas blocos anteriores para escolher `lookback + limite`;
2. aplica no próximo bloco;
3. repete até Fast53.

## Fonte

Report:

`backend/data/microstructure_v1/reports/latency-adjusted-density-gate-fast54-walkforward-jul07-sep20-100ms/report.json`

Hash:

`848e2015c9275253a4c07e38b66dae8536f4691614d1e6564843e1ab8e75c0e2`

Trades:

`backend/data/microstructure_v1/reports/latency-adjusted-density-gate-fast54-walkforward-jul07-sep20-100ms/density_gate_walkforward_trades.parquet`

Hash:

`9da6bb3d42219f2bd79c7c8f0441b157136ec8c0d09a4bccd4f3d2a1a26b4cbe`

Status:

`DENSITY_GATE_WALKFORWARD_FAILED`

## Resultado agregado

| Métrica | Sem gate nos mesmos folds | Gate walk-forward |
| --- | ---: | ---: |
| Trades | 1.844 | 1.185 |
| Dias com trade | 50 | 50 |
| Média com 1 bps | -0,2508 bps | +0,3016 bps |
| Média com 2 bps | -1,2508 bps | -0,6984 bps |
| Total com 1 bps | -462,4716 bps | +357,4378 bps |
| Total com 2 bps | -2.306,4716 bps | -827,5622 bps |
| Max DD | 994,6806 bps | 329,7008 bps |

## Resultado por fold

| Teste | Gate escolhido | Trades | Stress total | Baseline stress total | Leitura |
| --- | --- | ---: | ---: | ---: | --- |
| Fast46 | 5min ≤ 34 | 73 | -82,3203 | -19,1699 | piorou |
| Fast47 | 5min ≤ 55 | 17 | +85,8490 | +85,8490 | igual |
| Fast48 | 5min ≤ 55 | 16 | +25,2629 | +25,2629 | igual |
| Fast49 | 5min ≤ 55 | 702 | -600,2569 | -935,6237 | melhorou, mas perdeu |
| Fast50 | 24h ≤ 610 | 165 | -78,8500 | -210,4577 | melhorou, mas perdeu |
| Fast51 | 24h ≤ 377 | 112 | -142,7763 | -140,7492 | igual/pior |
| Fast52 | 24h ≤ 89 | 53 | -1,7023 | -394,8531 | quase zerou |
| Fast53 | 15min ≤ 13 | 47 | -32,7683 | -716,7299 | melhorou muito, mas perdeu |

## Decisão

Não temos estratégia aprovada.

O gate walk-forward confirmou a tese de que densidade de sinais é parte do
problema:

- reduziu o drawdown;
- cortou excesso de trades;
- melhorou o total stress em aproximadamente +1.479 bps contra o baseline.

Mas ainda não basta:

- stress agregado continuou negativo;
- apenas alguns folds foram positivos;
- o seletor walk-forward ainda escolheu gates frouxos demais no início da
  quebra de regime.

Sem Testnet/LIVE.

