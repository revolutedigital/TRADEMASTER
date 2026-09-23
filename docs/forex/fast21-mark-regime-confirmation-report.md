# Rodada 21 — terceira janela e troca de candidata principal para mark/index basis

Data: 2026-09-23  
Mercado: BTCUSDT futures, WAL prospectivo Fast21 de 30 minutos  
Resultado: regra `spot_perp_basis` rejeitada; família `mark_index_basis` sobreviveu e virou candidata principal

## Pergunta

A rodada 20 deu a primeira candidata replicada:

`SELL quando directed_spot_perp_basis_bps está alto`, horizonte 120s, alvo bruto 5 bps.

Mas o edge bruto era estreito: +0,688 bps antes de custo em Fast20. A pergunta desta rodada foi:

1. essa regra sobrevive numa terceira janela?
2. a candidata independente de Fast20, baseada em `directed_mark_index_basis_bps`, confirma em Fast21?
3. existe um sinal de regime que explique por que uma regra funciona em uma janela e falha na outra?

## Captura Fast21

Comando:

`PYTHONPATH=. .venv/bin/python scripts/research/record_microstructure.py --root data/microstructure_v1/prospective-wal-fast21-30m --symbol BTCUSDT --duration-seconds 1800 --include-spot-trades --audit-status-path data/microstructure_v1/prospective-wal-fast21-30m-audits/evidence-gate-status.json --audit-rolling-days 1 --audit-interval-seconds 120`

Streams capturados:

| Stream | Linhas | Início UTC | Fim UTC |
|---|---:|---|---|
| `trade` | 118.571 | 2026-09-23T03:58:09.924Z | 2026-09-23T04:28:08.384Z |
| `depth` | 17.634 | 2026-09-23T03:58:09.849Z | 2026-09-23T04:28:08.325Z |
| `mark_price` | 1.375 | 2026-09-23T03:58:08Z | 2026-09-23T04:28:08Z |
| `spot_trade` | 71.666 | 2026-09-23T03:58:09.639Z | 2026-09-23T04:28:08.387Z |

Scout WAL:

| Campo | Valor |
|---|---:|
| Status | `READY_FOR_SCOUT_MODELING` |
| Decisões | 330 |
| Linhas | 1.980 |
| Book disponível | 100% |
| Idade média do book | 52,35 ms |
| Máx. idade do book | 103 ms |

## Contexto de regime

Fast21 foi bem diferente de Fast20. No horizonte 120s:

| Lado | Terminal médio | MFE médio | MAE médio | Target 5 bps |
|---|---:|---:|---:|---:|
| BUY | +3,911 bps | +8,024 bps | -3,136 bps | 23,94% |
| SELL | -3,911 bps | +3,136 bps | -8,024 bps | 3,33% |

Ou seja: Fast21 foi um regime comprador. Isso é exatamente o tipo de mudança que precisava matar
uma regra vendida ruim.

## Teste cego da regra `spot_perp_basis` de Fast16/Fast20

Regra congelada:

| Campo | Valor |
|---|---:|
| Feature | `directed_spot_perp_basis_bps` |
| Direção | high |
| Quantil | 0,80 |
| Threshold | 5,203 bps |
| Lado selecionado | SELL |
| Horizonte | 120s |
| Alvo | 5 bps |

Resultado em Fast21:

| Campo | Valor |
|---|---:|
| Seleções | 42 |
| Target rate base | 39,09% |
| Target rate selecionado | 7,14% |
| Payoff bruto base | -0,809 bps |
| Payoff bruto selecionado | -2,284 bps |
| Lift bruto | -1,475 bps |

Decisão: rejeitada. A regra replicada em Fast20 não sobreviveu ao regime comprador de Fast21.

## Teste cego da regra `mark_index_basis` descoberta em Fast20

Regra congelada em Fast20:

| Campo | Valor |
|---|---:|
| Feature | `directed_mark_index_basis_bps` |
| Direção | high |
| Quantil | 0,90 |
| Threshold | 5,506 bps |
| Lado selecionado | SELL |
| Horizonte | 120s |
| Alvo | 5 bps |

Self-check em Fast20:

| Campo | Valor |
|---|---:|
| Seleções | 66 |
| Payoff bruto selecionado | +1,624 bps |
| Net com 0,5 bps custo | +1,124 bps |
| Net com 1,0 bps custo | +0,624 bps |

Confirmação em Fast21:

| Campo | Valor |
|---|---:|
| Seleções | 41 |
| Target rate base | 39,09% |
| Target rate selecionado | 43,90% |
| Payoff bruto base | -0,809 bps |
| Payoff bruto selecionado | +0,823 bps |
| Lift bruto | +1,632 bps |
| Net com 0,25 bps custo | +0,573 bps |
| Net com 0,5 bps custo | +0,323 bps |
| Net com 1,0 bps custo | -0,177 bps |

Decisão: candidata principal, mas ainda não estratégia aprovada. Ela sobreviveu a uma janela futura e
venceu a base num regime que derrubou a regra `spot_perp`; porém ainda morre se o custo total passar
de aproximadamente 0,823 bps.

## Scout independente em Fast21

Rodei também um scout independente dentro de Fast21, sem forçar a regra anterior. Melhor candidato:

| Campo | Valor |
|---|---:|
| Feature | `directed_mark_index_basis_bps` |
| Direção | low |
| Fração | 33% |
| Horizonte | 120s |
| Alvo | 5 bps |
| Seleções OOS | 77 |
| Target rate selecionado | 71,43% |
| Payoff bruto selecionado | +3,848 bps |

Congelando esse candidato no Fast21 inteiro para próxima janela:

| Campo | Valor |
|---|---:|
| Threshold | -3,444 bps |
| Lado selecionado | BUY |
| Seleções | 218 |
| Payoff bruto selecionado | +1,737 bps |
| Net com 1,0 bps custo | +0,737 bps |
| Net com 2,0 bps custo | -0,263 bps |

Isso sugere que a família `mark_index_basis` pode estar capturando um sinal direcional de regime:

- em um regime, `directed_mark_index_basis_bps` alto seleciona SELL;
- em outro, `directed_mark_index_basis_bps` baixo seleciona BUY;
- a regra `spot_perp_basis` foi menos robusta.

## Decisão

Ainda não temos estratégia aprovada.

Mas agora temos uma direção mais concreta:

1. abandonar a candidata `spot_perp_basis` Q20 como regra isolada;
2. promover `mark_index_basis` para a família principal de sinais;
3. tratar direção como dependente de regime, não como SELL fixo;
4. exigir custo total abaixo de 0,5 bps para qualquer hipótese curta ser interessante;
5. confirmar a regra BUY `mark_index low` de Fast21 numa quarta janela antes de qualquer Testnet shadow.

Nenhuma ordem foi enviada. Demo/Testnet/LIVE seguem bloqueados.

## Artefatos

- WAL scout Fast21:
  `backend/data/microstructure_v1/reports/book-wal-scout-fast21-30m/report.json`
  (`d5ed5c2b3f3dcd182224d390a1db476bc5d4bfd3ee622acb106b42f4cb8fc1d7`)
- Sensibilidade Fast21:
  `backend/data/microstructure_v1/reports/book-threshold-sensitivity-fast21-30m/report.json`
  (`8e25baae2857735a6ad6664e277f3f9ceed950bf8285db5e0dfc63ad97692c27`)
- Confirmação `spot_perp` Fast16→Fast21:
  `backend/data/microstructure_v1/reports/book-rule-confirmation-fast21-spot-q20/report.json`
  (`c757472c2bbbd093e5c765ce016324c4c401ab26fd33cb573fa72f02e81a3b9b`)
- Confirmação `mark_index high` Fast20→Fast21:
  `backend/data/microstructure_v1/reports/book-rule-confirmation-fast21-mark-q10/report.json`
  (`f74cc7cd8feba461d030d613594349ceedb06510770cfbf3addf3eba3319aa77`)
- Scout independente Fast21:
  `backend/data/microstructure_v1/reports/book-rule-scout-fast21-independent/report.json`
  (`b410b3cfcd5689124d6903514e7617938ff4cca4923dca49b72ef885d2392046`)
- Self-check candidato `mark_index low` para próxima janela:
  `backend/data/microstructure_v1/reports/book-rule-confirmation-fast22-mark-low-self-fast21-q33/report.json`
  (`0ce9df74c3bd57fc7d795f322b065d79b8a8f0ed23b5556f54f19799818527db`)
- Audit status Fast21:
  `backend/data/microstructure_v1/prospective-wal-fast21-30m-audits/evidence-gate-status.json`
  (`f23cdbae884cd4f8d6cbdc59cfd5312d01955ab8060c9dd2cf921c79604fc7ba`)
