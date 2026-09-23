# Rodada 22 — quarta janela e rejeição de regra única

Data: 2026-09-23  
Mercado: BTCUSDT futures, WAL prospectivo Fast22 de 30 minutos  
Resultado: regras isoladas alternam por regime; ainda não há estratégia aprovada

## Pergunta

Fast21 promoveu a família `mark_index_basis` como candidata principal e deixou uma regra BUY
`directed_mark_index_basis_bps` baixo congelada para a próxima janela. Fast22 testou:

1. `mark_index low` BUY, congelada em Fast21;
2. `mark_index high` SELL, congelada em Fast20;
3. `spot_perp high` SELL, congelada em Fast16 e já testada em Fast20/Fast21;
4. um scout independente para ver qual família reapareceria.

## Captura Fast22

Comando:

`PYTHONPATH=. .venv/bin/python scripts/research/record_microstructure.py --root data/microstructure_v1/prospective-wal-fast22-30m --symbol BTCUSDT --duration-seconds 1800 --include-spot-trades --audit-status-path data/microstructure_v1/prospective-wal-fast22-30m-audits/evidence-gate-status.json --audit-rolling-days 1 --audit-interval-seconds 120`

Streams capturados:

| Stream | Linhas | Início UTC | Fim UTC |
|---|---:|---|---|
| `trade` | 83.492 | 2026-09-23T04:31:22.170Z | 2026-09-23T05:01:21.095Z |
| `depth` | 17.637 | 2026-09-23T04:31:22.227Z | 2026-09-23T05:01:21.004Z |
| `mark_price` | 1.382 | 2026-09-23T04:31:21Z | 2026-09-23T05:01:21.003Z |
| `spot_trade` | 58.233 | 2026-09-23T04:31:22.052Z | 2026-09-23T05:01:21.130Z |

Scout WAL:

| Campo | Valor |
|---|---:|
| Status | `READY_FOR_SCOUT_MODELING` |
| Decisões | 330 |
| Linhas | 1.980 |
| Book disponível | 100% |
| Idade média do book | 50,57 ms |
| Máx. idade do book | 103 ms |

## Contexto de regime

Fast22 foi bem mais lateral que Fast21. No horizonte 120s:

| Lado | Terminal médio | MFE médio | MAE médio | Target 5 bps |
|---|---:|---:|---:|---:|
| BUY | -0,672 bps | +3,681 bps | -4,050 bps | 3,64% |
| SELL | +0,672 bps | +4,050 bps | -3,681 bps | 3,03% |

O mercado teve leve viés SELL, mas sem tendência forte.

## Confirmação de `mark_index low` BUY — falhou

Regra congelada em Fast21:

| Campo | Valor |
|---|---:|
| Feature | `directed_mark_index_basis_bps` |
| Direção | low |
| Quantil | 0,33 |
| Threshold | -3,444 bps |
| Lado selecionado | BUY |

Resultado em Fast22:

| Campo | Valor |
|---|---:|
| Seleções | 293 |
| Target rate base | 32,12% |
| Target rate selecionado | 27,30% |
| Payoff bruto base | -0,203 bps |
| Payoff bruto selecionado | -0,894 bps |
| Lift bruto | -0,691 bps |

Decisão: rejeitada como regra isolada.

## Confirmação de `mark_index high` SELL — falhou

Regra congelada em Fast20:

| Campo | Valor |
|---|---:|
| Feature | `directed_mark_index_basis_bps` |
| Direção | high |
| Quantil | 0,90 |
| Threshold | 5,506 bps |
| Lado selecionado | SELL |

Resultado em Fast22:

| Campo | Valor |
|---|---:|
| Seleções | 68 |
| Target rate selecionado | 17,65% |
| Payoff bruto selecionado | -0,157 bps |
| Lift bruto | +0,046 bps |

Decisão: também não é estratégia. Mesmo antes de custo ficou negativo.

## Controle: `spot_perp high` SELL — voltou a funcionar

Regra congelada em Fast16:

| Campo | Valor |
|---|---:|
| Feature | `directed_spot_perp_basis_bps` |
| Direção | high |
| Quantil | 0,80 |
| Threshold | 5,203 bps |
| Lado selecionado | SELL |

Histórico:

| Janela | Payoff bruto selecionado | Resultado |
|---|---:|---|
| Fast20 | +0,688 bps | positivo até 0,5 bps de custo |
| Fast21 | -2,284 bps | falhou forte |
| Fast22 | +0,946 bps | positivo até 0,5 bps de custo |

Resultado Fast22:

| Campo | Valor |
|---|---:|
| Seleções | 131 |
| Target rate base | 32,12% |
| Target rate selecionado | 38,93% |
| Payoff bruto selecionado | +0,946 bps |
| Net com 0,5 bps custo | +0,446 bps |
| Net com 1,0 bps custo | -0,054 bps |

Decisão: a regra não morreu, mas é regime-dependente. Ela funciona em Fast20/Fast22 e falha em Fast21.

## Scout independente em Fast22

O scout independente também escolheu `directed_spot_perp_basis_bps` alto:

| Campo | Valor |
|---|---:|
| Feature | `directed_spot_perp_basis_bps` |
| Direção | high |
| Fração | 20% |
| Horizonte | 120s |
| Alvo | 5 bps |
| Seleções OOS | 73 |
| Payoff bruto selecionado | +2,080 bps |
| Target rate selecionado | 36,99% |

Isso confirma que a família `spot_perp_basis` reapareceu em Fast22. O problema é a quebra em Fast21.

## Decisão

Ainda não temos estratégia aprovada.

A conclusão honesta mudou:

- não existe, até aqui, uma regra única estável;
- `spot_perp_basis` tem dois acertos e uma falha forte;
- `mark_index_basis` teve confirmação em Fast21, mas falhou em Fast22;
- as regras parecem depender do regime intrajanelas;
- qualquer versão bruta ainda exige custo total abaixo de aproximadamente 0,5 bps para ser interessante.

O próximo avanço real não é adicionar mais um indicador aleatório. É construir um gate de regime:

1. identificar previamente se a janela está em regime comprador, vendedor ou lateral;
2. só permitir `spot_perp SELL` nos regimes parecidos com Fast20/Fast22;
3. bloquear a regra no regime parecido com Fast21;
4. testar o gate em janelas futuras sem reotimizar;
5. só depois pensar em Testnet shadow.

Nenhuma ordem foi enviada. Demo/Testnet/LIVE seguem bloqueados.

## Artefatos

- WAL scout Fast22:
  `backend/data/microstructure_v1/reports/book-wal-scout-fast22-30m/report.json`
  (`9188c5aea4150777c4a96773bbc9ce6da48d94dc18ce7c10d5984080e8ecd9e4`)
- Sensibilidade Fast22:
  `backend/data/microstructure_v1/reports/book-threshold-sensitivity-fast22-30m/report.json`
  (`455391e1ba665ecf44331e61b3c41a5187bcaed743213500c5f75d5a3805d4ca`)
- Confirmação `mark low` Fast21→Fast22:
  `backend/data/microstructure_v1/reports/book-rule-confirmation-fast22-mark-low-q33/report.json`
  (`ff4451323902e97a761712733a409593e8d0d43ee3ee071bee516217ad9ba624`)
- Confirmação `mark high` Fast20→Fast22:
  `backend/data/microstructure_v1/reports/book-rule-confirmation-fast22-mark-high-q10/report.json`
  (`9b3b063b403313c45efac00169a79cac6f926adcf5d8f774f194f45cab107242`)
- Confirmação `spot_perp` Fast16→Fast22:
  `backend/data/microstructure_v1/reports/book-rule-confirmation-fast22-spot-q20/report.json`
  (`8668aa757c625ea315d10d1314f009aa0d3dc43678999a9905e2459daa078151`)
- Scout independente Fast22:
  `backend/data/microstructure_v1/reports/book-rule-scout-fast22-independent/report.json`
  (`f0837e98d83bc7620a1d4afde8fdd8a99c1eb98d87add14c529810529b25a837`)
- Audit status Fast22:
  `backend/data/microstructure_v1/prospective-wal-fast22-30m-audits/evidence-gate-status.json`
  (`b1a5aa23277b5c3e00b5f6a40806379c0b34c8aac5ee8f371d5f559019511e51`)
