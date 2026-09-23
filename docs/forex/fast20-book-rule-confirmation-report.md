# Rodada 20 — confirmação prospectiva de regra causal de book

Data: 2026-09-23  
Mercado: BTCUSDT futures, WAL prospectivo Fast20 de 30 minutos  
Resultado: primeira candidata replicada em janela futura, ainda sem aprovação operacional

## Pergunta

As rodadas 16 a 19 mostraram que o alvo curto de 30s era estreito demais e que modelos com features de
book ainda não separavam entradas com força suficiente. A próxima pergunta era mais próxima do que
queremos construir:

> existe uma intersecção simples de sinais causais que aparece em desenvolvimento e sobrevive numa
> janela futura sem reotimização?

Esta rodada buscou regras auditáveis, não caixa-preta. A regra é congelada em Fast16 e aplicada em
Fast20 sem recalibrar pelo resultado de Fast20.

## Nova captura Fast20

Comando:

`PYTHONPATH=. .venv/bin/python scripts/research/record_microstructure.py --root data/microstructure_v1/prospective-wal-fast20-30m --symbol BTCUSDT --duration-seconds 1800 --include-spot-trades --audit-status-path data/microstructure_v1/prospective-wal-fast20-30m-audits/evidence-gate-status.json --audit-rolling-days 1 --audit-interval-seconds 120`

Streams capturados:

| Stream | Linhas | Início UTC | Fim UTC |
|---|---:|---|---|
| `trade` | 88.910 | 2026-09-23T03:24:12.812Z | 2026-09-23T03:54:11.778Z |
| `depth` | 17.638 | 2026-09-23T03:24:13.114Z | 2026-09-23T03:54:11.986Z |
| `mark_price` | 1.376 | 2026-09-23T03:24:12Z | 2026-09-23T03:54:12.002Z |
| `spot_trade` | 54.203 | 2026-09-23T03:24:13.025Z | 2026-09-23T03:54:12.017Z |

Scout WAL:

| Campo | Valor |
|---|---:|
| Status | `READY_FOR_SCOUT_MODELING` |
| Decisões | 330 |
| Linhas | 1.980 |
| Book disponível | 100% |
| Idade média do book | 52,25 ms |
| Máx. idade do book | 103 ms |

## Scout de regras em Fast16

Script novo:

`PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_book_rule_scout.py --wal-root <wal> --output-root <out> --horizons-seconds 30,60,120 --thresholds-bps 0.5,1,2,3,5 --condition-fractions 0.10,0.20,0.33 --decision-stride-seconds 5 --temporal-blocks 4 --minimum-train-blocks 2 --minimum-selected-count 10 --max-pair-features 14 --max-report-candidates 80`

O script:

- testa regras de feature alta/baixa e intersecções de duas features;
- calcula thresholds só no treino passado e aplica no bloco futuro;
- exclui colunas futuras e labels;
- mantém `research_only=true`, `order_submission_allowed=false`, `execution_authorization=none`.

Melhor candidata em Fast16:

| Campo | Valor |
|---|---:|
| Sinal | `directed_spot_perp_basis_bps` alto |
| Fração da condição | 20% |
| Horizonte | 120s |
| Alvo bruto | 5 bps |
| Seleções OOS | 27 |
| Payoff bruto selecionado | +3,861 bps |
| Payoff bruto base | -0,254 bps |
| Target rate selecionado | 40,74% |

Interpretação do sinal: `spot_perp_basis_bps = perp/spot - 1`. Como a regra selecionou apenas SELL,
`directed_spot_perp_basis_bps` alto equivale a perp negociando abaixo do spot e ainda assim vender
perp. É uma tese de continuação de fraqueza do perp, não arbitragem de convergência.

## Congelamento da regra

Ao congelar o threshold no Fast16 inteiro, a candidata fica bem menor:

| Campo | Q20 |
|---|---:|
| Feature | `directed_spot_perp_basis_bps` |
| Direção | high |
| Quantil | 0,80 |
| Threshold congelado | 5,203 bps |
| Seleções em Fast16 | 75 |
| Payoff bruto selecionado em Fast16 | +0,527 bps |
| Target rate selecionado em Fast16 | 20,00% |

Isso já mostra fragilidade: o resultado walk-forward de +3,861 bps não pode ser lido como expectativa
real; a regra congelada fica abaixo de 1 bp bruto.

## Confirmação em Fast20 sem reotimização

Comando Q20:

`PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_book_rule_confirmation.py --development-wal-root backend/data/microstructure_v1/prospective-wal-fast16-18m --confirmation-wal-root backend/data/microstructure_v1/prospective-wal-fast20-30m --output-root backend/data/microstructure_v1/reports/book-rule-confirmation-fast20-q20 --horizon-seconds 120 --threshold-bps 5 --condition-fraction 0.20 --conditions-json '[{"feature":"directed_spot_perp_basis_bps","direction":"high"}]' --decision-stride-seconds 5 --minimum-selected-count 20`

Resultado Q20:

| Campo | Valor |
|---|---:|
| Status | `CONFIRMATION_READY_FOR_REPLICATION` |
| Decisões Fast20 | 330 |
| Linhas Fast20 | 660 |
| Seleções | 56 |
| Lado selecionado | 56 SELL / 0 BUY |
| Target rate base | 29,39% |
| Target rate selecionado | 39,29% |
| Lift de target rate | +9,89 p.p. |
| Payoff bruto base | -0,061 bps |
| Payoff bruto selecionado | +0,688 bps |
| Lift de payoff bruto | +0,749 bps |
| Break-even de custo total | 0,688 bps |

Sensibilidade líquida Q20:

| Custo total | Net selecionado |
|---:|---:|
| 0,00 bps | +0,688 bps |
| 0,25 bps | +0,438 bps |
| 0,50 bps | +0,188 bps |
| 1,00 bps | -0,312 bps |
| 2,00 bps | -1,312 bps |

Resultado Q33, mais largo:

| Campo | Valor |
|---|---:|
| Seleções | 122 |
| Payoff bruto selecionado | +0,115 bps |
| Target rate selecionado | 35,25% |
| Break-even de custo total | 0,115 bps |

Q33 só fica positivo antes de custo. Com 0,25 bps de custo total já vira -0,135 bps.

Q20 é a melhor versão confirmada; Q33 confirma direção, mas é fraca demais.

## Scout independente em Fast20

Também rodei o rule scout em Fast20 sem forçar a regra de Fast16:

`PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_book_rule_scout.py --wal-root backend/data/microstructure_v1/prospective-wal-fast20-30m --output-root backend/data/microstructure_v1/reports/book-rule-scout-fast20-independent --horizons-seconds 120 --thresholds-bps 5 --condition-fractions 0.10,0.20,0.33 --decision-stride-seconds 5 --temporal-blocks 4 --minimum-train-blocks 2 --minimum-selected-count 20 --max-pair-features 14 --max-report-candidates 80`

Melhor candidata independente:

| Campo | Valor |
|---|---:|
| Sinal | `directed_mark_index_basis_bps` alto |
| Fração | 10% |
| Horizonte | 120s |
| Alvo | 5 bps |
| Seleções | 26 |
| Payoff bruto selecionado | +2,305 bps |
| Target rate selecionado | 46,15% |

Isso é relevante porque é da mesma família econômica da candidata anterior: fraqueza relativa do perp
contra referência externa (`spot`/`index`). Mas como esse sinal foi descoberto dentro de Fast20, ele
precisa de uma terceira janela futura para confirmação.

## Decisão

Agora temos uma candidata, mas ainda não uma estratégia aprovada.

O que melhorou:

- pela primeira vez uma regra causal achada em uma janela sobreviveu em uma janela futura;
- a seleção foi toda SELL e tem interpretação econômica;
- o target rate subiu de 29,39% para 39,29% em Fast20;
- o payoff bruto saiu de -0,061 bps para +0,688 bps.

O que ainda bloqueia operação:

- +0,688 bps bruto é menor que custo/slippage comum;
- qualquer custo total acima de 0,688 bps mata o edge, e com 1,0 bps a regra Q20 já fica negativa;
- só temos duas janelas curtas;
- não há confirmação em múltiplos regimes;
- não passou por gate estatístico nem simulação de execução maker/taker;
- nenhuma autorização para Demo, Testnet ou LIVE existe.

Próximo caminho real:

1. capturar uma terceira janela prospectiva;
2. aplicar Q20 congelada sem mexer;
3. congelar também a candidata `directed_mark_index_basis_bps` de Fast20 e confirmar na terceira janela;
4. medir net com custos: 0,0 / 0,25 / 0,5 / 1,0 / 2,0 bps total;
5. só considerar Testnet shadow se a regra sobreviver com custo plausível.

Nenhuma ordem foi enviada.

## Artefatos

- Script rule scout: `backend/scripts/research/run_book_rule_scout.py`
- Script confirmação congelada: `backend/scripts/research/run_book_rule_confirmation.py`
- Relatório WAL Fast20:
  `backend/data/microstructure_v1/reports/book-wal-scout-fast20-30m/report.json`
  (`b94d5b13ba93e18e4f83db5ab466366a20cd42182836eb30f6ce1f6174be9f65`)
- Relatório sensibilidade Fast20:
  `backend/data/microstructure_v1/reports/book-threshold-sensitivity-fast20-30m/report.json`
  (`48a8d0affa6d4412cd934858e75bf09e15a5573888b5394d507177845cb5cf4c`)
- Relatório confirmação Q20:
  `backend/data/microstructure_v1/reports/book-rule-confirmation-fast20-q20/report.json`
  (`83a60a673416d77410db043ba9e6e29c3e0c42626d5b536b636b70e3b5e75de9`)
- Relatório confirmação Q33:
  `backend/data/microstructure_v1/reports/book-rule-confirmation-fast20-q33/report.json`
  (`4e8dffb860ad07604384b2cd6cc342d748d873183c105767ed2875c119acb8ed`)
- Scout independente Fast20:
  `backend/data/microstructure_v1/reports/book-rule-scout-fast20-independent/report.json`
  (`365b50f517d01cac15c82d832d5f868ac58975b9444b14580451a2d587b279fa`)
- Audit status Fast20:
  `backend/data/microstructure_v1/prospective-wal-fast20-30m-audits/evidence-gate-status.json`
  (`d6bf77cf79bbad5e273407b13bb7d07e0bdc2b917a2d4caaaf181ae7db452a69`)
