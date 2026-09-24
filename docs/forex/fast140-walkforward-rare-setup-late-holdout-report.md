# Fast140 — walk-forward rare setup com late holdout

Fast140 recomeçou a busca de setup raro do jeito correto depois que Fast139
matou as regras raras congeladas Fast80/Fast68. Em vez de reaproveitar uma regra
antiga, ele selecionou regras causais simples e de duas condições apenas nas
janelas antigas, depois auditou as finalistas nas janelas tardias.

Status:

`FAST140_RARE_SETUP_FAILED_LATE_HOLDOUT`

## Protocolo

Fonte:

- input:
  `backend/data/microstructure_v1/reports/fast92-broad-universe-scout-all-local/independent_settlements.jsonl`;
- linhas liquidadas: `6,579`;
- universo: broad near-threshold all-local, diagnóstico de pesquisa, não
  estratégia deployável.

Separação temporal:

- seleção/treino: `fast16`, `fast20`, `fast21`, `fast22`, `fast24`, `fast28`,
  `fast29`, `fast32`, `fast35`, `fast36`;
- holdout tardio: `fast82`, `fast83`.

Busca:

- `39` features causais;
- `546` candidatos atômicos;
- `10` pares candidatos após filtros de seed;
- mínimos de treino: `10` trades e `3` labels;
- mínimos no holdout tardio: `5` trades e `2` labels;
- ranking por stress total e estabilidade de labels.

## Resultado

| Métrica | Valor |
|---|---:|
| Regras elegíveis no treino | `12` |
| Finalistas auditadas | `12` |
| Sobreviventes no holdout tardio | `0` |

O melhor setup no treino foi:

`flow_imbalance_30s<=q0.1:-0.5755481364 AND probability<=q0.2:0.8105199679`

| Amostra | Trades | Labels | Stress total | Labels positivas | Labels negativas | Pior label |
|---|---:|---:|---:|---:|---:|---:|
| Treino | `20` | `9` | `+77.1058` bps | `6` | `3` | `-6.6685` bps |
| Holdout tardio | `9` | `2` | `-51.8665` bps | `0` | `2` | `-27.5416` bps |

O terceiro setup mais relevante, `directed_return_30s_bps>=q0.5:0`, também
falhou:

| Amostra | Trades | Labels | Stress total | Labels positivas | Labels negativas | Pior label |
|---|---:|---:|---:|---:|---:|---:|
| Treino | `72` | `10` | `+18.4930` bps | `6` | `4` | `-40.2011` bps |
| Holdout tardio | `11` | `2` | `-43.5944` bps | `0` | `2` | `-29.9805` bps |

## Interpretação

Fast140 testou a tese de “intersecção de sinais” de forma mais honesta que uma
otimização comum: sinais selecionados em janelas antigas, depois cobrados em
janelas recentes que não participaram da seleção.

A leitura é dura:

- havia setups que pareciam bons no treino;
- as regras elegíveis eram poucas, o que já indica fragilidade estatística;
- as melhores regras viraram negativas nas duas labels tardias disponíveis;
- nenhum setup passou simultaneamente cobertura mínima e stress positivo no
  holdout tardio;
- portanto, este caminho não autoriza Demo, Testnet nem LIVE.

Isso não prova que não exista edge no mercado. Prova que este lote de sinais de
microestrutura, neste universo broad/near-threshold, não entregou uma regra que
sobrevive fora da amostra.

## Segurança

Fast140 é pesquisa offline:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`.

Nenhum trade foi enviado. Nenhum ledger foi alterado. O resultado não autoriza
Demo, Testnet nem LIVE.

## Artefatos

- runner:
  `backend/scripts/research/run_fast140_walkforward_rare_setup_late_holdout.py`;
- teste:
  `backend/tests/unit/test_fast140_walkforward_rare_setup_late_holdout.py`;
- report JSON:
  `backend/data/microstructure_v1/reports/fast140-walkforward-rare-setup-late-holdout/report.json`;
- avaliações atômicas:
  `backend/data/microstructure_v1/reports/fast140-walkforward-rare-setup-late-holdout/atom-evaluations.json`;
- avaliações de pares:
  `backend/data/microstructure_v1/reports/fast140-walkforward-rare-setup-late-holdout/pair-evaluations.json`;
- finalistas:
  `backend/data/microstructure_v1/reports/fast140-walkforward-rare-setup-late-holdout/selected-finalists.json`;
- sobreviventes tardios:
  `backend/data/microstructure_v1/reports/fast140-walkforward-rare-setup-late-holdout/late-survivors.json`;
- linhas one-position das finalistas:
  `backend/data/microstructure_v1/reports/fast140-walkforward-rare-setup-late-holdout/finalist-one-position-rows.jsonl`.

Hashes:

- input:
  `6cdd6f9fa257ddc56b0d5537329d7bd7495e7603d40917d6bcaf17407c3f8cb6`;
- report:
  `df8e44239b6ee9a06e76d3ba4b28de66a3b587c7aa41cb9056540dab6c72c31d`;
- avaliações atômicas:
  `05918d7e9b89d0c2d861fb9ecbb2d59a369f8d8b8462e8c75a81d7262a19be3c`;
- avaliações de pares:
  `d54a9dbff4f097771da375ced9297bfa1ea7a28e0bc708012058d866a2b0e280`;
- finalistas:
  `c00643d83e603f5f1997d3471466256805e6c7127df57e5d5e6593d10c890433`;
- sobreviventes tardios:
  `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`;
- linhas one-position:
  `73cdef3c349263b6e9f867741137e116ad15fb91e7bad4cff158d16c92f5c51f`.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast140 descarta este lote de setups raros walk-forward como estratégia
executável. O próximo caminho precisa sair de “micro sinal curto que paga custo
demais” e testar candidatos com movimento líquido maior, ou aguardar dados
pós-pré-registro dos candidatos já congelados antes de qualquer promoção.
