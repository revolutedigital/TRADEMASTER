# Fast138 — large-move fixed-veto forward audit

Fast138 testou a família mais promissora de microestrutura/“operações maiores”
sem deixar a auditoria nova participar da seleção. A pergunta foi:

> Se escolhermos o melhor veto fixo usando só as janelas antigas
> `fast16`–`fast36`, ele sobrevive nas janelas posteriores
> `fast82`, `fast83`, `fast89`?

Status:

`FAST138_LARGE_MOVE_FIXED_VETO_FAILED_LATE_AUDIT`

## Protocolo

Entrada base:

- modelo: `hist_gradient_payoff`;
- horizonte: `300s`;
- target / breakeven activation: `10 bps`;
- trailing distance: `15 bps`;
- top fraction: `5%`;
- latência: `100ms`;
- initial stop: `20 bps`;
- modo operacional primário: one-position;
- custo principal: `1 bps`;
- custo stress: `2 bps`.

Split cronológico:

| Papel | Janelas |
|---|---|
| Seleção/treino | `fast16`, `fast20`, `fast21`, `fast22`, `fast24`, `fast28`, `fast29`, `fast32`, `fast35`, `fast36` |
| Audit tardio | `fast82`, `fast83`, `fast89` |

Seleção:

- dentro das janelas antigas, cada política foi avaliada em leave-one-window-out;
- a melhor política foi escolhida por:
  1. maior número de janelas positivas a `2 bps`;
  2. maior número de janelas com mínimo de execuções;
  3. maior net total a `2 bps`;
  4. desempates por métricas a `1 bps`.

Leakage guard:

- as janelas `fast82`, `fast83`, `fast89` não foram usadas para escolher a
  política;
- o modelo final e os thresholds do veto foram fitados só em `fast16`–`fast36`;
- o audit tardio foi somente aplicação do modelo/veto congelado.

## Política escolhida no treino antigo

Melhor política:

`spot_quote_volume_1s_low80`

Regra:

```text
spot_quote_volume_1s <= q0.80 do conjunto selecionado no treino antigo
```

Resultado nas janelas antigas:

| Métrica | Valor |
|---|---:|
| selected signals | `221` |
| execuções one-position | `25` |
| gross total | `+97.1303` bps |
| net total @ 1 bps | `+72.1303` bps |
| net total @ 2 bps | `+47.1303` bps |
| janelas positivas @ 1 bps | `9/10` |
| janelas positivas @ 2 bps | `9/10` |
| janelas com mínimo de execuções | `5/10` |
| pior janela @ 2 bps | `-46.0000` bps |

O treino antigo tinha sinal, mas já mostrava fragilidade operacional: nem todas
as janelas tinham mínimo de execuções e uma janela ainda carregava perda grande.

## Audit tardio

Resultado agregado em `fast82`, `fast83`, `fast89`:

| Métrica | Valor |
|---|---:|
| selected signals | `67` |
| execuções one-position | `10` |
| gross total | `-8.8370` bps |
| net total @ 1 bps | `-18.8370` bps |
| net total @ 2 bps | `-28.8370` bps |
| janelas positivas @ 1 bps | `1/3` |
| janelas positivas @ 2 bps | `1/3` |
| janelas com mínimo de execuções | `2/3` |
| pior janela @ 2 bps | `-17.8457` bps |

Por janela:

| Janela | Pré-gate | Pós-gate | Execuções one-position | Gross | Net @ 1 bps | Net @ 2 bps |
|---|---:|---:|---:|---:|---:|---:|
| `fast82` | `28` | `24` | `4` | `-5.4821` | `-9.4821` | `-13.4821` |
| `fast83` | `29` | `19` | `4` | `-9.8457` | `-13.8457` | `-17.8457` |
| `fast89` | `28` | `24` | `2` | `+6.4907` | `+4.4907` | `+2.4907` |

## Interpretação

Este é um corte importante: o ramo large-move com veto fixo ainda parecia
promissor em Fast104, mas quando a seleção é cronológica ele não replica nas
janelas posteriores.

O problema não é só custo. Antes mesmo de custo, o audit tardio ficou negativo
no agregado (`-8.8370` bps gross). Com custo stress, cai para `-28.8370` bps.

Leitura honesta:

- a família de operações maiores tem sinais locais;
- os vetos simples reduzem alguns outliers;
- mas o edge não está estável o bastante para virar estratégia;
- `spot_return_1s >= 0` de Fast104 não foi o melhor quando a escolha foi feita
  só nas janelas antigas;
- o melhor veto antigo (`spot_quote_volume_1s_low80`) falhou nas janelas novas.

## Segurança

Fast138 é pesquisa offline:

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
  `backend/scripts/research/run_fast138_large_move_fixed_veto_forward_audit.py`;
- teste:
  `backend/tests/unit/test_fast138_large_move_fixed_veto_forward_audit.py`;
- report JSON:
  `backend/data/microstructure_v1/reports/fast138-large-move-fixed-veto-forward-audit/report.json`;
- holdouts de seleção:
  `backend/data/microstructure_v1/reports/fast138-large-move-fixed-veto-forward-audit/train-policy-holdouts.jsonl`;
- holdouts de audit:
  `backend/data/microstructure_v1/reports/fast138-large-move-fixed-veto-forward-audit/audit-holdouts.jsonl`.

Hashes:

- report:
  `10b49b54ca7ed6c098cacc8f4209f89f1c37fb2e56a4a8b5b4f9ecd0e5f3b22c`;
- holdouts de seleção:
  `e939abd1d03b7fffd2840febcf97be1d07356a793bacd13ce6f8bc613380e4c9`;
- holdouts de audit:
  `a2d6fc9038457eb8ba382edacbc693194ff5e62132bd54956a8420f9f17981f0`.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast138 enfraquece o caminho large-move com vetos fixos simples. Se seguirmos
microestrutura, o próximo caminho deve ser uma regra rara/conjunta
pré-registrada ou uma abordagem de regime que seja selecionada sem olhar as
janelas tardias.
