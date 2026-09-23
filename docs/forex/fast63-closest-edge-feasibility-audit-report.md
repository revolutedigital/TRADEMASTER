# Fast63 — auditoria de viabilidade do edge mais próximo

Fast63 audita o candidato mais próximo de funcionar até aqui: o Fast40
short-horizon payoff scout.

Pergunta:

> Esse “quase edge” vira estratégia se melhorarmos custo/execução?

Resposta: **não com segurança**. O agregado parece quase viável, mas a pior
janela tem margem praticamente zero.

## Fonte auditada

Relatório Fast40:

`backend/data/microstructure_v1/reports/book-payoff-model-scout-fast40-30s-short-horizon-100ms/report.json`

Fast40 testava:

- horizonte: 30s;
- modelo: `ridge_payoff`;
- target: 1 bps;
- trailing distance: 2 bps;
- top fraction: 2%;
- one-position;
- 6 janelas de holdout: Fast24, Fast28, Fast29, Fast32, Fast35, Fast36.

## Safety boundary

O Fast63 mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Resultado

Arquivo:

- `backend/data/microstructure_v1/reports/fast63-closest-edge-feasibility-audit/report.json`

Hash:

- report: `1713a6bcadb09391b2268b6be450fe06d8a9f2d24c0992103d687dcc8470e504`.

Resumo:

- status: `FAST63_CLOSEST_EDGE_TOO_THIN_AND_REGIME_UNSTABLE`;
- agregado break-even explícito: `1.8743595069049235` bps;
- agregado no custo stress 2 bps: `-7.161508106419362` bps;
- janelas positivas em 2 bps: `3/6`;
- pior janela: `fast32`;
- break-even da pior janela: `0.007164007006558195` bps.

## Interpretação

O número agregado engana. Parece que o candidato quase paga 2 bps, mas a
replicação por janela quebra:

| Janela | Break-even custo explícito | Net 1 bps | Net 2 bps |
| --- | ---: | ---: | ---: |
| Fast24 | 2.5927 | 1.5927 | 0.5927 |
| Fast28 | 0.6506 | -0.3494 | -1.3494 |
| Fast29 | 2.2210 | 1.2210 | 0.2210 |
| Fast32 | 0.0072 | -0.9928 | -1.9928 |
| Fast35 | 3.4639 | 2.4639 | 1.4639 |
| Fast36 | 1.2918 | 0.2918 | -0.7082 |

A pior janela só suporta `0.0072` bps de custo explícito antes de zerar. Isso é
fino demais para virar plano operacional: qualquer fee, fila, slippage,
latência extra ou degradação mínima destrói.

## Conclusão

Fast40 não é caminho de promoção. Ele continua valioso como pista: existe algum
sinal muito fraco em short-horizon/payoff, mas ele não é robusto. O próximo
caminho precisa aumentar margem por trade e estabilidade por regime, não apenas
reduzir custo ou trocar trailing.
