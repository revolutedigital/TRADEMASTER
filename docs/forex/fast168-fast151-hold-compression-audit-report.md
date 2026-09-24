# Fast168 — Fast151 hold-compression audit

Fast168 auditou se a política operacional Fast151 pode trocar o hold máximo de
20 dias por 5 ou 10 dias para reduzir rollover/swap sem usar lookahead do
período atual.

Família auditada:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold*:stop2_fixed_1r:short_only:max_per_pair_1:dd800_cooldown180d`

Status:

`FAST168_HOLD20_REMAINS_SELECTED_COMPRESSED_HOLDS_FAIL_PRE_GATE`

## Resposta curta

Não dá para promover `hold5` ou `hold10` com segurança.

O `hold5` é tentador porque ficou muito melhor no período atual:

- current LCB: `+19.1220` pips;
- current mean: `+35.2947` pips/trade;
- current swap break-even: `5.6610` pips por rollover/dia;
- current drawdown: `-238.8375` pips.

Mas ele falha no bloco pré-2019, que é o bloco que impede lookahead:

- pré LCB: `-8.5652`;
- dois anos negativos consecutivos: `true`;
- positive-year share: `0.625`;
- falhas: `lcb_non_positive`,
  `two_consecutive_negative_years`,
  `positive_year_share_below_minimum`.

O único hold que passa pré e current segue sendo `hold20`.

## Comparativo

| Entry policy | Hold | Pré gate | Pré LCB | Pré positive years | Current gate | Current LCB | Current swap BE | Decisão |
|---|---:|---|---:|---:|---|---:|---:|---|
| `d1_donchian_fade_ch100_hold5` | 5 | FAIL | `-8.5652` | `0.625` | PASS | `+19.1220` | `5.6610` | rejeitar: current-only |
| `d1_donchian_fade_ch100_hold10` | 10 | FAIL | `+0.1297` | `0.7143` | FAIL | `+13.0575` | `1.8969` | rejeitar: instável |
| `d1_donchian_fade_ch100_hold20` | 20 | PASS | `+32.2878` | `0.875` | PASS | `+9.6335` | `0.9589` | manter |

## Interpretação

Esse audit é importante porque responde a uma rota que parecia resolver o maior
blocker operacional da candidata: reduzir tempo no trade e, portanto, reduzir
swap.

A armadilha é que `hold5` parece excelente quando olhamos 2019-2026, mas não
teria sido selecionável olhando apenas o bloco pré-2019. Promover `hold5` agora
seria escolher uma regra porque ela funcionou no holdout, exatamente o tipo de
overfit que estamos tentando evitar.

Leitura prática:

- `hold5` entra como hipótese observável futura, não como estratégia;
- `hold10` não melhora o suficiente e falha gates;
- `hold20` continua sendo a candidata histórica mais defensável;
- o blocker de swap não foi resolvido por compressão de hold.

## O que isso muda no caminho

Fast168 fecha um atalho perigoso:

- não devemos trocar para `hold5` só para ficar bonito no current;
- se quisermos usar hold curto, ele precisa ser pré-registrado como uma nova
  hipótese prospectiva e observado sem mexer regra;
- para a candidata principal, ainda precisamos da tabela real de swap da
  corretora e de M1 pós-pré-registro.

## Segurança

Fast168 é auditoria histórica offline:

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
  `backend/scripts/research/run_fast168_fast151_hold_compression_audit.py`;
- teste:
  `backend/tests/unit/test_fast168_fast151_hold_compression_audit.py`;
- report JSON:
  `backend/data/lab_fast168/fast151-hold-compression-audit/report.json`;
- métricas:
  `backend/data/lab_fast168/fast151-hold-compression-audit/hold-compression-metrics.csv`;
- resumo:
  `backend/data/lab_fast168/fast151-hold-compression-audit/hold-compression-summary.json`.

Hashes:

- report:
  `b05c0246b2787081d35f5cfcb25d4b97fc91eddeb9ec91dd68f523d359f51e48`;
- runner:
  `bcb04a9ae25fe80ec5a1ad185ff4de328e89a2cc8d935400d0bc51780e7569ea`;
- teste:
  `f165bc46932c905a539dbae0542c66de7f72867d5e9ba1f4ab3d5f8a777bc391`;
- métricas:
  `ee1d57fec096a990c8711340f639a5495bc72f4eb78908ee164aad616d76826b`;
- resumo:
  `bc7b0c4123cda7526b1b341c692d6064a7c8fc7fcb857766661577509e5c2b74`.
