# Fast156 — Fast151 bootstrap risk audit

Fast156 auditou risco de cauda da política operacional Fast151:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only:max_per_pair_1:dd800_cooldown180d`

Status:

`FAST156_FAST151_EDGE_POSITIVE_BUT_DRAWDOWN_TAIL_REQUIRES_SIZING_AND_PROSPECTIVE_OBSERVATION`

## Resposta curta

A candidata continua com edge positivo nas simulações, mas existe cauda de
drawdown relevante. Isso não mata a estratégia; mata a ideia de operar grande
sem sizing conservador.

Foram usadas `20.000` simulações por amostra:

- bootstrap por trade;
- bootstrap por bloco de ano, preservando sequência dentro de cada ano.

## Resultado atual real

| Amostra | Trades | Total stress | Média stress | LCB stress | Max drawdown |
|---|---:|---:|---:|---:|---:|
| Pré-2019 | `51` | `+3437.9421` | `+67.4106` | `+32.2878` | `-900.2085` |
| 2019–2026 | `86` | `+2891.0872` | `+33.6173` | `+9.6335` | `-616.6221` |

## Bootstrap por trade

| Amostra | Total p05 | Média p05 | DD p05 | Prob total <= 0 | Prob DD <= -1000 | Prob DD <= -1500 |
|---|---:|---:|---:|---:|---:|---:|
| Pré-2019 | `+1698.1438` | `+33.2969` | `-813.0608` | `0.05%` | `1.48%` | `0.06%` |
| 2019–2026 | `+837.6969` | `+9.7407` | `-1224.4240` | `1.09%` | `12.54%` | `1.56%` |

## Bootstrap por bloco de ano

| Amostra | Total p05 | Média p05 | DD p05 | Prob total <= 0 | Prob DD <= -1000 | Prob DD <= -1500 | Prob >=3 anos perdedores |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pré-2019 | `+710.8034` | `+13.6507` | `-1672.7423` | `1.91%` | `20.62%` | `10.08%` | `6.99%` |
| 2019–2026 | `+1190.0815` | `+16.4434` | `-728.2354` | `0.18%` | `0.46%` | `0.01%` | `31.81%` |

## Interpretação

O edge não desaparece no bootstrap:

- p05 de total ainda é positivo nas duas amostras;
- p05 de média ainda é positivo nas duas amostras;
- probabilidade de total <= 0 fica baixa.

Mas o caminho pode ser desconfortável:

- no current, bootstrap por trade dá `12.54%` de chance de drawdown pior que
  `-1000 pips`;
- no pré-2019, bootstrap por bloco de ano dá `20.62%` de chance de drawdown
  pior que `-1000 pips` e `10.08%` de chance pior que `-1500 pips`;
- no current, bloco de ano indica `31.81%` de chance de pelo menos `3` anos
  perdedores em uma janela de 8 anos, mesmo com total esperado positivo.

Leitura honesta:

Fast151 parece ter vantagem, mas é uma estratégia de baixa frequência e cauda
larga. O dinheiro real só faria sentido com:

- risco por trade baixo;
- limite de perda por período;
- circuit breaker mantido;
- expectativa de meses/anos ruins;
- forward prospectivo antes de qualquer Testnet/LIVE.

## Segurança

Fast156 é auditoria histórica offline:

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
  `backend/scripts/research/run_fast156_fast151_bootstrap_risk_audit.py`;
- teste:
  `backend/tests/unit/test_fast156_fast151_bootstrap_risk_audit.py`;
- report JSON:
  `backend/data/lab_fast156/fast151-bootstrap-risk-audit/report.json`;
- métricas:
  `backend/data/lab_fast156/fast151-bootstrap-risk-audit/bootstrap-risk-metrics.csv`;
- resumo:
  `backend/data/lab_fast156/fast151-bootstrap-risk-audit/bootstrap-risk-summary.json`.

Hashes:

- report:
  `9aa60074518a7794d9d1c3e9f9b928df183c17258af2be518e4a33318ea7b364`;
- runner:
  `8cd73993a2977acf36a41b370a5a7f4cab74b435d61c8cf01ed414a6b2bcf582`;
- teste:
  `fc25c26494431de9afe48d6483a4418a97fffda93bd820f5de717e83bbab45cb`;
- métricas:
  `8c7b8478acc1e3cb626c51d34d60273a00b6f76de8fdac298e00228c6e70a777`;
- resumo:
  `25fb7a79cf2c591bdbd2e81072db2837ceb53391ad462f4493263c838e359f15`.
