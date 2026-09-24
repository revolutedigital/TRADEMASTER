# Fast157 — Fast151 sizing-envelope audit

Fast157 converteu a política operacional Fast151 para R e estimou um envelope
de risco por trade:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only:max_per_pair_1:dd800_cooldown180d`

Status:

`FAST157_FAST151_SIZING_ENVELOPE_REQUIRES_CONSERVATIVE_RISK_AND_PROSPECTIVE_OBSERVATION`

## Resposta curta

Se essa estratégia for algum dia para shadow/Testnet, o risco por trade precisa
ser pequeno.

Usando a pior cauda p05 do bootstrap em R (`-12.6900R`):

| Budget de drawdown | Risco máximo por trade |
|---:|---:|
| `5%` | `0.394%` |
| `10%` | `0.788%` |
| `20%` | `1.576%` |

Regra prática:

- `0.25%` por trade parece conservador;
- `0.50%` por trade ainda é plausível, mas já exige estômago;
- `1.00%` por trade é agressivo demais para essa cauda histórica.

## Resultado em R

| Amostra | Trades | Total R | Média R | Mediana R | Max DD R | Pior trade R | Melhor trade R |
|---|---:|---:|---:|---:|---:|---:|---:|
| Pré-2019 | `51` | `+17.2326R` | `+0.3379R` | `+0.9910R` | `-6.8413R` | `-1.0201R` | `+0.9964R` |
| 2019–2026 | `86` | `+16.3389R` | `+0.1900R` | `+0.6041R` | `-5.0388R` | `-1.0844R` | `+0.9963R` |

## Cauda usada para sizing

A pior cauda p05 veio do bootstrap por bloco de ano no pré-2019:

`-12.6900R`

Isso é mais conservador que o drawdown real observado, porque reamostra anos e
preserva clusters de regime dentro de cada ano.

## Cenários de risco por trade

| Risco por trade | DD estimado na cauda p05 |
|---:|---:|
| `0.10%` | `1.2690%` |
| `0.25%` | `3.1725%` |
| `0.50%` | `6.3450%` |
| `0.75%` | `9.5175%` |
| `1.00%` | `12.6900%` |

## Interpretação

Essa auditoria ajuda a responder “dá para operar isso enquanto eu estou com as
minhas filhas?”:

Tecnicamente, só se for com risco baixo, circuit breaker ativo e aceitação de
períodos ruins. A estratégia não deve ser dimensionada como se fosse um sinal
frequente de alta certeza.

Leitura honesta:

- A vantagem por trade em R existe historicamente.
- O caminho tem cauda.
- O risco por trade inicial mais defensável, se um dia houver autorização para
  paper/shadow, seria algo como `0.25%`, não `1%`.
- Antes disso, ainda falta forward prospectivo pós-pré-registro.

## Segurança

Fast157 é auditoria histórica offline:

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
  `backend/scripts/research/run_fast157_fast151_sizing_envelope_audit.py`;
- teste:
  `backend/tests/unit/test_fast157_fast151_sizing_envelope_audit.py`;
- report JSON:
  `backend/data/lab_fast157/fast151-sizing-envelope-audit/report.json`;
- métricas:
  `backend/data/lab_fast157/fast151-sizing-envelope-audit/sizing-envelope-metrics.csv`;
- resumo:
  `backend/data/lab_fast157/fast151-sizing-envelope-audit/sizing-envelope-summary.json`.

Hashes:

- report:
  `25a38714077443f9d50288b91fc4e34cf1cfd0d1410a9b7b6b9ef20ca9648d4c`;
- runner:
  `608a3f55721cce00c61c087a9f679a171d3c7e88dadf9366f1c060c5758635fa`;
- teste:
  `8bf9d3b4402add0915c9cae0974642789f956051b414dbaaa947c647add2f59e`;
- métricas:
  `af670ab895a2321893fe35a9bd4f1531b28b1fcc4ca6de593cf8f325a88e417e`;
- resumo:
  `38ea3afa9960da83a9c9ac145d4557f209acd6e76e3d80fab014e19e8935253a`.
