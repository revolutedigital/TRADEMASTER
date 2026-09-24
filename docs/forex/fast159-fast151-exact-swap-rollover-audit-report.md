# Fast159 — Fast151 exact swap/rollover audit

Fast159 auditou a política operacional Fast151 com o calendário real de
rollover usado pelo simulador FX:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only:max_per_pair_1:dd800_cooldown180d`

Status:

`FAST159_FAST151_EXACT_ROLLOVER_CONFIRMS_TIGHT_SWAP_MARGIN_REQUIRES_BROKER_TABLE`

## Resposta curta

A conclusão do Fast158 ficou mais precisa: o edge segue vivo, mas o custo de
swap da corretora virou gate obrigatório.

Usando rollover exato às `17:00 America/New_York`, com quarta-feira cobrando
triplo:

- pré-2019 zera LCB com `3.1040` pips adversos por rollover;
- 2019–2026 zera LCB com `0.9589` pip adverso por rollover.

No período atual, `1.0 pip/rollover` adverso já deixa o LCB negativo.

## Exposição a rollover

| Amostra | Trades | Rollovers totais | Média | Mediana | Máximo | Trades sem rollover |
|---|---:|---:|---:|---:|---:|---:|
| Pré-2019 | `51` | `503` | `9.8627` | `8.0` | `20` | `1` |
| 2019–2026 | `86` | `845` | `9.8256` | `8.0` | `20` | `5` |

Por par, no current:

| Par | Trades | Rollovers totais | Média | Mediana | Máximo |
|---|---:|---:|---:|---:|---:|
| AUDUSD | `25` | `233` | `9.3200` | `7.0` | `20` |
| EURUSD | `20` | `218` | `10.9000` | `13.0` | `20` |
| GBPUSD | `25` | `208` | `8.3200` | `7.0` | `20` |
| NZDUSD | `16` | `186` | `11.6250` | `12.5` | `20` |

## Sensibilidade por swap adverso

| Swap adverso | Pré LCB | Pré média | Pré DD | Current LCB | Current média | Current DD |
|---:|---:|---:|---:|---:|---:|---:|
| `0.00` pip/rollover | `+32.2878` | `+67.4106` | `-900.2085` | `+9.6335` | `+33.6173` | `-616.6221` |
| `0.10` pip/rollover | `+31.2563` | `+66.4244` | `-910.9085` | `+8.6318` | `+32.6347` | `-626.4916` |
| `0.25` pip/rollover | `+29.7078` | `+64.9449` | `-926.9585` | `+7.1279` | `+31.1609` | `-651.2416` |
| `0.50` pip/rollover | `+27.1241` | `+62.4793` | `-953.7085` | `+4.6181` | `+28.7045` | `-692.4916` |
| `0.75` pip/rollover | `+24.5367` | `+60.0136` | `-980.4585` | `+2.1040` | `+26.2481` | `-733.7416` |
| `1.00` pip/rollover | `+21.9457` | `+57.5479` | `-1007.2085` | `-0.4142` | `+23.7917` | `-774.9916` |
| `1.50` pip/rollover | `+16.7528` | `+52.6165` | `-1060.7085` | `-5.4631` | `+18.8789` | `-857.4916` |
| `2.00` pip/rollover | `+11.5458` | `+47.6851` | `-1114.2085` | `-10.5283` | `+13.9661` | `-994.0821` |
| `3.00` pip/rollover | `+1.0906` | `+37.8224` | `-1221.2085` | `-20.7065` | `+4.1405` | `-1284.0821` |
| `5.00` pips/rollover | `-19.9747` | `+18.0969` | `-1435.2085` | `-41.2440` | `-15.5106` | `-1864.0821` |

## Interpretação

Esse audit melhora o Fast158 porque não usa mais aproximação por dia corrido.
Ele usa exatamente o mecanismo de rollover de FX:

- rollover às 17h de New York;
- quarta-feira tripla;
- fim de semana não cobrado diretamente.

A leitura operacional é simples:

- se o short swap real médio dos quatro pares for pior que aproximadamente
  `0.96 pip/rollover`, a amostra atual perde LCB;
- `0.75 pip/rollover` ainda passa, mas com folga pequena;
- a corretora, o tipo de conta e o momento de taxa de juros importam;
- antes de shadow/Testnet, precisamos preencher a tabela real de swap por par.

O runner já gera o template:

`backend/data/lab_fast159/fast151-exact-swap-rollover-audit/broker-swap-template.csv`

Colunas esperadas:

- `pair`;
- `long_pips_per_rollover_day`;
- `short_pips_per_rollover_day`.

Para essa estratégia, o campo decisivo é `short_pips_per_rollover_day`, porque a
política operacional atual é short-only.

## Segurança

Fast159 é auditoria histórica offline:

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
  `backend/scripts/research/run_fast159_fast151_exact_swap_rollover_audit.py`;
- teste:
  `backend/tests/unit/test_fast159_fast151_exact_swap_rollover_audit.py`;
- report JSON:
  `backend/data/lab_fast159/fast151-exact-swap-rollover-audit/report.json`;
- métricas:
  `backend/data/lab_fast159/fast151-exact-swap-rollover-audit/exact-swap-rollover-metrics.csv`;
- resumo:
  `backend/data/lab_fast159/fast151-exact-swap-rollover-audit/exact-swap-rollover-summary.json`;
- template de swap:
  `backend/data/lab_fast159/fast151-exact-swap-rollover-audit/broker-swap-template.csv`.

Hashes:

- report:
  `0066ab2e8502156e6b59c805b21eec23acbc1c81021726b00f99ec118866e53f`;
- runner:
  `4bc4105c26e7c8af232e8e7880f2004ccd33c2e97f0828bc5d9aaf58ec6f19c0`;
- teste:
  `a340ed1722704440ca403e6e82b65bf3868b76548dc76ddec20e7c73d3271fcb`;
- métricas:
  `6392b3932e47bb7bf47f39c95b3111a638b96f2074048a7bf2dab08eae2cd239`;
- resumo:
  `99490a941049390b4c743b1a754d470fbdfcf488a7d251aecf99c3fefda33bdb`;
- template de swap:
  `51baa43aa3135eae58c3222fee61492279c89682730e62861b87788f48e7b52e`.
