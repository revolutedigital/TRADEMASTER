# Fast158 — Fast151 carry/rollover cost audit

Fast158 auditou a sensibilidade da política operacional Fast151 a custo de
carregamento/swap/rollover:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only:max_per_pair_1:dd800_cooldown180d`

Status:

`FAST158_FAST151_CARRY_MARGIN_POSITIVE_BUT_TIGHT_REQUIRES_SWAP_MEASUREMENT`

## Resposta curta

A estratégia candidata continua viva, mas não pode ser tratada como “edge que
aguenta qualquer corretora”.

O recorte atual 2019–2026 tem LCB positivo sem carry extra, mas o LCB zera com
aproximadamente:

- pré-2019: `2.8284` pips adversos por dia billado;
- 2019–2026: `0.8736` pip adverso por dia billado.

Ou seja: no período atual, se o custo médio adverso real de swap/rollover ficar
perto de `1 pip/dia` de exposição, a margem estatística some no LCB.

## Exposição média

| Amostra | Trades | Dias billados totais | Média billada | Mediana billada | Máximo billado | Média elapsed |
|---|---:|---:|---:|---:|---:|---:|
| Pré-2019 | `51` | `557` | `10.9216` | `9.0` | `20` | `10.5183` |
| 2019–2026 | `86` | `927` | `10.7791` | `9.0` | `20` | `10.3937` |

Essa não é uma estratégia intraday pura. Ela segura posições por volta de
`10–11` dias em média. Por isso carry/swap importa.

## Sensibilidade por carry adverso

| Carry adverso | Pré LCB | Pré média | Pré DD | Current LCB | Current média | Current DD |
|---:|---:|---:|---:|---:|---:|---:|
| `0.00` pip/dia | `+32.2878` | `+67.4106` | `-900.2085` | `+9.6335` | `+33.6173` | `-616.6221` |
| `0.10` pip/dia | `+31.1533` | `+66.3185` | `-911.3085` | `+8.5333` | `+32.5394` | `-628.9916` |
| `0.25` pip/dia | `+29.4505` | `+64.6802` | `-927.9585` | `+6.8817` | `+30.9225` | `-657.4916` |
| `0.50` pip/dia | `+26.6098` | `+61.9498` | `-955.7085` | `+4.1259` | `+28.2278` | `-704.9916` |
| `0.75` pip/dia | `+23.7659` | `+59.2195` | `-983.4585` | `+1.3663` | `+25.5330` | `-752.4916` |
| `1.00` pip/dia | `+20.9187` | `+56.4891` | `-1011.2085` | `-1.3972` | `+22.8382` | `-799.9916` |
| `1.50` pip/dia | `+15.2147` | `+51.0283` | `-1066.7085` | `-6.9358` | `+17.4487` | `-894.9916` |
| `2.00` pip/dia | `+9.4981` | `+45.5675` | `-1122.2085` | `-12.4894` | `+12.0592` | `-1052.0821` |
| `3.00` pip/dia | `-1.9720` | `+34.6459` | `-1233.2085` | `-23.6406` | `+1.2801` | `-1371.0821` |
| `5.00` pip/dia | `-25.0515` | `+12.8028` | `-1455.2085` | `-46.1095` | `-20.2781` | `-2138.2531` |

## Interpretação

Esse audit não mata a estratégia. Ele mata uma suposição perigosa:

> “Se o backtest passou com spread/commission, carry não importa.”

Para Fast151, carry importa porque:

- a posição média dura cerca de `10–11` dias;
- o edge atual tem LCB de apenas `+9.6335` pips/trade antes de carry extra;
- `0.75` pip/dia ainda passa no current, mas com pouca folga;
- `1.00` pip/dia já deixa o LCB current negativo;
- carry pior também aumenta drawdown e reintroduz anos ruins em cenários mais
  severos.

Leitura operacional:

- medir tabela de swap/rollover real por par e lado short é obrigatório;
- não dá para aprovar shadow/Testnet só com custo fixo de spread/commission;
- se a corretora tiver carry líquido adverso perto de `1 pip/dia`, precisamos
  rejeitar ou adaptar a estratégia;
- se o carry real for baixo/positivo para os shorts selecionados, a candidata
  segue para observação prospectiva.

## Segurança

Fast158 é auditoria histórica offline:

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
  `backend/scripts/research/run_fast158_fast151_carry_cost_audit.py`;
- teste:
  `backend/tests/unit/test_fast158_fast151_carry_cost_audit.py`;
- report JSON:
  `backend/data/lab_fast158/fast151-carry-cost-audit/report.json`;
- métricas:
  `backend/data/lab_fast158/fast151-carry-cost-audit/carry-cost-metrics.csv`;
- resumo:
  `backend/data/lab_fast158/fast151-carry-cost-audit/carry-cost-summary.json`.

Hashes:

- report:
  `5cd65c1c37ee2fb9aea59e2ac8d2b88cbe73ab8312ae4811017b5bd2d6e703e4`;
- runner:
  `1faefd51b4643541f8a2b8f45de58aba690b4baa7a31b189e22dccbc7e8a1654`;
- teste:
  `48426a17060a1d68d4439f433a1f0c61e60deaba2ee160127a9fe538010c5fd7`;
- métricas:
  `638f317e1a2d542591fa57cc57d60c88a858c5b6a7df707bb124db5662d6bee4`;
- resumo:
  `093980a370600dd24882c427c68832a4f259b6426c5805e8da1acd4a5b3c8af9`.
