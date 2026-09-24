# Fast160 — Fast151 pair-specific swap sensitivity audit

Fast160 auditou a sensibilidade por par da política operacional Fast151:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only:max_per_pair_1:dd800_cooldown180d`

Status:

`FAST160_FAST151_PAIR_SWAP_LIMITS_IDENTIFIED_BROKER_TABLE_AND_FORWARD_REQUIRED`

## Resposta curta

O risco de swap ruim não é igualmente perigoso em todos os pares.

Se o swap adverso afetar apenas um par por vez, o current 2019–2026 zera LCB
aproximadamente nestes limites:

| Par | Trades current | Rollovers current | Média stress do par | Total stress do par | Break-even current |
|---|---:|---:|---:|---:|---:|
| AUDUSD | `25` | `233` | `+33.1789` | `+829.4732` | `3.4809` pips/rollover |
| EURUSD | `20` | `218` | `+24.4792` | `+489.5833` | `3.4728` pips/rollover |
| GBPUSD | `25` | `208` | `+40.6734` | `+1016.8338` | `3.8030` pips/rollover |
| NZDUSD | `16` | `186` | `+34.6998` | `+555.1969` | `4.2859` pips/rollover |

Ranking de fragilidade por swap ruim isolado:

1. `EURUSD`;
2. `AUDUSD`;
3. `GBPUSD`;
4. `NZDUSD`.

Mas isso não contradiz o Fast159: se o swap adverso vem nos quatro pares ao
mesmo tempo, o break-even current cai para `0.9589` pip/rollover.

## Break-even pré vs current

| Par | Break-even pré-2019 | Break-even 2019–2026 |
|---|---:|---:|
| AUDUSD | `17.4922` | `3.4809` |
| EURUSD | `7.5764` | `3.4728` |
| GBPUSD | `11.9135` | `3.8030` |
| NZDUSD | `13.2448` | `4.2859` |

O current é bem mais apertado que o pré-2019. Isso reforça que a corretora e o
regime de juros atual importam.

## Stress current por par isolado

| Par afetado | Swap adverso | LCB portfólio | Média portfólio | DD portfólio | Cluster de anos ruins |
|---|---:|---:|---:|---:|---|
| AUDUSD | `1.0` | `+6.9168` | `+30.9080` | `-641.9916` | `false` |
| AUDUSD | `3.0` | `+1.3602` | `+25.4894` | `-705.9916` | `false` |
| AUDUSD | `5.0` | `-4.3578` | `+20.0708` | `-769.9916` | `false` |
| EURUSD | `1.0` | `+6.9136` | `+31.0824` | `-643.9916` | `false` |
| EURUSD | `3.0` | `+1.3420` | `+26.0126` | `-711.9916` | `false` |
| EURUSD | `5.0` | `-4.3961` | `+20.9429` | `-809.0821` | `true` |
| GBPUSD | `1.0` | `+7.1544` | `+31.1987` | `-657.9916` | `false` |
| GBPUSD | `3.0` | `+2.0801` | `+26.3615` | `-753.9916` | `false` |
| GBPUSD | `5.0` | `-3.1450` | `+21.5243` | `-849.9916` | `false` |
| NZDUSD | `1.0` | `+7.4523` | `+31.4545` | `-660.9916` | `false` |
| NZDUSD | `3.0` | `+2.9680` | `+27.1289` | `-762.9916` | `false` |
| NZDUSD | `5.0` | `-1.6761` | `+22.8033` | `-864.9916` | `false` |

## Interpretação

Essa auditoria separa duas perguntas:

1. “Um par individual com swap ruim destrói tudo?”
   - Não imediatamente. Cada par isolado tolera algo acima de `3.47`
     pips/rollover no current.

2. “Se todos os pares tiverem swap ruim moderado, destrói?”
   - Sim, pode destruir. O Fast159 mostrou que `1.0 pip/rollover` nos quatro
     pares já deixa o LCB current negativo.

Leitura operacional:

- o broker real precisa ser avaliado com tabela por par;
- se apenas um par tiver short swap ruim, pode ser possível remover só esse par;
- se os quatro tiverem short swap ruim, a estratégia deve continuar reprovada;
- EURUSD e AUDUSD merecem atenção primeiro porque são os limites mais baixos no
  current.

## Segurança

Fast160 é auditoria histórica offline:

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
  `backend/scripts/research/run_fast160_fast151_pair_swap_sensitivity_audit.py`;
- teste:
  `backend/tests/unit/test_fast160_fast151_pair_swap_sensitivity_audit.py`;
- report JSON:
  `backend/data/lab_fast160/fast151-pair-swap-sensitivity-audit/report.json`;
- métricas:
  `backend/data/lab_fast160/fast151-pair-swap-sensitivity-audit/pair-swap-sensitivity-metrics.csv`;
- resumo:
  `backend/data/lab_fast160/fast151-pair-swap-sensitivity-audit/pair-swap-sensitivity-summary.json`.

Hashes:

- report:
  `111f3bd5aa1f5def8bfae2c0ef2182706122c4a67f2b0d95b051309ae0c856a8`;
- runner:
  `4c1e56cb9c7b184cff50d3cc9f7881c6cb0d032dea851b1e1f9d1f382b8c00ec`;
- teste:
  `f5fbf9da8a8354189638b1be10a7d8767618b29217b166cfe47816c5b1df8ce9`;
- métricas:
  `1339b993fb73bf4528fec76a8c5dd7a908ed67ebf093aade67a358917c06cd89`;
- resumo:
  `3e50f0ab946eaee9c0fd7a462aeb8eea8f20e44146fd4855938fc11492c0aa44`.
