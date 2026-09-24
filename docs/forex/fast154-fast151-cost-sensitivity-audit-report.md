# Fast154 — Fast151 cost-sensitivity audit

Fast154 auditou a margem de custo da política operacional Fast151:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only:max_per_pair_1:dd800_cooldown180d`

Status:

`FAST154_FAST151_COST_MARGIN_POSITIVE_BUT_NOT_UNLIMITED_REQUIRES_PROSPECTIVE_OBSERVATION`

## Resposta curta

A candidata tem margem contra custo adicional, mas não infinita.

Ela sobrevive historicamente a:

- custo `STRESS` já registrado;
- `STRESS + 5 pips` por trade;
- dobrar o drag base→stress;
- triplicar o drag base→stress, mas por muito pouco no current.

Ela não mantém LCB positivo no current com:

- `STRESS + 10 pips` por trade.

## Resultado

| Cenário | Pré média | Pré LCB | Current média | Current LCB | Ambos LCB > 0? |
|---|---:|---:|---:|---:|---|
| `stress_as_recorded` | `+67.4106` | `+32.2878` | `+33.6173` | `+9.6335` | sim |
| `stress_plus_1pip` | `+66.4106` | `+31.2878` | `+32.6173` | `+8.6335` | sim |
| `stress_plus_2pips` | `+65.4106` | `+30.2878` | `+31.6173` | `+7.6335` | sim |
| `stress_plus_5pips` | `+62.4106` | `+27.2878` | `+28.6173` | `+4.6335` | sim |
| `stress_plus_10pips` | `+57.4106` | `+22.2878` | `+23.6173` | `-0.3665` | não |
| `stress_plus_15pips` | `+52.4106` | `+17.2878` | `+18.6173` | `-5.3665` | não |
| `double_base_to_stress_drag` | `+66.6475` | `+31.4786` | `+29.8950` | `+5.1198` | sim |
| `triple_base_to_stress_drag` | `+65.8843` | `+30.6685` | `+26.1728` | `+0.1154` | sim |

## Break-even aproximado

O limite de custo fixo adicional antes do LCB bater zero é:

| Amostra | Extra pips/trade até LCB zero | Extra pips/trade até média zero |
|---|---:|---:|
| Pré-2019 | `32.2878` | `67.4106` |
| 2019–2026 | `9.6335` | `33.6173` |

Leitura: o gargalo é o current. A estratégia tem folga contra alguns pips a
mais de custo, mas não pode operar em venue/spread/slippage ruim.

## Diagnóstico de drag de custo

| Amostra | Execução base média | Execução stress média | Delta quote stress médio | Delta total base→stress |
|---|---:|---:|---:|---:|
| Pré-2019 | `0.7010` | `1.1010` | `0.3632` | `0.7632` |
| 2019–2026 | `0.6280` | `1.0280` | `3.3223` | `3.7223` |

O stress atual já embute piora relevante no período current: em média `3.72`
pips/trade contra base. Triplicar esse drag ainda deixa o LCB current em
`+0.1154`, praticamente no limite.

## Interpretação

Isso responde uma parte importante da pergunta do Igor sobre custo:

- não é uma estratégia que só existe sem custo;
- ela aguenta stress razoável;
- mas o edge não é grande o suficiente para ignorar execução;
- venue ruim, slippage alto ou spread instável podem destruir a vantagem.

Regra operacional implícita para o futuro:

Antes de qualquer shadow/Testnet, o paper/demo precisa medir custo realizado.
Se o custo efetivo ficar perto de `+10 pips/trade` além do stress histórico, a
estratégia perde a margem estatística no current.

## Segurança

Fast154 é auditoria histórica offline:

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
  `backend/scripts/research/run_fast154_fast151_cost_sensitivity_audit.py`;
- teste:
  `backend/tests/unit/test_fast154_fast151_cost_sensitivity_audit.py`;
- report JSON:
  `backend/data/lab_fast154/fast151-cost-sensitivity-audit/report.json`;
- métricas:
  `backend/data/lab_fast154/fast151-cost-sensitivity-audit/cost-sensitivity-metrics.csv`;
- resumo:
  `backend/data/lab_fast154/fast151-cost-sensitivity-audit/cost-sensitivity-summary.json`.

Hashes:

- report:
  `2d007618cf7070fd7915710b5a948cb0a9f691114cd54f9ce98e7c23d8956d34`;
- runner:
  `787890aadb657b531b30e9c662fd2eab02a8c7e716475913c6965bb3eda3ea83`;
- teste:
  `71bf4a1cd6c14dd233d7db60293e857ff67f09d8d1702790cd4dc9b5f18ea444`;
- métricas:
  `22e3836405abf3edd59b2735ea48bf741980c08b1bd57faa76133a98a3e31b6e`;
- resumo:
  `2892e5eeada8b9b874520e83ea6e226ff61b4eea0a5b23ea9b5be402820f64f3`.
