# Fast164 — Fast151 broker-swap intake pack

Fast164 transforma o bloqueio `broker_swap_table_missing` do Fast163 em um
fluxo executável para a candidata Fast151:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only:max_per_pair_1:dd800_cooldown180d`

Status:

`FAST164_BROKER_SWAP_INPUT_PENDING`

## Resposta curta

Temos uma estratégia candidata histórica, mas ainda não aprovada.

Fast164 não aprova trade. Ele cria o pacote para preencher a tabela real de swap
da corretora, normalizar para o formato do Fast159 e rodar o gate de custo real.

## O que foi criado

Template para preencher com dados reais da corretora:

`backend/data/lab_fast164/fast151-broker-swap-intake/broker-swap-raw-intake-template.csv`

Colunas principais:

- `pair`;
- `broker_symbol`;
- `long_swap_value`;
- `short_swap_value`;
- `value_unit`;
- `points_per_pip`;
- `notes`;
- `current_pair_only_break_even_pips_per_rollover_day`.

Também foi gerado um arquivo de limites por par:

`backend/data/lab_fast164/fast151-broker-swap-intake/broker-swap-pair-limits.csv`

## Convenção de preenchimento

Valores positivos são crédito. Valores negativos são débito.

A estratégia Fast151 usa apenas o lado `SHORT`, então a coluna crítica para o
gate atual é:

`short_swap_value`

Unidades aceitas:

- `pips_per_rollover_day`;
- `points_per_rollover_day`.

Se a corretora mostrar pontos e `10 pontos = 1 pip`, use:

- `value_unit=points_per_rollover_day`;
- `points_per_pip=10`.

## Limites de swap identificados no Fast160

| Par | Break-even current por par, pips/rollover |
|---|---:|
| `AUDUSD` | `3.4809` |
| `EURUSD` | `3.4728` |
| `GBPUSD` | `3.8030` |
| `NZDUSD` | `4.2859` |

O break-even uniforme combinado do Fast159 continua apertado:

`0.9589` pip/rollover

Isso significa que a candidata é sensível a custo real. A tabela de swap da
corretora não é detalhe operacional; ela decide se o edge histórico sobrevive.

## Como rodar quando a tabela real estiver preenchida

Com o template preenchido em algum CSV:

`backend/.venv/bin/python backend/scripts/research/run_fast164_fast151_broker_swap_intake.py --raw-swap-path <swap-preenchido.csv>`

O Fast164 então:

1. valida se os quatro pares estão presentes;
2. converte pontos para pips quando necessário;
3. gera `fast159-broker-swap-input.csv`;
4. roda o Fast159 com a tabela normalizada;
5. retorna se a tabela real passa ou falha o gate de swap.

Critério positivo:

`FAST164_BROKER_SWAP_INPUT_PASSES_FAST159_GATE`

Mesmo se isso passar, ainda segue faltando observação forward pós-pré-registro.

## Segurança

Fast164 é pacote de input e validação offline:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- sem contato com corretora;
- sem shadow;
- sem Testnet;
- sem LIVE;
- sem ledger.

## Artefatos

- runner:
  `backend/scripts/research/run_fast164_fast151_broker_swap_intake.py`;
- teste:
  `backend/tests/unit/test_fast164_fast151_broker_swap_intake.py`;
- report JSON gerado:
  `backend/data/lab_fast164/fast151-broker-swap-intake/report.json`;
- template bruto:
  `backend/data/lab_fast164/fast151-broker-swap-intake/broker-swap-raw-intake-template.csv`;
- limites por par:
  `backend/data/lab_fast164/fast151-broker-swap-intake/broker-swap-pair-limits.csv`;
- input normalizado Fast159:
  `backend/data/lab_fast164/fast151-broker-swap-intake/fast159-broker-swap-input.csv`;
- resumo:
  `backend/data/lab_fast164/fast151-broker-swap-intake/broker-swap-intake-summary.json`.

Hashes:

- report:
  `55300be473f518d842160ac6cb651a8c682f7528bb8604c725aceb0eac847298`;
- runner:
  `d8be43b5bbe2d7317ee3850fbfa0cbb160960c19ef2eca5d1a1dfa2e3a05e420`;
- teste:
  `5802a4952e22a8ded2a11eab87dfe283958458069332f3ee894f55aecfc8bdf3`;
- template bruto:
  `459259e5adde0acd1bba1e2755922a7bea3c9629ef898e869ef8b747441a8949`;
- limites por par:
  `7f22227bb87c8dbf5dd262479dae86bf24eb853f6138331b2570a466fbd73474`;
- input normalizado:
  `705cccdca740c1d488f603d3eb3c3fab7c2bb54e5cbf7ad25c8da347b9fba674`;
- resumo:
  `2803d966b75e639e8409fc1c10fd07a3476cb0750d316d17a74901fe760ca02e`.

## Validação

Comandos executados:

- `backend/.venv/bin/python -m pytest backend/tests/unit/test_fast164_fast151_broker_swap_intake.py -q --no-cov`;
- `cd backend && .venv/bin/ruff check scripts/research/run_fast164_fast151_broker_swap_intake.py tests/unit/test_fast164_fast151_broker_swap_intake.py`;
- `backend/.venv/bin/python backend/scripts/research/run_fast164_fast151_broker_swap_intake.py`.

Resultado:

- `6 passed`;
- `All checks passed`;
- status gerado: `FAST164_BROKER_SWAP_INPUT_PENDING`.
