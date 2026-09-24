# Fast153 — Fast151 robustness audit

Fast153 auditou a concentração histórica da política operacional Fast151:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only:max_per_pair_1:dd800_cooldown180d`

Status:

`FAST153_FAST151_ROBUST_BUT_CONCENTRATION_RISKS_REQUIRE_PROSPECTIVE_OBSERVATION`

## Resposta curta

A candidata não parece depender de um único par: todos os 4 pares contribuíram
positivo no pré-2019 e no 2019–2026.

Mas existe risco de concentração temporal no período atual:

- removendo `2025`, o LCB current fica levemente negativo (`-1.1499`);
- removendo `GBPUSD`, reaparece cluster de anos negativos consecutivos no
  current;
- removendo `2020`, também reaparece cluster de anos negativos consecutivos no
  current.

Isso não mata a candidata. Mas impede a leitura “já está resolvido”. O forward
pré-registrado continua indispensável.

## Resultado base

| Amostra | Trades | Média stress | LCB stress | Total stress | Max drawdown | Anos negativos seguidos |
|---|---:|---:|---:|---:|---:|---|
| Pré-2019 | `51` | `+67.4106` | `+32.2878` | `+3437.9421` | `-900.2085` | não |
| 2019–2026 | `86` | `+33.6173` | `+9.6335` | `+2891.0872` | `-616.6221` | não |

## Contribuição por par

Pré-2019:

| Par | Trades | Total stress | Média stress | Share do total |
|---|---:|---:|---:|---:|
| AUDUSD | `7` | `+297.1507` | `+42.4501` | `8.64%` |
| EURUSD | `16` | `+150.5349` | `+9.4084` | `4.38%` |
| GBPUSD | `13` | `+1644.4001` | `+126.4923` | `47.83%` |
| NZDUSD | `15` | `+1345.8565` | `+89.7238` | `39.15%` |

2019–2026:

| Par | Trades | Total stress | Média stress | Share do total |
|---|---:|---:|---:|---:|
| AUDUSD | `25` | `+829.4732` | `+33.1789` | `28.69%` |
| EURUSD | `20` | `+489.5833` | `+24.4792` | `16.93%` |
| GBPUSD | `25` | `+1016.8338` | `+40.6734` | `35.17%` |
| NZDUSD | `16` | `+555.1969` | `+34.6998` | `19.20%` |

Boa notícia: todos os pares ficaram positivos no período atual. Isso é muito
melhor do que uma regra que só vive de um ativo.

## Leave-one-out

Pares, 2019–2026:

| Removido | Trades | Média stress | LCB stress | Max drawdown | Cluster ruim? |
|---|---:|---:|---:|---:|---|
| AUDUSD | `61` | `+33.7970` | `+2.9574` | `-585.4722` | não |
| EURUSD | `66` | `+36.3864` | `+10.0207` | `-585.4722` | não |
| GBPUSD | `61` | `+30.7255` | `+4.9962` | `-652.2566` | sim |
| NZDUSD | `70` | `+33.3699` | `+5.2489` | `-542.4954` | não |

Anos, 2019–2026:

| Removido | Trades | Média stress | LCB stress | Max drawdown | Cluster ruim? |
|---|---:|---:|---:|---:|---|
| 2019 | `81` | `+38.2688` | `+13.5095` | `-616.6221` | não |
| 2020 | `61` | `+41.0289` | `+13.4441` | `-602.5630` | sim |
| 2021 | `78` | `+37.4493` | `+12.5913` | `-616.6221` | não |
| 2022 | `83` | `+31.8472` | `+7.6298` | `-616.6221` | não |
| 2023 | `79` | `+30.0903` | `+4.5301` | `-616.6221` | não |
| 2024 | `75` | `+32.3123` | `+6.1239` | `-616.6221` | não |
| 2025 | `67` | `+26.0181` | `-1.1499` | `-616.6221` | não |
| 2026 | `78` | `+32.3966` | `+6.7886` | `-616.6221` | não |

## Interpretação

O resultado é melhor do que “só sorte em um par”, mas pior do que “blindado”.

O que melhora a confiança:

- os 4 pares contribuem positivo no current;
- leave-one-pair mantém LCB positivo no current;
- a versão completa mantém LCB positivo, drawdown controlado e sem anos
  negativos consecutivos.

O que segura a aprovação:

- `2025` pesa bastante no current;
- sem `2025`, o LCB fica um pouco abaixo de zero;
- algumas exclusões reintroduzem cluster de anos ruins;
- ainda falta dado prospectivo real após `2026-09-24T06:18:19-03:00`.

Leitura honesta:

Fast151 continua sendo a melhor candidata da linha, mas a régua correta é:
observar forward, não promover por backtest.

## Segurança

Fast153 é auditoria histórica offline:

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
  `backend/scripts/research/run_fast153_fast151_robustness_audit.py`;
- teste:
  `backend/tests/unit/test_fast153_fast151_robustness_audit.py`;
- report JSON:
  `backend/data/lab_fast153/fast151-robustness-audit/report.json`;
- contribuição por par:
  `backend/data/lab_fast153/fast151-robustness-audit/pair-contribution.csv`;
- contribuição por ano:
  `backend/data/lab_fast153/fast151-robustness-audit/year-contribution.csv`;
- leave-one-pair:
  `backend/data/lab_fast153/fast151-robustness-audit/leave-one-pair.csv`;
- leave-one-year:
  `backend/data/lab_fast153/fast151-robustness-audit/leave-one-year.csv`;
- resumo:
  `backend/data/lab_fast153/fast151-robustness-audit/robustness-summary.json`.

Hashes:

- report:
  `a529b5afc09d94f22d41043f738c1a00f3cab7321cd5785db1e76032432025e7`;
- runner:
  `60dea7cebf95ec1070f130d953ddccfd963b3a33db38fe466912ef649e08566b`;
- teste:
  `d7e84a2c45938359844ba668b21c5fac975012337b6b1cd24ffe923ecbbad476`;
- contribuição por par:
  `a39838e5814b75ec60b1c5a6e6481469fe36bcb42dae1316e36b8b6cd962a9bc`;
- contribuição por ano:
  `e24477ea6de8d5813590e144434db8f7a77847c9dcc981910c83aa4394b98590`;
- leave-one-pair:
  `8099570f2a1fcd3606b2110632ae9c378901f96b04dcf5b88f9ed2b6e887bd76`;
- leave-one-year:
  `43fd420410f1da49b1aed83cb2194d47d303dfb5e895082bb52d4243f034d519`;
- resumo:
  `c5f48b2cd0f5d39ab3e646c38958c560e770d36175c539e85910a07b82345681`.
