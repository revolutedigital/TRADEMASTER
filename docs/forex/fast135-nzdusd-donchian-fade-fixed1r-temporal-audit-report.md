# Fast135 — NZDUSD Donchian fade fixed 1R temporal audit

Fast135 auditou a candidata Fast134 em uma janela temporal anterior à usada na
descoberta principal. O objetivo foi responder: o candidato `fixed_1r` aparece
só no regime 2019–2026 ou também tem sustentação no histórico pré-2019?

Status:

`FAST135_TEMPORAL_AUDIT_MIXED_STABILITY_REQUIRES_PROSPECTIVE_CONFIRMATION`

## Regra congelada

`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r`

- par: `NZDUSD`;
- sinal: D1 Donchian fade;
- canal: `100` dias;
- hold máximo: `20` dias;
- stop: `2.0 * ATR(14)` diário causal;
- alvo: `1R`;
- custos: base `FUSION_ZERO` e stress `STRESS`;
- cenário primário: `STRESS`.

## Bases usadas

| Slice | Arquivo | Janela | Observação |
|---|---|---|---|
| `pre_2011_2018` | `backend/data/raw/fx_m1_hd_pre/NZDUSD_M1_201101_201811.parquet` | 2011-01 a 2018-11 | histórico local pré-2019; feed difere do atual, então é diagnóstico temporal |
| `current_2019_2026` | `backend/data/raw/fx_m1_hd/NZDUSD_M1_201901_202108.parquet` + `NZDUSD_M1_202109_202608.parquet` | 2019-01 a 2026-08 | base principal usada na descoberta/audit |

## Resultado

| Slice | Sinais | Trades | Stress médio | Stress total | LCB 95% | Anos positivos |
|---|---:|---:|---:|---:|---:|---:|
| `pre_2011_2018` | `82` | `62` | `+17.5648` | `+1089.0181` | `-12.5979` | `6/8` |
| `current_2019_2026` | `79` | `65` | `+45.2657` | `+2942.2717` | `+21.8566` | `7/8` |

Stress médio por ano no pré-2019:

| Ano | Trades | Stress médio |
|---|---:|---:|
| 2011 | `1` | `+235.3461` |
| 2012 | `3` | `+126.9080` |
| 2013 | `5` | `+61.5176` |
| 2014 | `16` | `-32.8847` |
| 2015 | `13` | `-68.7467` |
| 2016 | `8` | `+102.7310` |
| 2017 | `6` | `+24.5450` |
| 2018 | `10` | `+61.6104` |

Stress médio por ano em 2019–2026:

| Ano | Trades | Stress médio |
|---|---:|---:|
| 2019 | `7` | `+30.2552` |
| 2020 | `12` | `+26.1522` |
| 2021 | `7` | `+85.7256` |
| 2022 | `12` | `-18.3948` |
| 2023 | `6` | `+102.4254` |
| 2024 | `8` | `+63.0757` |
| 2025 | `8` | `+52.2561` |
| 2026 | `5` | `+100.0223` |

## Interpretação

O candidato ficou positivo também no histórico pré-2019, mas não com força
estatística suficiente:

- pré-2019 tem média stress positiva (`+17.5648`), mas LCB negativo
  (`-12.5979`);
- 2014 e 2015 foram ruins;
- o período 2019–2026 é bem mais forte: média stress `+45.2657`, LCB
  `+21.8566`, anos positivos `7/8`.

Leitura honesta: a candidata é real o bastante para continuar sendo a melhor
pista, mas parece depender de regime. Isso reforça a necessidade do Fast134
prospectivo antes de qualquer shadow/Testnet.

## Segurança

Fast135 é diagnóstico offline:

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
  `backend/scripts/research/run_fast135_nzdusd_donchian_fade_fixed1r_temporal_audit.py`;
- teste:
  `backend/tests/unit/test_fast135_nzdusd_donchian_fade_fixed1r_temporal_audit.py`;
- report JSON:
  `backend/data/lab_fast135/nzdusd-donchian-fade-fixed1r-temporal-audit/report.json`;
- trades:
  `backend/data/lab_fast135/nzdusd-donchian-fade-fixed1r-temporal-audit/temporal-audit-trades.parquet`;
- resumo por slice:
  `backend/data/lab_fast135/nzdusd-donchian-fade-fixed1r-temporal-audit/slice-summary.csv`.

Hashes:

- report:
  `ec2710bd2b05eef432594e9b31d2276d788f667d570a7cf304f2d8e5066dcbcf`;
- trades:
  `a4de5c3a9cd0502c814cf423aee65239deb2fb01aae0654b1ae58f47fe5f63a6`;
- resumo:
  `1e6f005a833666564cc710a37bc07c5ba03e1f9dcec99359a6c104bb0992471b`.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Continuar com Fast134 prospectivo. Se os dados prospectivos confirmarem o
regime recente, aí sim escrever novo relatório propondo shadow offline.
