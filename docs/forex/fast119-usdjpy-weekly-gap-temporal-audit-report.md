# Fast119 — auditoria temporal do clue USDJPY weekly gap

Resposta objetiva: ainda não temos estratégia. O melhor clue pequeno do Fast118
foi congelado e testado contra o histórico local pré-2019; ele ficou negativo
em 2011–2018. Isso caracteriza instabilidade temporal/regime, não edge aprovado.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`;
- sem Demo;
- sem Testnet;
- sem LIVE;
- sem envio de ordem.

## Regra congelada

Runner:

`backend/scripts/research/run_fast119_usdjpy_weekly_gap_temporal_audit.py`

Regra:

```text
pair: USDJPY
direction: continuation
minimum_gap_pips: 10
entry_delay_minutes: 15
max_hold_minutes: 720
management: fixed_full_gap
stop_gap_multiple: 1.0
target_gap_fraction: 1.0
```

Essa regra veio do top bruto do Fast118, mas não passou como estratégia porque
tinha pouca amostra. Fast119 só verifica estabilidade temporal; não recalibra a
regra.

## Bases usadas

| Slice | Arquivo | Janela | Observação |
|---|---|---|---|
| `pre_2011_2018` | `backend/data/raw/fx_m1_hd_pre/USDJPY_M1_201101_201811.parquet` | 2011-01 a 2018-11 | histórico local pré-2019; validação H1 oracle indisponível, então é evidência diagnóstica, não aprovação |
| `current_2019_2026` | `backend/data/raw/fx_m1_hd/USDJPY_M1_201901_202108.parquet` + `USDJPY_M1_202109_202608.parquet` | 2019-01 a 2026-08 | base principal validada usada no Fast118 |

## Resultado

Status:

`FAST119_USDJPY_WEEKLY_GAP_TEMPORAL_INSTABILITY_REJECTED`

| Slice | Weekly gaps | Trades | Base médio | Stress médio | Stress total | Lower stress | Anos positivos |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pre_2011_2018` | `409` | `39` | `-1.6069` | `-3.5044` | `-136.6707` | `-11.8950` | `3/6` |
| `current_2019_2026` | `398` | `42` | `+14.0442` | `+11.2942` | `+474.3580` | `+0.6567` | `8/8` |

Leitura:

- em 2019–2026, a regra parece boa;
- em 2011–2018, a mesma regra perde dinheiro depois de custo de stress;
- os anos bons e ruins se alternam no pré-2019;
- portanto o edge é dependente de regime e não pode ser promovido.

Stress médio por ano no pré-2019:

| Ano | Stress médio |
|---|---:|
| 2013 | `-5.3416` |
| 2014 | `+13.0501` |
| 2015 | `-19.8999` |
| 2016 | `+0.2251` |
| 2017 | `+11.3084` |
| 2018 | `-18.5499` |

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast119 mata a tese de que “achamos o padrão de USDJPY” de forma robusta. O que
existe é um comportamento bom no regime 2019–2026 e ruim/instável no pré-2019.
Isso pode virar pesquisa de regime, mas não bot operável.

## Comando executado

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast119_usdjpy_weekly_gap_temporal_audit.py
```

## Verificações executadas

```bash
./backend/.venv/bin/python -m pytest \
  backend/tests/unit/test_fast118_fx_weekly_gap_scout.py \
  backend/tests/unit/test_fast119_usdjpy_weekly_gap_temporal_audit.py \
  -q --no-cov

./backend/.venv/bin/python -m ruff check \
  backend/scripts/research/run_fast118_fx_weekly_gap_scout.py \
  backend/scripts/research/run_fast119_usdjpy_weekly_gap_temporal_audit.py \
  backend/tests/unit/test_fast118_fx_weekly_gap_scout.py \
  backend/tests/unit/test_fast119_usdjpy_weekly_gap_temporal_audit.py
```

Resultado:

- unit tests: `9 passed`;
- lint: `All checks passed`.

## Arquivos e hashes

- report:
  `backend/data/lab_fast119/usdjpy-weekly-gap-temporal-audit/report.json`;
- report sha256:
  `2489cdd3f3bcfe727a3f9afe6e0136cd8b8ae4358fa1fab44a08ee0f780a2e94`;
- trades:
  `backend/data/lab_fast119/usdjpy-weekly-gap-temporal-audit/frozen-policy-trades.parquet`;
- trades sha256:
  `20ebf54347c91b478f2b4411ed4e0bafff31a51fcb8771634a26252baf2b1234`;
- slice summary:
  `backend/data/lab_fast119/usdjpy-weekly-gap-temporal-audit/slice-summary.json`;
- slice summary sha256:
  `f4d8a1823d6bd12a1c113c200f576516f5c81c80c99950f5a0b32d45fa81d686`.
