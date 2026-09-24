# Fast122 — auditoria temporal do clue USDJPY H1 compression breakout

Resposta objetiva: ainda não temos estratégia. A melhor pista do Fast121 foi
congelada e testada contra o histórico local pré-2019. Ela ficou positiva no
agregado, mas fraca/instável por ano e com lower bound negativo. Isso é clue,
não bot.

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

`backend/scripts/research/run_fast122_usdjpy_compression_breakout_temporal_audit.py`

Regra:

```text
pair: USDJPY
timeframe: H1
channel_bars: 20
compression_quantile: 0.40
stop_atr_multiple: 1.5
max_hold_minutes: 720
management: fixed_3r
```

Essa regra veio do Fast121. Fast122 só verifica estabilidade temporal; não
recalibra a regra.

## Bases usadas

| Slice | Arquivo | Janela | Observação |
|---|---|---|---|
| `pre_2011_2018` | `backend/data/raw/fx_m1_hd_pre/USDJPY_M1_201101_201811.parquet` | 2011-01 a 2018-11 | histórico local pré-2019; validação H1 oracle indisponível, então é evidência diagnóstica |
| `current_2019_2026` | `backend/data/raw/fx_m1_hd/USDJPY_M1_201901_202108.parquet` + `USDJPY_M1_202109_202608.parquet` | 2019-01 a 2026-08 | base principal usada no Fast121 |

## Resultado

Status:

`FAST122_USDJPY_COMPRESSION_BREAKOUT_WEAK_TEMPORAL_STABILITY_REJECTED`

| Slice | Trades | Base médio | Stress médio | Stress total | Lower stress | Anos positivos | Risco médio |
|---|---:|---:|---:|---:|---:|---:|---:|
| `pre_2011_2018` | `759` | `+0.9647` | `+0.1188` | `+90.1785` | `-1.8107` | `2/7` | `22.4324` pips |
| `current_2019_2026` | `954` | `+3.8419` | `+2.4618` | `+2348.5896` | `+0.4440` | `5/8` | `23.2195` pips |

Stress médio por ano no pré-2019:

| Ano | Stress médio |
|---|---:|
| 2012 | `-0.0535` |
| 2013 | `+1.6735` |
| 2014 | `-1.9791` |
| 2015 | `-2.6902` |
| 2016 | `+2.8056` |
| 2017 | `-0.0044` |
| 2018 | `-0.3945` |

Leitura:

- 2019–2026 ficou bom o suficiente para justificar atenção: stress médio
  `+2.4618` pips/trade e lower bound positivo;
- pré-2019 não confirma com estabilidade: stress médio só `+0.1188`, lower bound
  negativo e apenas `2/7` anos positivos;
- o padrão parece depender de regime pós-2019;
- portanto ainda não pode ser estratégia operável.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast122 é a melhor pista da sequência recente, mas ainda reprova por estabilidade
temporal. Se formos seguir, o próximo passo correto não é operar: é congelar uma
observação prospectiva dessa regra para dados novos pós-pré-registro, ou procurar
um filtro de regime causal que explique quando esse padrão liga/desliga sem usar
o futuro.

## Comando executado

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast122_usdjpy_compression_breakout_temporal_audit.py
```

## Verificações executadas

```bash
./backend/.venv/bin/python -m pytest \
  backend/tests/unit/test_fast121_fx_compression_breakout_scout.py \
  backend/tests/unit/test_fast122_usdjpy_compression_breakout_temporal_audit.py \
  -q --no-cov

./backend/.venv/bin/python -m ruff check \
  backend/scripts/research/run_fast121_fx_compression_breakout_scout.py \
  backend/scripts/research/run_fast122_usdjpy_compression_breakout_temporal_audit.py \
  backend/tests/unit/test_fast121_fx_compression_breakout_scout.py \
  backend/tests/unit/test_fast122_usdjpy_compression_breakout_temporal_audit.py
```

Resultado:

- unit tests: `7 passed`;
- lint: `All checks passed`.

## Arquivos e hashes

- report:
  `backend/data/lab_fast122/usdjpy-compression-breakout-temporal-audit/report.json`;
- report sha256:
  `0e3a0a895b0b3b7838cae60d516aca2255655f098383b58f8f83fbb3ca274055`;
- trades:
  `backend/data/lab_fast122/usdjpy-compression-breakout-temporal-audit/frozen-policy-trades.parquet`;
- trades sha256:
  `d70f0f8929568d09208d4b878d48d34a3119b4552a32f4c3e18f3382ea51f265`;
- slice summary:
  `backend/data/lab_fast122/usdjpy-compression-breakout-temporal-audit/slice-summary.json`;
- slice summary sha256:
  `c0222d44aa62c84bbbc751f5305c40b1181992dc2dbf4599eec3d3b21f0f3c0e`.
