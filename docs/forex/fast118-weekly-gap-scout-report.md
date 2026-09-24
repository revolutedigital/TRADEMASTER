# Fast118 — scout de gap semanal em FX M1

Resposta objetiva: ainda não temos estratégia. O ramo novo testou uma hipótese
estrutural diferente — gap de abertura semanal em FX — mas nada com amostra
mínima passou na auditoria 2024–2026 sob custo de stress.

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

## O que foi testado

Runner:

`backend/scripts/research/run_fast118_fx_weekly_gap_scout.py`

Base:

- `backend/data/raw/fx_m1_hd`;
- 10 pares: AUDUSD, EURGBP, EURJPY, EURUSD, GBPJPY, GBPUSD, NZDUSD, USDCAD,
  USDCHF e USDJPY;
- janela usada: `2019-01-01` até `2026-09-01`;
- setembro/2026 parcial e não-validado ficou fora.

Hipótese:

- detectar o primeiro candle M1 depois de uma pausa de mercado maior que 36h;
- medir gap do mid close pré-fim-de-semana para o mid open da reabertura;
- operar continuação ou reversão do gap;
- delays de entrada: 5, 15 e 30 minutos;
- holds máximos: 4h e 12h;
- thresholds de gap: 2, 5 e 10 pips;
- saídas: alvo de meio gap, alvo de gap inteiro ou trailing após meio gap.

Custos:

- base: spread bid/ask observado + comissão Fusion Zero + slippage;
- stress: spread dobrado + slippage maior + comissão.

## Resultado

Status:

`FAST118_FX_WEEKLY_GAP_FAILED_AUDIT`

Resumo:

| Métrica | Valor |
|---|---:|
| Candidatos par-política | `1080` |
| Trade rows simuladas | `87426` |
| Candidatos positivos elegíveis na seleção | `1` |
| Finalistas auditados | `1` |
| Candidatos aprovados | `0` |
| Estratégia aprovada | `false` |

O único finalista com amostra mínima foi:

| Candidate | Trades seleção | Stress seleção | Trades auditoria | Stress auditoria | Lower stress auditoria | Passou |
|---|---:|---:|---:|---:|---:|---|
| `USDJPY:continuation_gap2_d30_h12_fixed_full_gap` | `60` | `+0.1301` pips/trade | `50` | `-5.9258` pips/trade | `-13.2102` | não |

Ou seja: quando exigimos quantidade mínima de trades, a pista some no período
fora da seleção.

## Pista fraca encontrada

O ranking bruto da seleção achou uma concentração em USDJPY: continuação de gaps
grandes (`>=10 pips`) com hold de 12h.

Top bruto da seleção:

| Candidate | Trades seleção | Base seleção | Stress seleção | Anos positivos |
|---|---:|---:|---:|---:|
| `USDJPY:continuation_gap10_d15_h12_fixed_full_gap` | `23` | `+14.8616` | `+12.4225` | `5/5` |
| `USDJPY:continuation_gap10_d30_h12_fixed_full_gap` | `31` | `+7.8749` | `+6.2991` | `4/5` |
| `USDJPY:continuation_gap10_d15_h12_fixed_half_gap` | `23` | `+9.2247` | `+6.2040` | `4/5` |

Mas isso não é estratégia porque a amostra é pequena. Na auditoria 2024–2026,
as versões positivas também ficaram com `19–27` trades, abaixo do mínimo. A
melhor ficou positiva, mas pequena demais para aprovar:

| Candidate | Trades auditoria | Base auditoria | Stress auditoria |
|---|---:|---:|---:|
| `USDJPY:continuation_gap10_d15_h12_fixed_full_gap` | `19` | `+13.0548` | `+9.9284` |

Essa pista virou o Fast119: regra congelada, testada contra o histórico local
pré-2019.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast118 respondeu uma pergunta importante: gap semanal existe como fenômeno, mas
o formato amplo par-política não encontra um edge robusto com amostra mínima. O
único sinal interessante é USDJPY/gap grande/continuação, mas ele é pequeno
demais e exige auditoria temporal separada.

## Comando executado

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast118_fx_weekly_gap_scout.py
```

## Verificações executadas

```bash
./backend/.venv/bin/python -m pytest \
  backend/tests/unit/test_fast118_fx_weekly_gap_scout.py \
  -q --no-cov

./backend/.venv/bin/python -m ruff check \
  backend/scripts/research/run_fast118_fx_weekly_gap_scout.py \
  backend/tests/unit/test_fast118_fx_weekly_gap_scout.py
```

Resultado:

- unit tests: `7 passed`;
- lint: `All checks passed`.

## Arquivos e hashes

- report:
  `backend/data/lab_fast118/fx-weekly-gap-scout/report.json`;
- report sha256:
  `c34087a16580f2a9b475574a3ffd129ef8478f2c9570754cc64ba07cda593ba3`;
- trades:
  `backend/data/lab_fast118/fx-weekly-gap-scout/weekly-gap-trades.parquet`;
- trades sha256:
  `33346ec8df6667992e39df2d586387127ba9d488a6e4e30069dedcfc22cf5b2f`;
- seleção:
  `backend/data/lab_fast118/fx-weekly-gap-scout/selection-metrics.csv`;
- seleção sha256:
  `8fd4ea1025c6b11aa71f03fbc95130fad495145171414479851a9b9019c3d523`;
- auditoria:
  `backend/data/lab_fast118/fx-weekly-gap-scout/audited-finalists.csv`;
- auditoria sha256:
  `9c0f9540ae91c8fe4574aa4e5ed28d5ee96faccc2a297c5b654f14a05ce1e8e4`.
