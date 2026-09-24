# Fast126 — FX H1 overextension continuation/reversion scout

Fast126 testou uma família diferente da regra USDJPY compression breakout:
operações intraday após estouros H1 raros, com duas leituras:

- `continuation`: seguir o movimento forte;
- `reversion`: fazer fade/reversão do movimento forte.

O runner usou M1 bid/ask, custo base e stress, e bloqueio non-overlap por
par/política: uma nova entrada só é aceita depois do fim da janela de hold da
entrada anterior.

Status:

`FAST126_FX_H1_OVEREXTENSION_FAILED_AUDIT`

## Protocolo

Universo:

- pares: `AUDUSD`, `EURGBP`, `EURJPY`, `EURUSD`, `GBPJPY`, `GBPUSD`,
  `NZDUSD`, `USDCAD`, `USDCHF`, `USDJPY`;
- dados: M1 local `backend/data/raw/fx_m1_hd`;
- sinal: H1 midpoint;
- execução: M1 bid/ask.

Grade:

- direção: `continuation`, `reversion`;
- lookback H1: `4`, `8`;
- overextension mínima: `3.0 ATR`, `3.5 ATR`;
- stop: `1.5 ATR`;
- hold máximo: `240`, `480` minutos;
- gestão: `fixed_1r`, `trail_0_5r_0_5r`.

Seleção:

- período de seleção: antes de `2024-01-01`;
- mínimo seleção: `40` trades;
- precisa média base > `0`;
- precisa média stress > `0`.

Audit:

- período audit: `2024-01-01` até `2026-09-01`;
- mínimo audit: `15` trades;
- mínimo anos audit: `2`;
- média base e stress positivas;
- LCB stress positivo com Bonferroni;
- participação de anos positivos >= `2/3`.

## Resultado

Números:

- candidatos: `320`;
- trades simulados: `236,396`;
- trades continuation: `118,198`;
- trades reversion: `118,198`;
- finalistas elegíveis na seleção: `1`;
- candidatos aprovados no audit: `0`.

Único finalista:

`USDJPY:h1_continuation_nonoverlap_lb8_thr3_atr1.5_hold480_fixed_1r`

Seleção:

- trades: `506`;
- média base: `+1.1908` pips/trade;
- média stress: `+0.1667` pips/trade;
- LCB stress: `-1.8597`;
- anos positivos: `2/5`.

Audit:

- trades: `315`;
- média base: `-2.8288` pips/trade;
- média stress: `-3.5135` pips/trade;
- LCB stress: `-6.8147`;
- anos positivos: `0/3`;
- `pilot_gate=false`.

## Interpretação

A família morreu. Reversão pós-estouro H1 não teve nenhum candidato positivo
após custo na seleção. Continuação teve só um candidato fraco em seleção e ele
quebrou forte no audit 2024–2026.

Isso reduz uma hipótese importante: "quando o mercado anda muito em H1, ou ele
continua ou reverte com edge simples". Com stop/hold/trailing simples e custo
stress, não apareceu base robusta.

## Segurança

Fast126 é pesquisa offline:

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
  `backend/scripts/research/run_fast126_fx_h1_overextension_scout.py`;
- teste:
  `backend/tests/unit/test_fast126_fx_h1_overextension_scout.py`;
- report JSON:
  `backend/data/lab_fast126/fx-h1-overextension-scout/report.json`;
- trades:
  `backend/data/lab_fast126/fx-h1-overextension-scout/overextension-trades.parquet`;
- seleção:
  `backend/data/lab_fast126/fx-h1-overextension-scout/selection-metrics.csv`;
- audit:
  `backend/data/lab_fast126/fx-h1-overextension-scout/audited-finalists.csv`;
- qualidade de pares:
  `backend/data/lab_fast126/fx-h1-overextension-scout/pair-quality.json`.

Hashes:

- report:
  `6903c03c269f5e27999854da7c07f1ef9b68bf75d9f916a1d3cd536ef97142a0`;
- trades:
  `8f178511266928732cdee266a5050b303286274f9d78f081592b6cb5af1c17e9`;
- seleção:
  `7f7baffb12c3d6630f878a4ec28eae4f5ccfa2611ca1c55f1e3e6b8267501d7f`;
- audit:
  `9bf7677eb16717babccb35904d75f2bb731512dcbd19c9563e3d2414ba062049`;
- pair quality:
  `1dc11165fc3ea86d9e930dc51b14ec02df56075b62e7e5e973059c34582dd119`.

## Próximo caminho

Fast126 sugere não insistir em H1 overextension simples. O caminho mais
defensável continua sendo:

1. observar prospectivamente a hipótese USDJPY compression breakout congelada
   no Fast124; ou
2. abrir uma família estrutural realmente diferente, preferencialmente com
   evento/agenda, sessão específica ou dado adicional, em vez de só derivar
   mais thresholds de preço.
