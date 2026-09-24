# Fast132 — FX intraday Donchian fade scout

Fast132 testou a tentativa de trazer a pista do Fast129/Fast130 para trades
mais curtos. Em vez de D1/NZDUSD com hold de até `20` dias, o scout testou fade
de rompimentos Donchian em H1 e H4, com stop e alvo, em todos os pares FX do
laboratório.

Status:

`FAST132_FX_INTRADAY_DONCHIAN_FADE_FAILED_AUDIT`

## Protocolo

Universo:

- pares: `AUDUSD`, `EURGBP`, `EURJPY`, `EURUSD`, `GBPJPY`, `GBPUSD`,
  `NZDUSD`, `USDCAD`, `USDCHF`, `USDJPY`;
- dados: M1 local `backend/data/raw/fx_m1_hd`;
- sinal: midpoint H1/H4;
- execução: M1 bid/ask;
- direção: fade/contra o rompimento.

Grade:

- timeframes: `1h`, `4h`;
- canal Donchian: `20`, `55`, `100` barras;
- hold máximo: `6`, `12` barras;
- stop: `1.0 ATR`, `1.5 ATR`;
- gestão: `fixed_1r`, `fixed_2r`.

Split:

- seleção: antes de `2024-01-01`;
- audit: `2024-01-01` até `2026-09-01`.

Gates:

- mínimo seleção: `60` trades;
- seleção precisa média base > `0` e média stress > `0`;
- mínimo audit: `20` trades;
- mínimo audit years: `2`;
- audit precisa média base/stress positiva;
- LCB stress precisa ser positivo com Bonferroni;
- anos positivos em stress precisam ser pelo menos `2/3`.

## Resultado

Números:

- candidatos: `480`;
- trades simulados: `216,056`;
- trades H1: `210,056`;
- trades H4: `6,000`;
- finalistas elegíveis na seleção: `3`;
- candidatos aprovados no audit: `0`.

Finalistas:

1. `GBPUSD:1h_donchian_fade_ch55_stop1.5_hold12b_fixed_2r`
   - seleção: `391` trades, média stress `+0.8668` pips/trade;
   - audit: `154` trades, média stress `-4.9603` pips/trade, LCB
     `-10.2697`, anos positivos `1/3`;
   - `pilot_gate=false`.
2. `GBPUSD:4h_donchian_fade_ch20_stop1_hold6b_fixed_2r`
   - seleção: `96` trades, média stress `+0.7258` pips/trade;
   - audit: `3` trades, média stress `+0.9992` pips/trade, amostra
     insuficiente, LCB `-177.5912`;
   - `pilot_gate=false`.
3. `GBPUSD:1h_donchian_fade_ch100_stop1.5_hold12b_fixed_2r`
   - seleção: `273` trades, média stress `+0.6755` pips/trade;
   - audit: `108` trades, média stress `-7.4201` pips/trade, LCB
     `-13.3158`, anos positivos `0/3`;
   - `pilot_gate=false`.

## Interpretação

A versão curta da tese Donchian fade não sobreviveu. O melhor que apareceu foi
GBPUSD H1/H4 com lucro pequeno na seleção, mas o audit independente 2024–2026
derrubou a família:

- H1 gerou bastante amostra, mas virou negativo no audit;
- H4 gerou pouca amostra demais;
- todos os LCBs ficaram negativos;
- nenhum candidato passou o gate estatístico.

Isso reforça a leitura atual: a pista robusta encontrada até agora está no
timeframe diário (`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_2r`), não
na versão intraday curta.

## Segurança

Fast132 é pesquisa offline:

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
  `backend/scripts/research/run_fast132_fx_intraday_donchian_fade_scout.py`;
- teste:
  `backend/tests/unit/test_fast132_fx_intraday_donchian_fade_scout.py`;
- report JSON:
  `backend/data/lab_fast132/fx-intraday-donchian-fade-scout/report.json`;
- trades:
  `backend/data/lab_fast132/fx-intraday-donchian-fade-scout/intraday-donchian-fade-trades.parquet`;
- seleção:
  `backend/data/lab_fast132/fx-intraday-donchian-fade-scout/selection-metrics.csv`;
- audit:
  `backend/data/lab_fast132/fx-intraday-donchian-fade-scout/audited-finalists.csv`;
- qualidade de pares:
  `backend/data/lab_fast132/fx-intraday-donchian-fade-scout/pair-quality.json`.

Hashes:

- report:
  `c91eb795beb4ce657395d994c5706dae37c179482cf0b2664d3bc79bf07d7bf7`;
- trades:
  `bddfdb9bb20cce3828f6f0269bc4c3185962ff5b5369b33bce04206dd3cd8df3`;
- seleção:
  `e5cca71044b60f490d355ae5f0453854f8dc06578ec8edad27b5811a405872c3`;
- audit:
  `932f3fd0ac4cff4ae5b87ccec72413b62755a13574638358cb6ada2860232275`;
- pair quality:
  `1dc11165fc3ea86d9e930dc51b14ec02df56075b62e7e5e973059c34582dd119`.

## Próximo caminho

Não repetir Donchian fade intraday simples sem nova informação. Próximas linhas
com melhor relação sinal/tempo:

1. observar o Fast131 com dado realmente pós-pré-registro;
2. testar filtros causais sobre a candidata D1 sem alterar a regra congelada do
   Fast131;
3. se insistirmos em trades curtos, mudar a fonte de edge, não só comprimir o
   Donchian diário para H1/H4.
