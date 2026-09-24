# Fast129 — FX daily Donchian fade scout

Fast129 testou a contraparte do Fast128. Em vez de seguir rompimentos Donchian
em D1, ele fez fade/reversão desses rompimentos e segurou por alguns dias.

Status:

`FAST129_FX_DAILY_DONCHIAN_FADE_RESEARCH_CANDIDATE_REQUIRES_RISK_AUDIT`

## Protocolo

Universo:

- pares: `AUDUSD`, `EURGBP`, `EURJPY`, `EURUSD`, `GBPJPY`, `GBPUSD`,
  `NZDUSD`, `USDCAD`, `USDCHF`, `USDJPY`;
- dados: M1 local `backend/data/raw/fx_m1_hd`;
- sinal: D1 midpoint;
- execução: entrada no M1 posterior ao fechamento diário;
- direção: contra o rompimento;
- saída: tempo fixo.

Grade:

- canal Donchian: `20`, `55`, `100` dias;
- hold: `5`, `10`, `20` dias;
- candidatos totais: `90`.

Split:

- seleção: antes de `2024-01-01`;
- audit: `2024-01-01` até `2026-09-01`.

## Resultado

Números:

- trades simulados: `8,314`;
- candidatos: `90`;
- finalistas elegíveis na seleção: `50`;
- candidatos aprovados no audit: `1`.

Candidato aprovado:

`NZDUSD:d1_donchian_fade_ch100_hold20`

Seleção:

- trades: `44`;
- média base: `+42.2922` pips/trade;
- média stress: `+40.5638` pips/trade;
- LCB stress seleção: `-2.0733`;
- anos positivos: `4/5`.

Audit:

- trades: `21`;
- média base: `+62.8242` pips/trade;
- média stress: `+61.3837` pips/trade;
- LCB stress Bonferroni: `+1.3387`;
- anos positivos: `3/3`.

## Interpretação

Esta é a primeira família recente que não morreu na auditoria. Mas o Fast129
ainda era time-exit sem hard stop, então ele não poderia virar estratégia. O
resultado correto foi promover para Fast130, um audit de risco com stop/target/
trailing.

## Segurança

Fast129 é pesquisa offline:

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
  `backend/scripts/research/run_fast129_fx_daily_donchian_fade_scout.py`;
- teste:
  `backend/tests/unit/test_fast129_fx_daily_donchian_fade_scout.py`;
- report JSON:
  `backend/data/lab_fast129/fx-daily-donchian-fade-scout/report.json`;
- trades:
  `backend/data/lab_fast129/fx-daily-donchian-fade-scout/daily-donchian-fade-trades.parquet`;
- seleção:
  `backend/data/lab_fast129/fx-daily-donchian-fade-scout/selection-metrics.csv`;
- audit:
  `backend/data/lab_fast129/fx-daily-donchian-fade-scout/audited-finalists.csv`;
- qualidade de pares:
  `backend/data/lab_fast129/fx-daily-donchian-fade-scout/pair-quality.json`.

Hashes:

- report:
  `d6ae2769b745be9d72ead8bd0f24aa6ddfde19092e2a31b919931a1a5c94051f`;
- trades:
  `886158b88d7cf3c0131de9960258595a45838eb93b5fc94db10a97b0b6895157`;
- seleção:
  `273a451240fefa39301cbb701cb91ba95eab5428224951f134bfdf233e2fda3b`;
- audit:
  `b9a41e40d80dd57dcc265ee071d3012ced96da81709a307999ec767c184cac14`;
- pair quality:
  `1dc11165fc3ea86d9e930dc51b14ec02df56075b62e7e5e973059c34582dd119`.

## Próximo passo

Fast130 deve congelar o candidato `NZDUSD:d1_donchian_fade_ch100_hold20` e
testar stop/target/trailing com risco causal. Se sobreviver, ainda precisará de
pré-registro prospectivo antes de qualquer execução.
