# Fast128 — FX daily Donchian breakout scout

Fast128 testou uma família de movimento maior: rompimentos Donchian em D1 com
saída por tempo depois de vários dias. A intenção foi reduzir o peso relativo
dos custos, atacando movimentos maiores que os intraday curtos.

Status:

`FAST128_FX_DAILY_DONCHIAN_FAILED_AUDIT`

## Protocolo

Universo:

- pares: `AUDUSD`, `EURGBP`, `EURJPY`, `EURUSD`, `GBPJPY`, `GBPUSD`,
  `NZDUSD`, `USDCAD`, `USDCHF`, `USDJPY`;
- dados: M1 local `backend/data/raw/fx_m1_hd`;
- sinal: D1 midpoint;
- execução: entrada no M1 posterior ao fechamento diário;
- saída: tempo fixo;
- primeira passada sem hard stop/trailing. Qualquer candidato positivo exigiria
  audit de risco separado antes de pré-registro.

Grade:

- canal Donchian: `20`, `55`, `100` dias;
- hold: `5`, `10`, `20` dias;
- candidatos totais: `90`.

Split:

- seleção: antes de `2024-01-01`;
- audit: `2024-01-01` até `2026-09-01`.

Critério:

- seleção: mínimo `30` trades, média base positiva e média stress positiva;
- audit: mínimo `12` trades, `2` anos, média base e stress positivas, LCB stress
  positivo com Bonferroni, anos positivos >= `2/3`.

## Resultado

Números:

- trades simulados: `8,314`;
- candidatos: `90`;
- finalistas elegíveis na seleção: `21`;
- candidatos aprovados no audit: `0`.

Top 5 no audit:

| Rank | Candidato | Seleção stress | Audit stress | LCB audit | Anos positivos audit |
|---:|---|---:|---:|---:|---:|
| 1 | `USDJPY:d1_donchian_ch55_hold20` | `+27.5937` | `-55.6081` | `-175.6461` | `1/3` |
| 2 | `GBPJPY:d1_donchian_ch55_hold10` | `+23.5003` | `-1.2208` | `-160.6419` | `2/3` |
| 3 | `USDJPY:d1_donchian_ch20_hold20` | `+20.9358` | `+34.1887` | `-67.4445` | `1/3` |
| 4 | `NZDUSD:d1_donchian_ch20_hold20` | `+16.1037` | `-38.3290` | `-75.2505` | `0/3` |
| 5 | `GBPJPY:d1_donchian_ch100_hold10` | `+13.4903` | `+4.9053` | `-169.4700` | `1/3` |

Nenhum passou o gate. Mesmo os que mantiveram média positiva no audit tinham
limite inferior muito negativo e/ou estabilidade anual ruim.

## Interpretação

Daily Donchian time-exit mostrou sinais bonitos em seleção, especialmente
`USDJPY` e `GBPJPY`, mas não resistiu ao audit 2024–2026. Isso indica mudança
de regime forte ou alta dependência de poucos movimentos grandes.

O resultado bloqueia a promoção para stop/trailing audit como estratégia. Ainda
pode haver aprendizado: `USDJPY d1_donchian_ch20_hold20` manteve média audit
positiva, mas com LCB negativo e só `1/3` anos positivos; não é robusto.

## Segurança

Fast128 é pesquisa offline:

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
  `backend/scripts/research/run_fast128_fx_daily_donchian_scout.py`;
- teste:
  `backend/tests/unit/test_fast128_fx_daily_donchian_scout.py`;
- report JSON:
  `backend/data/lab_fast128/fx-daily-donchian-scout/report.json`;
- trades:
  `backend/data/lab_fast128/fx-daily-donchian-scout/daily-donchian-trades.parquet`;
- seleção:
  `backend/data/lab_fast128/fx-daily-donchian-scout/selection-metrics.csv`;
- audit:
  `backend/data/lab_fast128/fx-daily-donchian-scout/audited-finalists.csv`;
- qualidade de pares:
  `backend/data/lab_fast128/fx-daily-donchian-scout/pair-quality.json`.

Hashes:

- report:
  `73bcc669e226fd348b29e103d76bcca6eb85699573c5faf48116c1d86763f197`;
- trades:
  `8f78d6df64adf177fb25a01ba1eed551a3b880033741bbc769b37d24e618160c`;
- seleção:
  `77511aea02e46367de8ff58535d7cac2ea13b6e29e392f8318f8e306e55b33a3`;
- audit:
  `5956d84d996da0e4c29c45c607bb7d44de587a65fca42602581aa6fc0d03f396`;
- pair quality:
  `1dc11165fc3ea86d9e930dc51b14ec02df56075b62e7e5e973059c34582dd119`.

## Próximo caminho

Não promover daily Donchian. A evidência atual continua apontando para:

1. observar prospectivamente as hipóteses já congeladas (`Fast78` FOMC e
   `Fast124` USDJPY compression breakout);
2. buscar uma família com informação exógena ou microestrutura adicional, não
   apenas preço OHLC derivado.
