# Fast127 — FX session drift scout

Fast127 testou uma família diferente de breakout/top-p/overextension:
drift direcional por par e hora UTC, com entrada fixa, direção fixa e saída por
tempo. É uma primeira passada sem filtro por weekday, para evitar explodir a
busca antes de aparecer uma pista base.

Status:

`FAST127_FX_SESSION_DRIFT_FAILED_AUDIT`

## Protocolo

Universo:

- pares: `AUDUSD`, `EURGBP`, `EURJPY`, `EURUSD`, `GBPJPY`, `GBPUSD`,
  `NZDUSD`, `USDCAD`, `USDCHF`, `USDJPY`;
- dados: M1 local `backend/data/raw/fx_m1_hd`;
- execução: bid/ask M1;
- custo: cenário base e stress.

Grade:

- horas UTC: `0`, `7`, `8`, `13`, `14`, `16`, `20`;
- holds: `240`, `480`, `1440` minutos;
- direção: `long`, `short`;
- weekday: `all` apenas nesta primeira passada;
- políticas por par: `42`;
- candidatos totais: `420`.

Split:

- seleção: antes de `2024-01-01`;
- audit: `2024-01-01` até `2026-09-01`.

Critério:

- seleção: mínimo `80` trades, média base positiva e média stress positiva;
- audit: mínimo `30` trades, `2` anos, média base e stress positivas, LCB stress
  positivo com Bonferroni, anos positivos >= `2/3`.

## Resultado

Números:

- trades simulados: `715,510`;
- candidatos: `420`;
- finalistas elegíveis na seleção: `5`;
- candidatos aprovados no audit: `0`.

Top 3 finalistas no audit:

| Rank | Candidato | Seleção stress | Audit stress | LCB audit | Anos positivos audit |
|---:|---|---:|---:|---:|---:|
| 1 | `EURUSD:utc13_wdall_hold1440_short` | `+0.2797` | `-3.0291` | `-7.7513` | `1/3` |
| 2 | `EURUSD:utc14_wdall_hold1440_short` | `+0.0854` | `-2.6802` | `-7.3527` | `1/3` |
| 3 | `EURUSD:utc16_wdall_hold480_long` | `+0.0825` | `-0.0231` | `-2.2924` | `2/3` |

O melhor padrão aparente foi short EURUSD à tarde UTC com hold de 24h, mas ele
inverteu no audit.

## Interpretação

Session drift simples não é estratégia. A seleção encontrou micro-médias
positivas, mas nenhuma sobreviveu ao período 2024–2026.

O resultado também bloqueia uma expansão óbvia: não vale abrir uma grade maior
por weekday antes de existir sinal base em `all weekdays`, porque isso só
aumentaria risco de overfit.

## Segurança

Fast127 é pesquisa offline:

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
  `backend/scripts/research/run_fast127_fx_session_drift_scout.py`;
- teste:
  `backend/tests/unit/test_fast127_fx_session_drift_scout.py`;
- report JSON:
  `backend/data/lab_fast127/fx-session-drift-scout/report.json`;
- trades:
  `backend/data/lab_fast127/fx-session-drift-scout/session-drift-trades.parquet`;
- seleção:
  `backend/data/lab_fast127/fx-session-drift-scout/selection-metrics.csv`;
- audit:
  `backend/data/lab_fast127/fx-session-drift-scout/audited-finalists.csv`;
- qualidade de pares:
  `backend/data/lab_fast127/fx-session-drift-scout/pair-quality.json`.

Hashes:

- report:
  `1bd8f168035f960a5b3f9675ab01f85062dc6c7924262e1eac838baaf0048326`;
- trades:
  `43cfbd4071c54a85dbbd97a3d1b913636cd392255f7f4b1426967bb7483febc0`;
- seleção:
  `66ab2af496a8882244ce8e3fb0665d67843c70322d86180d286a7c2cbf7953e8`;
- audit:
  `e10278064a83487bbcbc2a6d6b8597103a21fa8057eb913757506b8e7c297272`;
- pair quality:
  `1dc11165fc3ea86d9e930dc51b14ec02df56075b62e7e5e973059c34582dd119`.

## Próximo caminho

Não insistir em drift horário simples. Depois de Fast126 e Fast127, a evidência
local favorece duas trilhas:

1. observar prospectivamente as hipóteses já congeladas (`Fast78` FOMC e
   `Fast124` USDJPY compression breakout);
2. procurar uma família nova com informação exógena ou estrutural, não apenas
   preço + horário.
