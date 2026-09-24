# Fast142 — NZDUSD Donchian fade side-filter audit

Fast142 pegou a candidata Fast134/Fast135 e fez uma pergunta cronológica
simples:

> Se eu escolho o lado da operação usando apenas o pré-2019, esse filtro
> sobrevive em 2019–2026?

Status:

`FAST142_NZDUSD_SHORT_ONLY_SIDE_FILTER_CANDIDATE_REQUIRES_PREREGISTRATION`

## Regra base

`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r`

- par: `NZDUSD`;
- sinal: D1 Donchian fade;
- canal: `100` dias;
- hold máximo: `20` dias;
- stop: `2.0 * ATR(14)` diário causal;
- alvo: `1R`;
- custos: cenário `STRESS`;
- side filter auditado: `all_sides`, `long_only`, `short_only`.

## Protocolo

Fonte:

`backend/data/lab_fast135/nzdusd-donchian-fade-fixed1r-temporal-audit/temporal-audit-trades.parquet`

Hash:

`a4de5c3a9cd0502c814cf423aee65239deb2fb01aae0654b1ae58f47fe5f63a6`

Separação:

- seleção: `pre_2011_2018`;
- audit: `current_2019_2026`.

Gates:

- seleção: mínimo `12` trades, `3` anos, média stress > `0`, LCB stress > `0`,
  pelo menos `2/3` dos anos positivos;
- audit: mínimo `12` trades, `3` anos, média stress > `0`, LCB stress > `0`,
  pelo menos `2/3` dos anos positivos;
- seleção do filtro acontece antes do audit atual.

## Resultado

| Filtro | Seleção trades | Seleção média | Seleção LCB | Seleção anos+ | Audit trades | Audit média | Audit LCB | Audit anos+ | Passou? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `all_sides` | `62` | `+17.5648` | `-12.5979` | `6/8` | `65` | `+45.2657` | `+21.8566` | `7/8` | não |
| `long_only` | `43` | `-18.3952` | `-56.2293` | `5/8` | `36` | `+38.3085` | `+1.9418` | `6/8` | não |
| `short_only` | `19` | `+98.9480` | `+64.0934` | `5/5` | `29` | `+53.9022` | `+25.2283` | `6/7` | sim |

O melhor filtro aprovado:

`short_only`

Detalhes:

- seleção pré-2019:
  - `19` trades;
  - stress total `+1880.0125` pips;
  - média `+98.9480` pips/trade;
  - LCB `+64.0934`;
  - anos positivos `5/5`;
  - top 3 trades positivos respondem por `26.55%` do total;
- audit 2019–2026:
  - `29` trades;
  - stress total `+1563.1642` pips;
  - média `+53.9022` pips/trade;
  - LCB `+25.2283`;
  - anos positivos `6/7`;
  - top 3 trades positivos respondem por `24.98%` do total.

## Interpretação

Esse é um achado melhor que o Fast134 bruto:

- o `all_sides` era bom em 2019–2026, mas não tinha LCB positivo no pré-2019;
- o `long_only` falha claramente no pré-2019;
- o `short_only` passa no pré-2019 e também no período atual;
- a contribuição não está concentrada em 1 ou 2 trades;
- o filtro é causal e simples: operar apenas o lado `SHORT`, ou seja, fade
  vendido após rompimento superior do canal Donchian.

Leitura honesta: encontramos uma candidata histórica mais limpa. Mas ela ainda
não é estratégia aprovada porque a família de entrada nasceu de pesquisa
histórica, e este side-filter também foi descoberto/auditado offline. Precisa de
pré-registro prospectivo próprio antes de qualquer shadow/Testnet.

## Segurança

Fast142 é pesquisa offline:

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
  `backend/scripts/research/run_fast142_nzdusd_donchian_fade_side_filter.py`;
- teste:
  `backend/tests/unit/test_fast142_nzdusd_donchian_fade_side_filter.py`;
- report JSON:
  `backend/data/lab_fast142/nzdusd-donchian-fade-side-filter/report.json`;
- avaliações:
  `backend/data/lab_fast142/nzdusd-donchian-fade-side-filter/side-filter-evaluations.json`;
- filtros aprovados:
  `backend/data/lab_fast142/nzdusd-donchian-fade-side-filter/passed-side-filters.json`.

Hashes:

- report:
  `b0f0cb45b717ec22d4caee6ed570d7cf0e110c222002800ad8a50abeb5d714f6`;
- avaliações:
  `41d353edd491f6eee93c6e6c989ac7f8f6dc6dd21c447a3ca65bf1720f6508f8`;
- filtros aprovados:
  `c07c1fc17ee7e9ea7d402966bd18a304d1e54b51efe540df2d4b0e1f00bd2d87`.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Criar pré-registro prospectivo para:

`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only`

Essa é agora a candidata mais limpa da linha NZDUSD Donchian fade.
