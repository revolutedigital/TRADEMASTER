# Fast144 — NZDUSD short-only neighborhood robustness audit

Fast144 respondeu a pergunta prática:

> A candidata `NZDUSD` vendida do Fast142 é uma regra minimamente robusta ou
> só um ponto exato encaixado no histórico?

Status:

`FAST144_NZDUSD_SHORT_ONLY_EDGE_REPLICATED_BUT_NARROW_REQUIRES_OBSERVATION`

## Resposta curta

Temos uma candidata estatística séria, mas ainda não temos estratégia aprovada.

A regra congelada passou novamente:

`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only`

E um vizinho direto de risco também passou:

`NZDUSD:d1_donchian_fade_ch100_hold20:stop1.5_fixed_1r:short_only`

Mas o edge ficou concentrado no desenho `channel=100` + `hold=20`. Nenhuma
variação de `channel/hold` passou junto. Portanto, a leitura honesta é:

- não é “não temos nada”;
- também não é “bot liberado”;
- é uma candidata promissora, estreita, já pré-registrada no Fast143, que
  precisa de observação prospectiva antes de Demo/Testnet/LIVE.

## Protocolo

Fontes:

- pré-2019 full grid:
  `backend/data/lab_fast137/fx-daily-donchian-fade-temporal-reverse/temporal-reverse-trades.parquet`;
- 2019–2026 full grid:
  `backend/data/lab_fast133/fx-daily-donchian-fade-global-risk/daily-donchian-fade-global-risk-trades.parquet`.

Hashes de entrada:

- pré-2019:
  `2ea0b3b650035529b0601f3f2d0559e25a1a1554a2da2bef98f28ac80b423bdb`;
- 2019–2026:
  `13bad134cd1da482eb69e4ad081daa255f5e0d2a685b94589a0fa1e5750d41da`.

Nota metodológica: o current do Fast137 tem apenas `100` finalistas selecionados
no pré-2019, então Fast144 usou o Fast133 para ter a grade atual completa dos
`27` vizinhos `NZDUSD`/`SHORT`/`fixed_1r`.

Grade auditada:

- par: `NZDUSD`;
- lado: `SHORT`;
- gestão: `fixed_1r`;
- canais: `20`, `55`, `100`;
- hold: `5`, `10`, `20`;
- stops: `1.0`, `1.5`, `2.0` ATR;
- total: `27` candidatos.

Gate por amostra:

- mínimo `12` trades;
- mínimo `3` anos;
- média base > `0`;
- média stress > `0`;
- LCB stress > `0`;
- anos positivos >= `2/3`.

O candidato só passa se passar no pré-2019 e no 2019–2026.

## Resultado agregado

| Métrica | Resultado |
|---|---:|
| Candidatos na grade | `27` |
| Passaram pré-2019 | `3` |
| Passaram 2019–2026 | `5` |
| Passaram ambos | `2` |
| Vizinhos locais | `8` |
| Vizinhos locais que passaram | `2` |
| Vizinhos de 1 passo | `4` |
| Vizinhos de 1 passo que passaram | `2` |
| Shapes `channel/hold` aprovados | `ch100_hold20` |

## Candidatos que passaram

| Candidato | Pré trades | Pré média | Pré LCB | Pré anos+ | Current trades | Current média | Current LCB | Current anos+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ch100_hold20:stop2_fixed_1r` | `19` | `+98.9480` | `+64.0934` | `5/5` | `29` | `+53.9022` | `+25.2283` | `6/7` |
| `ch100_hold20:stop1.5_fixed_1r` | `19` | `+69.5356` | `+36.2688` | `5/5` | `29` | `+31.5457` | `+6.2867` | `5/7` |

Detalhe da candidata congelada:

- pré-2019:
  - `19` trades;
  - stress total `+1880.0125` pips;
  - média stress `+98.9480` pips/trade;
  - LCB stress `+64.0934`;
  - anos positivos `5/5`;
  - top 3 trades positivos = `26.55%` do total;
- 2019–2026:
  - `29` trades;
  - stress total `+1563.1641` pips;
  - média stress `+53.9022` pips/trade;
  - LCB stress `+25.2283`;
  - anos positivos `6/7`;
  - top 3 trades positivos = `24.98%` do total.

## Interpretação

O Fast144 melhora a confiança em relação a um achado exato isolado, porque o
edge sobreviveu em `stop2` e também em `stop1.5`, mantendo mesmo canal, mesmo
hold e mesmo lado vendido.

Mas ele também reduz o entusiasmo: o edge não apareceu de forma ampla em
`channel=55`, `hold=10` ou `hold=5`. Isso quer dizer que a entrada parece estar
mais ligada ao regime de rompimento/fade de prazo mais longo (`100` dias + `20`
dias de hold) do que a uma verdade universal de Donchian fade.

Decisão:

- manter a candidata Fast143 em observação prospectiva;
- não aprovar Demo;
- não aprovar Testnet;
- não aprovar LIVE;
- não trocar a regra congelada por trailing;
- não expandir para `long_only` ou `all_sides`.

## Segurança

Fast144 é pesquisa offline:

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
  `backend/scripts/research/run_fast144_nzdusd_short_only_neighborhood_robustness.py`;
- teste:
  `backend/tests/unit/test_fast144_nzdusd_short_only_neighborhood_robustness.py`;
- report JSON:
  `backend/data/lab_fast144/nzdusd-short-only-neighborhood-robustness/report.json`;
- métricas:
  `backend/data/lab_fast144/nzdusd-short-only-neighborhood-robustness/candidate-neighborhood-metrics.csv`;
- candidatos aprovados:
  `backend/data/lab_fast144/nzdusd-short-only-neighborhood-robustness/passed-candidates.csv`;
- resumo:
  `backend/data/lab_fast144/nzdusd-short-only-neighborhood-robustness/neighborhood-summary.json`;
- avaliações completas:
  `backend/data/lab_fast144/nzdusd-short-only-neighborhood-robustness/candidate-neighborhood-evaluations.json`.

Hashes:

- report:
  `d7c8d364a582caca34fae86e142b03456b93cf1c7fa75e15a45b3253e824b56d`;
- métricas:
  `b2938579b6faf361ff43849ea0c3ed778cb8a45f1f9ed7c1589b8dcd1571ff4f`;
- candidatos aprovados:
  `fb3dcee312f5d6ac69f85d76e971870cf92dc65c10f353106bb0fedfe4de8ef1`;
- resumo:
  `c761588fe9e48fcdd8a934268ec746345de7b941b299960180fc607038074933`;
- avaliações:
  `8d98529709aaf0df9175eeed7028024236de6f804445022bf15921b2a3d6c812`.
