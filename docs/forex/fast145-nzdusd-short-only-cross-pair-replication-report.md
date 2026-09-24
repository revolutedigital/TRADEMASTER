# Fast145 — NZDUSD short-only cross-pair replication audit

Fast145 fez a próxima falsificação da candidata:

> A regra exata `NZDUSD` short-only também aparece em outros pares, ou é um edge
> específico de `NZDUSD`?

Status:

`FAST145_NZDUSD_SHORT_ONLY_CROSS_PAIR_NOT_REPLICATED_REQUIRES_OBSERVATION`

## Resposta curta

A candidata `NZDUSD` continua viva, mas **não replicou cross-pair**.

Na regra exata:

`d1_donchian_fade_ch100_hold20:stop2_fixed_1r:SHORT`

somente `NZDUSD` passou no pré-2019 e no 2019–2026.

`AUDUSD` é um lead interessante porque passou no período atual, mas falhou no
pré-2019 por número baixo de trades e LCB negativo. Portanto, ele não conta como
replicação.

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

Grade:

- pares: `AUDUSD`, `EURGBP`, `EURJPY`, `EURUSD`, `GBPJPY`, `GBPUSD`,
  `NZDUSD`, `USDCAD`, `USDCHF`, `USDJPY`;
- entrada: `d1_donchian_fade_ch100_hold20`;
- risco: `stop2_fixed_1r`;
- lado: `SHORT`;
- gestão: `fixed_1r`.

Gate por amostra:

- mínimo `12` trades;
- mínimo `3` anos;
- média base > `0`;
- média stress > `0`;
- LCB stress > `0`;
- anos positivos >= `2/3`.

O par só passa se passar no pré-2019 e no 2019–2026.

## Resultado agregado

| Métrica | Resultado |
|---|---:|
| Pares avaliados | `10` |
| Passaram pré-2019 | `1` |
| Passaram 2019–2026 | `2` |
| Passaram ambos | `1` |
| Pares aprovados | `NZDUSD` |
| Pares current-only | `AUDUSD` |

## Pares principais

| Par | Pré trades | Pré média | Pré LCB | Pré anos+ | Current trades | Current média | Current LCB | Current anos+ | Passou ambos? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `NZDUSD` | `19` | `+98.9480` | `+64.0934` | `5/5` | `29` | `+53.9022` | `+25.2283` | `6/7` | sim |
| `AUDUSD` | `10` | `+69.1874` | `-9.4019` | `5/5` | `42` | `+52.1892` | `+26.9628` | `6/8` | não |
| `GBPUSD` | `28` | `+54.4286` | `-8.5976` | `5/7` | `37` | `+43.2298` | `-2.7634` | `6/8` | não |
| `USDCHF` | `44` | `-0.4936` | `-36.5939` | `6/8` | `30` | `+22.6237` | `-13.4606` | `3/5` | não |
| `USDCAD` | `54` | `-25.2803` | `-62.9278` | `4/7` | `36` | `+12.0592` | `-34.2114` | `3/7` | não |

## Interpretação

Esse resultado é bom e ruim ao mesmo tempo.

Bom:

- `NZDUSD` segue passando em pré e atual;
- `AUDUSD` sugere que pode haver uma família de comportamento em pares commodity
  contra USD, mas ainda sem prova no pré;
- o achado de Fast142/Fast144 não foi negado pela auditoria cross-pair.

Ruim:

- não existe replicação limpa em outro par;
- o edge parece específico de `NZDUSD`, não uma regra geral de FX;
- isso reduz a confiança para transformar a candidata em estratégia de execução.

Leitura honesta:

`NZDUSD short-only ch100 hold20 stop2 fixed1r` continua sendo nossa melhor
candidata, mas agora classificada como **edge específico de instrumento**. Isso
exige ainda mais disciplina no pré-registro prospectivo do Fast143.

## Decisão

- manter Fast143 congelado em observação;
- não aprovar Demo;
- não aprovar Testnet;
- não aprovar LIVE;
- não expandir automaticamente para AUDUSD;
- não tratar isso como estratégia genérica de Donchian fade.

## Segurança

Fast145 é pesquisa offline:

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
  `backend/scripts/research/run_fast145_nzdusd_short_only_cross_pair_replication.py`;
- teste:
  `backend/tests/unit/test_fast145_nzdusd_short_only_cross_pair_replication.py`;
- report JSON:
  `backend/data/lab_fast145/nzdusd-short-only-cross-pair-replication/report.json`;
- métricas:
  `backend/data/lab_fast145/nzdusd-short-only-cross-pair-replication/cross-pair-metrics.csv`;
- pares aprovados:
  `backend/data/lab_fast145/nzdusd-short-only-cross-pair-replication/passed-pairs.csv`;
- resumo:
  `backend/data/lab_fast145/nzdusd-short-only-cross-pair-replication/cross-pair-summary.json`;
- avaliações:
  `backend/data/lab_fast145/nzdusd-short-only-cross-pair-replication/cross-pair-evaluations.json`.

Hashes:

- report:
  `e133b2dbdd9814f80fcd411279f7f6b082811a97aa7acf32a3bbb49ec01750e2`;
- métricas:
  `922883c20a208a6a00b9e5679a9511c3473048789bc29dd798a7651e2c63e671`;
- pares aprovados:
  `cc3f44d42c8eafd39f60fb41e34be9cb1762dc592886dfcc8a5de54024ea12bf`;
- resumo:
  `f1dc8973d802ae8b01495f7c62babfe4629f0e7b12efe863cecb95362c39c358`;
- avaliações:
  `70924b71570e7428ede6b5bfe4831267b6958d50ae27746e0f1acee12aa04c0d`.
