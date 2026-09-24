# Fast146 — USD-quoted major basket audit

Fast146 testou uma hipótese mais operável do que `NZDUSD` isolado:

> Em vez de exigir que cada par replique sozinho, uma cesta causal pré-definida
> de pares USD-quoted passa no pré-2019 e no 2019–2026?

Status:

`FAST146_USD_QUOTED_MAJOR_BASKET_CANDIDATE_REQUIRES_PREREGISTRATION`

## Resposta curta

Sim: encontramos uma candidata melhor que o `NZDUSD` isolado.

Cesta:

`AUDUSD`, `EURUSD`, `GBPUSD`, `NZDUSD`

Política:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only`

Ela foi selecionada no pré-2019 e passou no 2019–2026. Ainda assim, não é
estratégia aprovada: nasceu de auditoria histórica e precisa de pré-registro
prospectivo próprio antes de qualquer Demo/Testnet/LIVE.

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

Regra comum:

- entrada: `d1_donchian_fade_ch100_hold20`;
- risco: `stop2_fixed_1r`;
- lado: `SHORT`;
- gestão: `fixed_1r`.

Grupos pré-definidos auditados:

- `all_10_pairs`;
- `usd_quoted_majors`;
- `oceanic_usd`;
- `europe_usd_quoted`;
- `commodity_linked`;
- `non_jpy`;
- `usd_base`;
- `jpy_crosses`;
- `eur_crosses`;
- `gbp_crosses`.

Gate por amostra:

- mínimo `30` trades;
- mínimo `3` anos;
- média base > `0`;
- média stress > `0`;
- LCB stress > `0`;
- anos positivos >= `2/3`.

A seleção do grupo é feita no pré-2019. Depois o grupo selecionado é auditado
no 2019–2026.

## Resultado agregado

| Métrica | Resultado |
|---|---:|
| Grupos avaliados | `10` |
| Grupos que passaram pré-2019 | `1` |
| Grupos que passaram 2019–2026 | `5` |
| Grupos que passaram ambos | `1` |
| Grupo selecionado no pré | `usd_quoted_majors` |
| Grupo aprovado pré + atual | `usd_quoted_majors` |

## Grupo aprovado

| Amostra | Trades | Pares | Média stress | Total stress | LCB stress | Anos+ |
|---|---:|---:|---:|---:|---:|---:|
| Pré-2019 | `89` | `4` | `+37.8272` | `+3366.6223` | `+9.7967` | `6/8` |
| 2019–2026 | `143` | `4` | `+42.3340` | `+6053.7689` | `+24.6986` | `7/8` |

Contribuição por par no pré-2019:

| Par | Trades | Média stress | Total stress |
|---|---:|---:|---:|
| `AUDUSD` | `10` | `+69.1874` | `+691.8744` |
| `EURUSD` | `32` | `-22.7896` | `-729.2659` |
| `GBPUSD` | `28` | `+54.4286` | `+1524.0014` |
| `NZDUSD` | `19` | `+98.9480` | `+1880.0125` |

Contribuição por par no 2019–2026:

| Par | Trades | Média stress | Total stress |
|---|---:|---:|---:|
| `AUDUSD` | `42` | `+52.1892` | `+2191.9455` |
| `EURUSD` | `35` | `+19.9759` | `+699.1564` |
| `GBPUSD` | `37` | `+43.2298` | `+1599.5029` |
| `NZDUSD` | `29` | `+53.9022` | `+1563.1641` |

## Interpretação

Esse é o achado mais útil até agora.

O `NZDUSD` isolado era forte, mas estreito. O Fast146 mostra que existe uma
forma mais estável de expressar o mesmo fenômeno: operar vendido em pares
USD-quoted após rompimento superior de canal longo, esperando fade/reversão.

O ponto fraco: no pré-2019, `EURUSD` foi negativo dentro da cesta. Mesmo assim,
a cesta total passou com LCB positivo e anos positivos suficientes. No período
atual, todos os quatro pares contribuíram positivamente.

Leitura honesta:

- há uma base estatística melhor que antes;
- ainda não é autorização de execução;
- a próxima ação correta é pré-registrar essa cesta e observar prospectivamente;
- se a observação prospectiva passar, essa cesta fica mais defensável para
  shadow/Testnet do que `NZDUSD` sozinho.

## Decisão

- promover `usd_quoted_majors` a candidata histórica principal;
- criar pré-registro prospectivo próprio;
- manter `research_only`;
- não Demo;
- não Testnet;
- não LIVE.

## Segurança

Fast146 é pesquisa offline:

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
  `backend/scripts/research/run_fast146_usd_quoted_major_basket_audit.py`;
- teste:
  `backend/tests/unit/test_fast146_usd_quoted_major_basket_audit.py`;
- report JSON:
  `backend/data/lab_fast146/usd-quoted-major-basket-audit/report.json`;
- métricas:
  `backend/data/lab_fast146/usd-quoted-major-basket-audit/group-basket-metrics.csv`;
- grupos aprovados:
  `backend/data/lab_fast146/usd-quoted-major-basket-audit/passed-groups.csv`;
- resumo:
  `backend/data/lab_fast146/usd-quoted-major-basket-audit/group-basket-summary.json`;
- avaliações:
  `backend/data/lab_fast146/usd-quoted-major-basket-audit/group-basket-evaluations.json`.

Hashes:

- report:
  `450971147b555a83151ec99bf9e5d50f26d99c68868a3257ff79170468cad6a2`;
- métricas:
  `e01e67c7758707592ac49de8926e0c47c2df1eb254dae063f733b8b578e0870b`;
- grupos aprovados:
  `8b5718a3165c36d0c71edf11c7a558a0b3f0166ec87d10b5e0575b9b9cf3ecde`;
- resumo:
  `6138a80846af96418c9f59616366528945d2c157c6a8e8c49c82c052e14e367f`;
- avaliações:
  `086509c1d61b6826328ebf094990346a379074b51c60cd42c2f58660f762a465`.
