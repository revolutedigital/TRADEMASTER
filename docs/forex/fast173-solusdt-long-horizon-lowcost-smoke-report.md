# Fast173 — SOLUSDT long-horizon low-cost smoke

Fast173 continuou a rota aberta pelo Fast172: testar se um ativo de beta maior
e mais volátil que BTC/ETH melhora a família low-cost/long-horizon.

Status:

`FAST173_SOLUSDT_LONG_HORIZON_LOWCOST_FAILED_WALKFORWARD_OR_EXTENSION`

## Resposta curta

SOLUSDT também não resolveu.

O screen achou `7` candidatos positivos, mas todos falharam na validação
walk-forward/extensão. Pior: nos 7 casos, o bloco histórico validado agregado
ficou negativo.

## Dados

Fonte:

- símbolo: `SOLUSDT`;
- produto: Binance USD-M perpetual public archive;
- período baixado: `2026-08-30` até `2026-09-22`;
- partições: `24`;
- qualidade: todas as partições `VALID`, com `0` sequence gaps;
- funding rows: `73`;
- mark-price 1m rows: `34,561`.

O root foi isolado em:

`backend/data/microstructure_solusdt_v1`

Esse diretório é dado gerado/ignorado pelo Git e não sobrescreve o BTCUSDT
canônico em `backend/data/microstructure_v1`.

## Protocolo

Runner reaproveitado:

`backend/scripts/research/run_fast106_long_horizon_lowcost_scout.py`

Parâmetros:

- historical root limit: `4`;
- max days per root: `2`;
- horizontes: `900`, `1800` segundos;
- targets: `8`, `12`, `16`, `24` bps;
- top fractions: `0.005`, `0.01`, `0.02`, `0.05`;
- custo explícito default Fast106: `2.0` bps;
- stop inicial default Fast106: `20.0` bps.

Comando executado:

```bash
backend/.venv/bin/python backend/scripts/research/build_microstructure_dataset.py \
  --symbol SOLUSDT \
  --market usdm_perpetual \
  --start-date 2026-08-30 \
  --end-date 2026-09-22 \
  --kinds aggTrades \
  --root backend/data/microstructure_solusdt_v1

backend/.venv/bin/python backend/scripts/research/run_fast106_long_horizon_lowcost_scout.py \
  --data-root backend/data/microstructure_solusdt_v1 \
  --trade-root backend/data/microstructure_solusdt_v1/normalized/aggTrades \
  --output-root backend/data/microstructure_solusdt_v1/reports/fast173-solusdt-long-horizon-lowcost-smoke \
  --historical-root-limit 4 \
  --max-days-per-root 2 \
  --progress
```

## Resultado

Status bruto do runner:

`FAST106_LONG_HORIZON_FAILED_WALKFORWARD_OR_EXTENSION`

Resumo:

| Métrica | Valor |
|---|---:|
| Specs avaliadas | `32` |
| Screen survivors | `7` |
| Validações executadas | `7` |
| Survivors finais | `0` |

Validações dos 7 screen survivors:

| Horizonte | Target | Top fraction | Historical net | Blocos negativos | Extension net | Extension trades |
|---:|---:|---:|---:|---|---:|---:|
| `900s` | `8` | `0.020` | `-14.8897` | `fast51`, `fast52` | `+16.6235` | `29` |
| `900s` | `16` | `0.005` | `-60.8557` | `fast50`, `fast51`, `fast53` | `+42.0000` | `3` |
| `900s` | `16` | `0.010` | `-111.7028` | `fast50`, `fast51`, `fast53` | `+67.6971` | `11` |
| `900s` | `16` | `0.020` | `-107.7398` | `fast50`, `fast51` | `+29.8144` | `21` |
| `900s` | `24` | `0.020` | `-249.0486` | `fast50`, `fast51`, `fast52`, `fast53` | `+6.1602` | `20` |
| `1800s` | `16` | `0.010` | `-86.9732` | `fast50`, `fast53` | `+22.0000` | `17` |
| `1800s` | `24` | `0.050` | `-465.6041` | `fast50`, `fast51`, `fast52`, `fast53` | `+34.3382` | `36` |

## Interpretação

O resultado é importante porque ele separa duas coisas:

- "ativo mais volátil" até ajuda a gerar candidatos bonitos no screen;
- mas não resolve a estabilidade temporal.

No SOLUSDT, a extensão recente até ficou positiva em todos os screen survivors,
mas o walk-forward histórico que deveria proteger contra miragem ficou negativo
em todos. Isso sugere que o modelo está escolhendo regimes recentes que não
replicam no passado imediato, ou seja, ainda é ajuste de janela.

Decisão:

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Depois de BTCUSDT, ETHUSDT e SOLUSDT, a hipótese "mesmo low-cost/top-p em outro
par resolve" fica fraca. Se continuarmos microstructure, a próxima tentativa
precisa mudar o tipo de informação: regime pré-entrada, depth/book/fill, ou
outro mercado com custo/fill estruturalmente diferente.

## Segurança

Fast173 é pesquisa offline:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `dry_run=true`.

Nenhum trade foi enviado. Nenhum ledger foi alterado.

## Artefatos

- dataset manifest:
  `backend/data/microstructure_solusdt_v1/dataset-manifest.json`;
- report:
  `backend/data/microstructure_solusdt_v1/reports/fast173-solusdt-long-horizon-lowcost-smoke/report.json`;
- screen results:
  `backend/data/microstructure_solusdt_v1/reports/fast173-solusdt-long-horizon-lowcost-smoke/screen_results.json`;
- validations:
  `backend/data/microstructure_solusdt_v1/reports/fast173-solusdt-long-horizon-lowcost-smoke/validations.json`;
- row manifest:
  `backend/data/microstructure_solusdt_v1/reports/fast173-solusdt-long-horizon-lowcost-smoke/row_manifest.json`.

Hashes:

- dataset manifest:
  `fb23cb03c01aedadec854550884994b5e1fd3187a49f9b52d7d7baccc529ef75`;
- partition manifest:
  `10cdb72da2874a9b078de83f4b6e6ba56004ceee56f58fbadea11c1db18b06e0`;
- report:
  `2cfeaae745b652ef8c9be9ee76cfb42144569140e8e65858fe51f302a636fc41`;
- screen results:
  `98b2ad7b60814332b9850fbcf56bb4f728690d396e6a4ef89e167748cd23f2a0`;
- validations:
  `5463d48096a359cd1c23a3a22d3857111482a21f63fbdc5bc6100391b7bc2a77`;
- row manifest:
  `ff966d162803a78b13d45bf1d5db7bd3ea19253b33a8bd32750cfb2db840d877`.
