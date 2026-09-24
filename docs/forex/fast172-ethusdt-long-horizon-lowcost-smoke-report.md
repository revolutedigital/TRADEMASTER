# Fast172 — ETHUSDT long-horizon low-cost smoke

Fast172 testou a recomendação que sobrou depois das falhas em BTCUSDT
microstructure: trocar o mercado/par, mantendo o desenho de menos decisões,
alvos maiores e horizonte mais longo.

Status:

`FAST172_ETHUSDT_LONG_HORIZON_LOWCOST_FAILED_WALKFORWARD_OR_EXTENSION`

## Resposta curta

ETHUSDT não resolveu o problema no smoke reduzido.

O screen achou `5` candidatos aparentemente positivos, mas todos falharam na
validação walk-forward/extensão. Isso mata a versão curta da hipótese
"o mesmo modelo low-cost funciona melhor em ETHUSDT".

## Dados

Fonte:

- símbolo: `ETHUSDT`;
- produto: Binance USD-M perpetual public archive;
- período baixado: `2026-08-30` até `2026-09-22`;
- partições: `24`;
- qualidade: todas as partições `VALID`, com `0` sequence gaps;
- funding rows: `73`;
- mark-price 1m rows: `34,561`.

O root foi isolado em:

`backend/data/microstructure_ethusdt_v1`

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
  --symbol ETHUSDT \
  --market usdm_perpetual \
  --start-date 2026-08-30 \
  --end-date 2026-09-22 \
  --kinds aggTrades \
  --root backend/data/microstructure_ethusdt_v1

backend/.venv/bin/python backend/scripts/research/run_fast106_long_horizon_lowcost_scout.py \
  --data-root backend/data/microstructure_ethusdt_v1 \
  --trade-root backend/data/microstructure_ethusdt_v1/normalized/aggTrades \
  --output-root backend/data/microstructure_ethusdt_v1/reports/fast172-ethusdt-long-horizon-lowcost-smoke \
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
| Screen survivors | `5` |
| Validações executadas | `5` |
| Survivors finais | `0` |

Validações dos 5 screen survivors:

| Horizonte | Target | Top fraction | Historical net | Blocos negativos | Extension net | Extension trades |
|---:|---:|---:|---:|---|---:|---:|
| `900s` | `8` | `0.005` | `-74.0000` | `fast50`, `fast52`, `fast53` | `+22.0000` | `13` |
| `900s` | `12` | `0.005` | `-50.0183` | `fast50`, `fast52`, `fast53` | `+26.0000` | `9` |
| `900s` | `16` | `0.005` | `+116.7390` | `fast53` | `+20.7572` | `16` |
| `900s` | `24` | `0.005` | `+46.0970` | `fast52`, `fast53` | `+49.0785` | `21` |
| `900s` | `24` | `0.010` | `+79.5588` | `fast51`, `fast52`, `fast53` | `+48.9554` | `38` |

O melhor caso visualmente tentador (`900s`, target `16`, top `0.005`) ainda
tem um bloco suficiente negativo (`fast53`). Pelo padrão do projeto, isso não
pode virar candidato: é justamente o tipo de instabilidade temporal que vem
destruindo os edges curtos.

## Interpretação

Trocar BTCUSDT por ETHUSDT melhorou o screen superficial, mas não resolveu o
ponto central:

- o modelo ainda encontra bolsões positivos;
- esses bolsões não ficam estáveis entre blocos recentes;
- custo + regime continuam quebrando a replicação;
- trailing/top-p sem regime prévio continua insuficiente.

Decisão:

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast172 fecha o smoke curto de "mesmo desenho low-cost em ETHUSDT". Se formos
continuar microstructure, a próxima tentativa precisa mudar mais do que o par:
regime pré-entrada, book/fill/slippage mais rico, ou outro mercado com estrutura
de custo muito diferente.

## Segurança

Fast172 é pesquisa offline:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `dry_run=true`.

Nenhum trade foi enviado. Nenhum ledger foi alterado.

## Artefatos

- dataset manifest:
  `backend/data/microstructure_ethusdt_v1/dataset-manifest.json`;
- report:
  `backend/data/microstructure_ethusdt_v1/reports/fast172-ethusdt-long-horizon-lowcost-smoke/report.json`;
- screen results:
  `backend/data/microstructure_ethusdt_v1/reports/fast172-ethusdt-long-horizon-lowcost-smoke/screen_results.json`;
- validations:
  `backend/data/microstructure_ethusdt_v1/reports/fast172-ethusdt-long-horizon-lowcost-smoke/validations.json`;
- row manifest:
  `backend/data/microstructure_ethusdt_v1/reports/fast172-ethusdt-long-horizon-lowcost-smoke/row_manifest.json`.

Hashes:

- dataset manifest:
  `a8be1f23cafb75ab1b28f394ebeca626ffd6cdf10eb6d10bb445c163458080b7`;
- partition manifest:
  `2113db923a6c332ed4afc535a81202244b39c4517c6def6b5f3b92947e86eea5`;
- report:
  `8feee2776689b0160181c91d1af8f2ebc5c4ac67333e0cded465fcd5169aa7eb`;
- screen results:
  `a29e9e6066ed33f40482e2d32c780f5bdce40bb336faa54ca52be3e71f064c88`;
- validations:
  `7256ace72552e5ee062b6285a6809a5dcf182693c9caea67e9be20bcc7ccb62e`;
- row manifest:
  `ff966d162803a78b13d45bf1d5db7bd3ea19253b33a8bd32750cfb2db840d877`.
