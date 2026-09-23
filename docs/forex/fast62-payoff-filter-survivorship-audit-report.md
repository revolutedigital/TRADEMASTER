# Fast62 — auditor de filtros de payoff e sobrevivência histórica

Fast61 mostrou que os 11 sinais gated do Fast60 falharam quando liquidados com
trailing e custo. Fast62 abriu o raw top-p do Fast60 para responder:

> existe algum filtro simples no pré-trade que salva o payoff?

Resultado: aparecem filtros positivos no dia 2026-09-23, mas eles são
contraditos pelo histórico Fast43–Fast53. Portanto não há estratégia aprovada.

## Escopo

- stream atual: Fast60 raw top-p;
- sinais raw: 550;
- sinais raw com horizonte completo: 333;
- política: `micro_8_3`;
- latência: 100ms;
- custo esperado/stress: 1 / 2 bps;
- validação histórica: holdouts Fast43–Fast53, 11 blocos entre 2026-07-07 e
  2026-09-20.

## Safety boundary

O relatório mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Arquivos

- relatório JSON: `backend/data/microstructure_v1/reports/fast62-payoff-filter-survivorship-audit/report.json`;
- settlements raw completos: `fast60_raw_complete_settlements.jsonl`;
- sinais incompletos: `fast60_raw_incomplete_signals.jsonl`;
- candidatos de filtro: `fast60_filter_candidates.json`;
- sobrevivência histórica por lado: `historical_side_survivorship.json`.

Hashes:

- report: `056211ea85ca6235c9d9d298700c4671c6c78636fc8997cf2d611baf3edb0f64`;
- fast60 filter candidates: `9900afabdf27ed9347399853ae73b3e50dcf39197e78ba5dde1d4161a8cbdfc4`;
- historical side survivorship: `42ebc31eee0e6d5bb61cdb6d711d9a08d06fb790e80676e8ae7cfff0d6e20f49`;
- raw complete settlements: `7a2f10d29d84fcdabe77a4f559a10f18fa1aaf2a44dfab9ab8b917f7ff573785`;
- raw incomplete signals: `ef47bdb3b1db0b390430d31be6d4f04ab88534e8a4fd52f4052100a676f1c33d`.

## Resultado no Fast60 raw

Baseline raw top-p, com one-position:

- trades: 28;
- stress total: `-107.85493096931133` bps;
- stress mean: `-3.8519618203325474` bps;
- expected win-rate: `0.6071428571428571`;
- max drawdown esperado: `92.61268840892029` bps.

Top filtros diagnósticos do dia:

| Filtro | Trades one-position | Stress total | Stress mean | Observação |
| --- | ---: | ---: | ---: | --- |
| `side=SELL` | 23 | `8.060461378523888` | `0.35045484254451686` | positivo só no dia |
| `side=BUY&recent_signal_count<=1` | 6 | `5.7649580140419285` | `0.9608263356736547` | amostra pequena |
| `side=BUY&recent_signal_count<=2` | 6 | `5.7649580140419285` | `0.9608263356736547` | amostra pequena |

Esses filtros não podem ser promovidos porque foram achados no mesmo dia que
avaliam.

## Sobrevivência histórica por lado

Fast43–Fast53:

| Lado | Trades | Blocos positivos | Stress total | Stress mean | Expected win-rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| BUY | 1157 | 5/11 | `-587.8872152612495` | `-0.5081134099060065` | `0.675885911840968` |
| SELL | 901 | 4/11 | `-1237.313287899453` | `-1.373266690232467` | `0.6459489456159823` |

Isso mata a interpretação simples “opera só SELL” e também mostra por que a
intuição de win-rate/trailing é perigosa: mesmo com 64–68% de expected wins, o
stress fica negativo.

## Status

`FAST62_FAST60_FILTER_POSITIVE_BUT_HISTORICAL_SIDE_SURVIVORSHIP_FAILED`

Motivos:

- `fast60_positive_diagnostic_filters:3`;
- `historical_side_failures:BUY,SELL`;
- `requires_payoff_target_rebuild_not_side_filter_promotion`.

## Conclusão

Ainda não temos estratégia. O caminho correto agora é reconstruir o alvo do
modelo para payoff/distribuição:

- prever deslocamento suficiente para pagar custo, não só probabilidade de
  acertar direção;
- penalizar setups com cauda ruim de stop;
- avaliar por blocos temporais e por one-position antes de qualquer shadow
  prospective;
- só pré-registrar um filtro depois que ele sobreviver em histórico fora da
  seleção.
