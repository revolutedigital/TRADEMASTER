# Fast174 — crypto cross-asset lead-lag smoke

Fast174 testou uma mudança real de informação depois de BTC/ETH/SOL isolados
falharem no desenho low-cost/top-p. Em vez de olhar um ativo sozinho, o modelo
usa features fechadas de BTCUSDT, ETHUSDT e SOLUSDT juntas para tentar prever
movimentos curtos em ETHUSDT ou SOLUSDT.

Status:

`FAST174_CROSS_ASSET_LEADLAG_FAILED_EXTENSION`

## Resposta curta

Cross-asset quase achou um caminho, mas ainda não virou estratégia.

O teste teve `16` specs:

- `16` passaram no treino;
- `1` passou na validação;
- `0` passou na extensão.

O único sobrevivente até validação foi:

`ETHUSDT`, horizonte `900s`, top fraction `0.02`.

Ele fez:

- treino: `+893.4026` bps, `34` trades;
- validação `fast53`: `+29.1748` bps, `11` trades;
- extensão `fast56`: `-272.3073` bps, `11` trades.

Ou seja: não dá para promover. O último holdout matou a hipótese.

## Dados

Símbolos usados:

- `BTCUSDT`;
- `ETHUSDT`;
- `SOLUSDT`.

Período:

- `fast50`: `2026-08-30`, `2026-08-31`;
- `fast51`: `2026-09-06`, `2026-09-07`;
- `fast52`: `2026-09-13`, `2026-09-14`;
- `fast53`: `2026-09-19`, `2026-09-20`;
- `fast56`: `2026-09-21`, `2026-09-22`.

Blocos:

- treino: `fast50`, `fast51`, `fast52`;
- validação: `fast53`;
- extensão/late holdout: `fast56`.

## Protocolo

Runner:

`backend/scripts/research/run_fast174_crypto_cross_asset_leadlag_smoke.py`

Features:

- retornos fechados de `1`, `5`, `15`, `30` minutos;
- fluxo taker buy/sell agregado;
- quote volume;
- volatilidade realizada;
- hora do dia codificada em seno/cosseno;
- features dos três ativos disponíveis para prever ETHUSDT ou SOLUSDT.

Modelo:

- `Ridge(alpha=10)`;
- imputação constante `0`;
- `StandardScaler`;
- direção = sinal da previsão;
- seleção = maior `abs(prediction)`;
- custo explícito: `2.0` bps.

Grade:

- targets negociados: `ETHUSDT`, `SOLUSDT`;
- horizontes: `900`, `1800` segundos;
- top fractions: `0.005`, `0.01`, `0.02`, `0.05`;
- specs totais: `16`.

Critério:

Para virar candidata, a spec precisava passar simultaneamente:

1. treino positivo com pelo menos `3` trades;
2. validação positiva com pelo menos `3` trades;
3. extensão positiva com pelo menos `3` trades.

## Resultado

Resumo:

| Métrica | Valor |
|---|---:|
| Rows de decisão | `2,880` |
| Features | `47` |
| Specs | `16` |
| Screen survivors | `16` |
| Validation survivors | `1` |
| Strict survivors | `0` |

Único validation survivor:

| Target | Horizonte | Top | Treino net | Validação net | Extensão net | Decisão |
|---|---:|---:|---:|---:|---:|---|
| `ETHUSDT` | `900s` | `0.02` | `+893.4026` | `+29.1748` | `-272.3073` | Rejeitar |

Melhores extensões que falharam na validação:

| Target | Horizonte | Top | Treino net | Validação net | Extensão net |
|---|---:|---:|---:|---:|---:|
| `ETHUSDT` | `1800s` | `0.05` | `+2294.4518` | `-204.0790` | `+505.4354` |
| `SOLUSDT` | `1800s` | `0.05` | `+2274.9065` | `-374.3027` | `+369.9319` |
| `SOLUSDT` | `1800s` | `0.01` | `+923.9730` | `-122.4124` | `+176.9000` |

Essas linhas são tentadoras porque a extensão é positiva, mas elas não passam
no bloco `fast53`. Isso reforça o padrão que vem aparecendo: o sinal captura
bolsões de regime, mas não tem detector causal estável antes da entrada.

## Interpretação

Fast174 é o primeiro teste dessa rodada em que a informação cross-asset mostrou
alguma tração fora do treino. Mas ainda não é suficiente:

- se aceitarmos o único sobrevivente de validação, ele perde feio na extensão;
- se escolhermos pelo melhor resultado na extensão, estamos olhando o holdout;
- o comportamento muda demais entre `fast53` e `fast56`.

Decisão:

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O aprendizado válido é que lead-lag/cross-asset é mais promissor que "mesmo
top-p isolado em outro par", mas precisa de uma camada de regime antes da
entrada. A próxima variação, se continuarmos nesse caminho, deve tentar detectar
quando `fast53`-like está ativo antes de operar, sem usar `fast56` para escolher.

## Segurança

Fast174 é pesquisa offline:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `dry_run=true`.

Nenhum trade foi enviado. Nenhum ledger foi alterado.

## Artefatos

- runner:
  `backend/scripts/research/run_fast174_crypto_cross_asset_leadlag_smoke.py`;
- teste:
  `backend/tests/unit/test_fast174_crypto_cross_asset_leadlag_smoke.py`;
- report JSON:
  `backend/data/microstructure_cross_asset_v1/reports/fast174-crypto-cross-asset-leadlag-smoke/report.json`;
- summary:
  `backend/data/microstructure_cross_asset_v1/reports/fast174-crypto-cross-asset-leadlag-smoke/summary.json`;
- evaluations:
  `backend/data/microstructure_cross_asset_v1/reports/fast174-crypto-cross-asset-leadlag-smoke/spec-evaluations.json`.

Hashes:

- report:
  `63f71324fad607dca020f15dac4e98861c58d64683d655d7f6402fe0a98507e6`;
- summary:
  `f2e5f0508241371f39da54dd7cdb4bb96eadf61c96476bfd30fab600cd6253f2`;
- evaluations:
  `138f96e6138a4868d292fbdf2808e0fb31fcec3a12063cf5f2ea07376c876ae4`.
