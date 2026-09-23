# Fast36 — holdout one-position-first

Data: 2026-09-23  
Mercado: BTCUSDT futures  
Tese: candidato Fast34/Fast35 operado somente como uma posição por vez  
Resultado: tese rejeitada no holdout Fast36.

## Captura Fast36

WAL:

`backend/data/microstructure_v1/prospective-wal-fast36-30m`

Streams capturados:

| Stream | Linhas | Primeiro evento | Último evento |
| --- | ---: | --- | --- |
| trade | 374.146 | 2026-09-23T13:52:02.291Z | 2026-09-23T14:22:00.799Z |
| depth | 16.091 | 2026-09-23T13:52:02.397Z | 2026-09-23T14:22:00.839Z |
| mark_price | 1.377 | 2026-09-23T13:52:01Z | 2026-09-23T14:22:00Z |
| spot_trade | 209.853 | 2026-09-23T13:52:02.223Z | 2026-09-23T14:22:01.195Z |

Houve reconnects durante a captura. A emenda técnica aplicada antes de observar
PnL foi excluir decisões com book mais velho que 1000ms, que é o mesmo limite
usado pelo gate de frescor.

Filtro aplicado:

- decisões originais: 330;
- decisões retidas: 308;
- decisões removidas: 22;
- motivo: `book features are incomplete: 0 decisions without book, 22 decisions older than 1000ms`.

## WAL scout

Report:

`backend/data/microstructure_v1/reports/book-wal-scout-fast36-30m-120s/report.json`

Hash:

`6c928022a0f62ba3a3823f5d7ee459c2e4cd84ed273f752ab3947805f78715a7`

Status:

`READY_FOR_SCOUT_MODELING`

Resumo:

- decision_count: 308;
- row_count: 616;
- intervalo de decisões: 2026-09-23T13:52:35Z até 2026-09-23T14:20:00Z.

## Holdout one-position congelado

Report:

`backend/data/microstructure_v1/reports/book-frozen-model-holdout-fast36-one-position-100ms/report.json`

Hash:

`e78a69d1ee6008b42c42baf530d1ed9cae76e3e17a0e626548261fe98e9c3471`

Status:

`FROZEN_MODEL_HOLDOUT_FAILED`

Motivo:

- `one_position_net_at_1bps_non_positive`.

## Candidato avaliado

Parâmetros pré-registrados em
`docs/forex/fast36-one-position-frozen-holdout-preregistration.md`:

- modelo: `hist_gradient`;
- treino: Fast24 + Fast28 + Fast29 + Fast32 + Fast35;
- holdout: Fast36;
- horizonte: 120s;
- target / breakeven / ativação do trailing: 8 bps;
- trailing distance: 3 bps;
- top-p: 5%;
- latência: 100ms;
- stop inicial: 20 bps;
- custo principal: 1 bps;
- custo de estresse: 2 bps;
- required positive modes: `one_position`.

## Resultado

| Modo | Execuções | Gross médio | Net 1 bps | Net 2 bps | Win rate | Max DD | Saídas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| independente | 31 | -6,7107 | -7,7107 | -8,7107 | 45,16% | 260,38 bps | 17 INITIAL_STOP / 14 TRAILING_STOP |
| uma posição | 9 | -9,9045 | -10,9045 | -11,9045 | 33,33% | 89,14 bps | 6 INITIAL_STOP / 3 TRAILING_STOP |

Outros pontos:

- holdout target-rate base: 49,51%;
- selected target-rate: 45,16%;
- sinais selecionados: 31.

## Decisão

Fast36 rejeita a tese one-position-first.

O ponto mais importante: o modelo não só perdeu dinheiro no modo operacional,
como selecionou pior que a base do próprio holdout. Isso indica instabilidade de
regime/modelo, não apenas custo ou execução.

Conclusão:

- Fast34/Fast35 não devem virar estratégia;
- a família `hist_gradient top 5%, 8 bps, trailing 3 bps` está rejeitada como
  caminho principal;
- trailing stop não salvou a entrada ruim em Fast36;
- sem Testnet/LIVE.

Próximo caminho precisa voltar para descoberta de outra família de sinal ou
para um modelo explicitamente treinado em estabilidade por regime, não só
probabilidade de tocar alvo.
