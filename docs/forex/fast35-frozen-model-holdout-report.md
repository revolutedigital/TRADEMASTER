# Fast35 — holdout prospectivo do candidato Fast34 congelado

Data: 2026-09-23  
Mercado: BTCUSDT futures  
Janela prospectiva: Fast35  
Treino: Fast24 + Fast28 + Fast29 + Fast32  
Holdout: Fast35  
Resultado: reprova no critério estrito pré-registrado, mas mantém sinal forte no modo uma-posição.

## Captura Fast35

WAL:

`backend/data/microstructure_v1/prospective-wal-fast35-30m`

Streams capturados:

| Stream | Linhas | Primeiro evento | Último evento |
| --- | ---: | --- | --- |
| trade | 138.440 | 2026-09-23T13:18:24.937Z | 2026-09-23T13:48:12.035Z |
| depth | 17.523 | 2026-09-23T13:18:24.993Z | 2026-09-23T13:48:12.182Z |
| mark_price | 1.371 | 2026-09-23T13:18:24.005Z | 2026-09-23T13:48:11Z |
| spot_trade | 120.896 | 2026-09-23T13:18:24.875Z | 2026-09-23T13:48:13.263Z |

## WAL scout

Report:

`backend/data/microstructure_v1/reports/book-wal-scout-fast35-30m-120s/report.json`

Hash:

`b684abed229e79d501b1db51022bccb267ff7abd09ade2542f12d4c6ad6129a5`

Status:

`READY_FOR_SCOUT_MODELING`

Resumo:

- decision_count: 330;
- row_count: 660;
- intervalo de decisões: 2026-09-23T13:18:55Z até 2026-09-23T13:46:20Z.

## Holdout congelado

Report:

`backend/data/microstructure_v1/reports/book-frozen-model-holdout-fast35-100ms/report.json`

Hash:

`b2d3c761d9b79464b71aaeb235f98090ded5051609811dd2f15bdb2b509e08af`

Status:

`FROZEN_MODEL_HOLDOUT_FAILED`

Motivo:

- `independent_net_at_1bps_non_positive`.

## Candidato avaliado

Parâmetros pré-registrados em
`docs/forex/fast35-frozen-model-holdout-preregistration.md`:

- modelo: `hist_gradient`;
- horizonte: 120s;
- target / breakeven / ativação do trailing: 8 bps;
- trailing distance: 3 bps;
- top-p: 5%;
- latência: 100ms;
- stop inicial: 20 bps;
- custo principal: 1 bps;
- custo de estresse: 2 bps;
- minimum executable count: 5.

## Resultado

| Modo | Execuções | Gross médio | Net 1 bps | Net 2 bps | Win rate | Max DD | Saídas |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| independente | 33 | +0,4752 | -0,5248 | -1,5248 | 54,55% | 94,13 bps | 4 INITIAL_STOP / 17 TIMEOUT / 12 TRAILING_STOP |
| uma posição | 10 | +5,4421 | +4,4421 | +3,4421 | 80,00% | 20,00 bps | 1 INITIAL_STOP / 2 TIMEOUT / 7 TRAILING_STOP |

Outros pontos:

- holdout target-rate base: 33,03%;
- selected target-rate: 51,52%;
- sinais selecionados: 33.

## Decisão

Pelo critério estrito pré-registrado, Fast35 reprova porque o modo independente
fica negativo no custo de 1 bps.

Mas o resultado muda a leitura do caminho:

- como grade independente, o candidato ainda não é robusto;
- como política operacional de uma posição por vez, Fast35 foi forte;
- diferente de Fast34, aqui o trailing apareceu de verdade: 7 de 10 saídas no
  modo uma-posição foram `TRAILING_STOP`;
- o sinal selecionou melhor que a base: 51,52% vs 33,03% de target-rate.

Conclusão honesta:

Ainda não temos estratégia aprovada. Mas agora temos uma tese mais estreita e
mais plausível:

> modelo `hist_gradient` top 5%, 8 bps de ativação, trailing 3 bps, com execução
> limitada a uma posição por vez por símbolo.

Esse caminho precisa de nova pré-registração. Não devemos salvar a hipótese
antiga alterando critério depois de ver Fast35. O próximo teste correto é
Fast36 com critério explicitamente "portfolio/one-position first".

Sem Testnet/LIVE.
