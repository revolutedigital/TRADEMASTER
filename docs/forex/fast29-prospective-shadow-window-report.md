# Rodada 29 — terceira janela shadow prospectiva da candidata fixa

Data: 2026-09-23  
Mercado: BTCUSDT futures, WAL prospectivo Fast29 de 30 minutos  
Resultado: a janela foi válida para modelagem, mas a candidata falhou no modo mais realista de execução com uma posição por vez.

## Captura Fast29

WAL:

`backend/data/microstructure_v1/prospective-wal-fast29-30m`

Cobertura:

| Stream | Eventos | Primeiro evento | Último evento |
| --- | ---: | --- | --- |
| trade | 95.822 | 2026-09-23T11:49:42.598Z | 2026-09-23T12:19:34.205Z |
| depth | 17.556 | 2026-09-23T11:49:43.648Z | 2026-09-23T12:19:34.161Z |
| spot_trade | 71.235 | 2026-09-23T11:49:35.892Z | 2026-09-23T12:19:34.185Z |
| mark_price | 1.364 | 2026-09-23T11:49:34Z | 2026-09-23T12:19:34.001Z |

WAL scout:

- status: `READY_FOR_SCOUT_MODELING`;
- decisões: 328;
- rows: 656;
- report hash: `5e496271f3e0fc5e85398418a87dde67098c730390424dacef82de2e6b9e6fe9`.

Houve um reconnect do stream spot durante a captura, mas o stream voltou e o gate causal continua fail-closed quando spot fica indisponível ou velho.

## Regra testada

Mesma regra congelada:

- lado: `SELL`;
- entrada base: `directed_spot_perp_basis_bps >= 5.2032340279142275`;
- gate: `spot_update_age_ms <= 140.48`;
- horizonte: 120s;
- alvo: 5 bps;
- stop inicial: 20 bps;
- replay: marketable top-of-book, com spread embutido no gross.

Sem reotimização, sem ordem, sem ledger:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `approved_strategy=false`.

## Janela shadow

Wrapper:

`backend/scripts/research/run_book_fixed_gate_prospective_window.py`

Janela efetiva:

- start: `2026-09-23T11:50:15Z`;
- último decision time: `2026-09-23T12:17:30Z`;
- end exclusivo: `2026-09-23T12:17:35Z`;
- decisões: 328;
- sinais shadow: 7.

Hashes:

- window report: `58ad41f2ad34de66d318c666ba86f26e777556314f1a2cc5a5e0e691de5afa70`;
- shadow policy report: `d956c86809fb1ea75c904750ad5e0858c0dda9ed4639474ae98c0092a59b3a16`;
- shadow decisions JSONL: `1165ecfe900f2d561ea23881d5142d5a9dba5fa573f3bce747a59e5f7dd4d7b3`.

Status do wrapper:

`PROSPECTIVE_SHADOW_WINDOW_SETTLED_FAILED_GATE`

Motivos:

- `independent_executable_count_below_minimum:7<10`;
- `one_position_executable_count_below_minimum:4<10`;
- `one_position_net_at_1bps_non_positive`.

## Settlement por latência

| Latência | Modo | Trades exec. | Gross médio | Net 1 bps | Net 2 bps | Saídas | MDD bps |
| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: |
| 0ms | independente | 7 | +1,8877 | +0,8877 | -0,1123 | 3 TARGET / 4 TIMEOUT | 4,1046 |
| 0ms | uma posição | 4 | -0,4465 | -1,4465 | -2,4465 | 4 TIMEOUT | 4,1046 |
| 100ms | independente | 7 | +1,7759 | +0,7759 | -0,2241 | 3 TARGET / 4 TIMEOUT | 4,8872 |
| 100ms | uma posição | 4 | -0,6421 | -1,6421 | -2,6421 | 4 TIMEOUT | 4,8872 |
| 250ms | independente | 7 | +1,7642 | +0,7642 | -0,2358 | 3 TARGET / 4 TIMEOUT | 4,9690 |
| 250ms | uma posição | 4 | -0,6626 | -1,6626 | -2,6626 | 4 TIMEOUT | 4,9690 |
| 500ms | independente | 7 | +1,6791 | +0,6791 | -0,3209 | 3 TARGET / 4 TIMEOUT | 5,5647 |
| 500ms | uma posição | 4 | -0,8115 | -1,8115 | -2,8115 | 4 TIMEOUT | 5,5647 |
| 1000ms | independente | 7 | +1,6624 | +0,6624 | -0,3376 | 3 TARGET / 4 TIMEOUT | 5,5647 |
| 1000ms | uma posição | 4 | -0,8408 | -1,8408 | -2,8408 | 4 TIMEOUT | 5,5647 |

Arquivos principais:

- 100ms settlement: `backend/data/microstructure_v1/reports/book-fixed-gate-prospective-window-fast29-30m-100ms/settlement-100ms/report.json`;
- 100ms settlement hash: `7900db36d1a622f410499eebc0e63f81573c4e656b5802e83463fd723ca1989f`.

## Leitura

Fast29 enfraqueceu bastante a tese:

- a regra independente ainda ficou positiva em 1 bps, mas com só 7 trades;
- em 2 bps continuou negativa;
- o modo uma posição por vez ficou negativo em todas as latências testadas;
- portanto, o edge não está robusto o suficiente para execução sequencial realista.

Conclusão: esta candidata não deve ser promovida. Ela ainda pode servir como feature/gate dentro de uma política melhor, mas isolada não passa.
