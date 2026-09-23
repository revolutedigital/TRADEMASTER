# Rodada 32 — holdout futuro do segundo filtro congelado

Data: 2026-09-23  
Mercado: BTCUSDT futures, WAL prospectivo Fast32 de 30 minutos  
Resultado: o segundo filtro `directed_spot_perp_basis_bps <= 5.578475508410952` falhou no holdout futuro.

## Captura Fast32

WAL:

`backend/data/microstructure_v1/prospective-wal-fast32-30m`

Cobertura:

| Stream | Eventos | Primeiro evento | Último evento |
| --- | ---: | --- | --- |
| trade | 47.040 | 2026-09-23T12:31:30.316Z | 2026-09-23T13:01:18.906Z |
| depth | 16.434 | 2026-09-23T12:31:30.303Z | 2026-09-23T13:01:19.381Z |
| spot_trade | 48.454 | 2026-09-23T12:31:31.131Z | 2026-09-23T12:59:59.953Z |
| mark_price | 1.382 | 2026-09-23T12:31:29Z | 2026-09-23T13:01:28.006Z |

Houve reconnects perto do fim da coleta, mas o WAL scout ainda ficou pronto para modelagem:

- status: `READY_FOR_SCOUT_MODELING`;
- decisões: 327;
- rows: 654;
- WAL scout hash: `e2ba24f936dedcbce584f83b2a5640b89510354a64ebb2b55ffa6de3d04049bd`.

## Regra base congelada

Primeiro rodei a regra fixa original, sem segundo filtro:

- shadow decisions: 327;
- sinais shadow: 50;
- window report hash: `95bde3cb6b4ca4739da7d82b57346361e8ff277fdc3f52b506059bb115dbf109`;
- settlement 100ms hash: `2747e7d4b59662d4d654035db1f455c114deb22a6bd2dbd2957134a3a33c95ff`.

Resultado da regra base em Fast32:

| Modo | Trades exec. | Gross médio | Net 1 bps | Net 2 bps | Saídas |
| --- | ---: | ---: | ---: | ---: | --- |
| independente | 45 | +0,9295 | -0,0705 | -1,0705 | 15 TARGET / 30 TIMEOUT |
| uma posição | 11 | +1,5310 | +0,5310 | -0,4690 | 4 TARGET / 7 TIMEOUT |

Leitura: Fast32 não salva a regra base. O modo uma posição foi positivo em 1 bps, mas o modo independente ficou negativo e 2 bps segue ruim.

## Segundo filtro congelado

Filtro escolhido na Rodada 31, sem reescolher com Fast32:

`directed_spot_perp_basis_bps <= 5.578475508410952`

Holdout report:

`backend/data/microstructure_v1/reports/book-fixed-gate-second-filter-holdout-fast32-100ms/report.json`

Hash:

`9cabfafd204e888699bcd961c192ac3f8bb20f1e51c0d4c8c954f6b0a58f0ab5`

Status:

`SECOND_FILTER_FIXED_CANDIDATE_HOLDOUT_FAILED`

Motivo:

- `fixed_candidate_profit_cost_non_positive`.

Resultado:

| Modo | Sinais após filtro | Trades exec. | Net 1 bps | Net 2 bps | Saídas |
| --- | ---: | ---: | ---: | ---: | --- |
| independente | 19 | 19 | -0,1233 | -1,1233 | — |
| uma posição | 19 | 8 | -0,4006 | -1,4006 | 3 TARGET / 5 TIMEOUT |

## Decisão

O segundo filtro falhou no primeiro holdout futuro.

Portanto:

- descartar `directed_spot_perp_basis_bps <= 5.578475508410952` como estratégia;
- não promover para Testnet;
- não insistir em trailing stop para esconder entrada instável;
- manter essas features apenas como material de pesquisa.

Ainda existe algum sinal na família, mas ele não está estável o suficiente para execução.
