# Fast85 — outcome histórico dos matches Fast80

Resposta objetiva: o regime Fast80 apareceu em WALs locais antigos, mas quando
foi liquidado com o trailing/custo congelado o resultado foi negativo. Portanto
Fast80, do jeito atual, não vira estratégia.

Este audit é histórico e inclui eventos antes do pré-registro Fast80. Ele serve
para diagnóstico do caminho, não para aprovação.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Comando

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast85_fast80_historical_outcome_audit.py \
  --output-root backend/data/microstructure_v1/reports/fast85-fast80-historical-outcome-audit
```

Input:

`backend/data/microstructure_v1/reports/fast84-top-p-silence-diagnostic-all-local-wals/all_scored_decisions.jsonl`

## Resultado

- status: `FAST85_FAST80_HISTORICAL_OUTCOME_FAILED`;
- scored rows de entrada: `8.026`;
- candidatos Fast80+top-p: `15`;
- labels com candidatos: `3`;
- outcomes completos independentes: `14`;
- outcome incompleto: `1`;
- one-position outcomes: `5`;
- candidatos pós-pré-registro Fast80: `0`;
- candidatos pré-pré-registro Fast80: `15`.

### Independente

| Métrica | Valor |
|---|---:|
| trades | `14` |
| expected mean | `-3.04430713953509` bps |
| expected total | `-42.62029995349126` bps |
| stress mean | `-4.04430713953509` bps |
| stress total | `-56.620299953491255` bps |
| win rate expected | `0.6428571428571429` |
| max drawdown | `53.20328921405858` bps |

### One-position

| Métrica | Valor |
|---|---:|
| trades | `5` |
| expected mean | `-1.0862161356715396` bps |
| expected total | `-5.431080678357699` bps |
| stress mean | `-2.0862161356715396` bps |
| stress total | `-10.431080678357699` bps |
| win rate expected | `0.8` |
| max drawdown | `17.006895797289694` bps |

## Leitura dos trades

O problema não é só custo médio. O cluster principal em fast36 acendeu os dois
lados em timestamps muito próximos. A seleção one-position por maior
probabilidade escolheu:

- fast20 `SELL`: stress `-0.008744253452875439` bps;
- fast36 `BUY` 14:10:50: stress `+7.6002943933876494` bps;
- fast36 `BUY` 14:11:05: stress `-18.006895797289694` bps;
- fast36 `BUY` 14:13:10: stress `-0.013754741836487616` bps;
- fast36 `BUY` 14:13:25: stress `-0.001980279166291954` bps.

O resultado final ficou negativo mesmo com `80%` de win rate expected, porque
um initial stop grande anulou vários breakevens pequenos. Esse é exatamente o
risco que o Igor levantou: trailing protege, mas não salva entrada ruim quando
o stop inicial pega antes do movimento certo.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast80 ainda é uma hipótese útil para aprender, mas não é estratégia:

1. as duas janelas prospectivas novas não tiveram match;
2. quando matches aparecem no histórico local, o replay com trailing/custo é
   negativo;
3. a regra acende conflitos BUY/SELL no mesmo regime, sugerindo que falta
   decisão direcional robusta.

Próximo caminho: parar de tentar promover Fast80 diretamente e usar o que ele
ensinou para buscar uma família nova: regime de volatilidade alta + filtro
direcional explícito + veto de conflito BUY/SELL, sempre com settlement
one-position e custo de stress.

## Arquivos e hashes

- Fast85 script:
  `062a564e37b1343876f3ef2a7fa042285c470a711dbfb07484fa803aa7c5a65c`;
- Fast85 tests:
  `bd4e8a85f9c06c4eed862858244abac5be2ac2db8820f2a10b5e4debfc9afaf7`;
- Fast85 report:
  `c76414a9c4bffdf471e188b6d14220c7e44d4ff511f208412388826585804975`;
- independent settlements:
  `d22525d0ea566657e24af4a7651bba9176263d2f565b9c037f744a2a600047ff`;
- one-position settlements:
  `507370b79f4d9fa7c4c2eace27f3dbb27a1b9a794cfb77a7d31f7e1bb4ab5cf6`.

## Validação

```bash
PYTHONPATH=backend ./backend/.venv/bin/pytest \
  backend/tests/unit/test_fast85_fast80_historical_outcome_audit.py -q --no-cov
```

Resultado: `5 passed`.

```bash
./backend/.venv/bin/ruff check \
  backend/scripts/research/run_fast85_fast80_historical_outcome_audit.py \
  backend/tests/unit/test_fast85_fast80_historical_outcome_audit.py
```

Resultado: `All checks passed!`.
