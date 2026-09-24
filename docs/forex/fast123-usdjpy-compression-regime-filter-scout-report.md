# Fast123 — filtro de regime para o clue USDJPY H1 compression breakout

Resposta objetiva: ainda não temos estratégia. O Fast123 tentou salvar a melhor
pista do Fast121/Fast122 com filtros causais simples, escolhidos no histórico
antigo e validados fora da seleção. Nenhum filtro passou.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`;
- sem Demo;
- sem Testnet;
- sem LIVE;
- sem envio de ordem.

## O que foi testado

Runner:

`backend/scripts/research/run_fast123_usdjpy_compression_regime_filter_scout.py`

Fonte:

`backend/data/lab_fast122/usdjpy-compression-breakout-temporal-audit/frozen-policy-trades.parquet`

Regra base congelada:

```text
USDJPY
timeframe: H1
channel_bars: 20
compression_quantile: 0.40
stop_atr_multiple: 1.5
max_hold_minutes: 720
management: fixed_3r
```

Slices:

- train: antes de `2016-01-01`;
- pre-validation: `2016-01-01` até `2018-12-01`;
- current: `2019-01-01` até `2026-09-01`.

Filtros entry-known avaliados:

- lado (`LONG`/`SHORT`);
- sessão UTC (`asia`, `london`, `new_york`, `late_us`);
- trimestre;
- weekday;
- bucket causal de ATR;
- bucket causal de razão de compressão;
- bucket causal de rompimento/ATR;
- combinações focadas de duas dimensões.

Total:

- `82` filtros.

## Resultado

Status:

`FAST123_USDJPY_COMPRESSION_REGIME_FILTER_FAILED`

Resumo:

| Métrica | Valor |
|---|---:|
| Filtros avaliados | `82` |
| Filtros fracos | `0` |
| Filtros estritos | `0` |
| Estratégia aprovada | `false` |

Critério fraco exigia:

- trades mínimos em train, pre-validation e current;
- stress médio positivo nos três slices.

Critério estrito exigia adicionalmente:

- lower bound de stress positivo em pre-validation e current;
- estabilidade anual mínima.

## Por que falhou

Alguns filtros ficaram muito bons no período atual, mas não sobrevivem ao treino
antigo / pre-validation. Exemplos:

| Filtro | Train stress | Pre-val stress | Current stress | Problema |
|---|---:|---:|---:|---|
| `session_bucket=london&side=LONG` | `+7.9599` | `-5.6094` | `+9.8159` | quebra em 2016–2018 |
| `breakout_over_atr_bucket=high&session_bucket=london` | `+0.5578` | `-2.4836` | `+10.1473` | quebra em 2016–2018 e amostra baixa |
| `entry_weekday=3` | `+2.2834` | `+4.3313` | `+8.6983` | amostra insuficiente e lower bound pre-val negativo |
| `compression_ratio_bucket=mid&side=SHORT` | `-4.3756` | `+6.6078` | `+8.2836` | quebra no treino antigo |

Leitura:

- o clue USDJPY continua melhor que os ramos anteriores;
- mas filtros simples não explicam causalmente quando ele liga/desliga;
- escolher o filtro pela performance atual seria vazamento/overfit;
- sem filtro estável ou observação prospectiva, não há estratégia.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast123 elimina a rota “vamos só filtrar por sessão/volatilidade/lado” como
solução simples. O próximo passo honesto, se quisermos insistir nessa pista, é
pré-registrar observação prospectiva da regra congelada — sem mexer no filtro —
ou buscar dados/mercados onde o mesmo padrão apareça com estabilidade real.

## Comando executado

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast123_usdjpy_compression_regime_filter_scout.py
```

## Verificações executadas

```bash
./backend/.venv/bin/python -m pytest \
  backend/tests/unit/test_fast123_usdjpy_compression_regime_filter_scout.py \
  -q --no-cov

./backend/.venv/bin/python -m ruff check \
  backend/scripts/research/run_fast123_usdjpy_compression_regime_filter_scout.py \
  backend/tests/unit/test_fast123_usdjpy_compression_regime_filter_scout.py
```

Resultado:

- unit tests: `6 passed`;
- lint: `All checks passed`.

## Arquivos e hashes

- report:
  `backend/data/lab_fast123/usdjpy-compression-regime-filter-scout/report.json`;
- report sha256:
  `24a2018cf484a6e2f66e009bbe5d362ae75dcba96590422fbf5d8c81b026d9d8`;
- filter evaluations:
  `backend/data/lab_fast123/usdjpy-compression-regime-filter-scout/filter-evaluations.json`;
- filter evaluations sha256:
  `e04b2a20a1b501224a3edfb060befb02c0fe558f3c092b819a1d8ee2a42ca810`;
- weak filters:
  `backend/data/lab_fast123/usdjpy-compression-regime-filter-scout/weak-filters.json`;
- weak filters sha256:
  `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`;
- strict filters:
  `backend/data/lab_fast123/usdjpy-compression-regime-filter-scout/strict-filters.json`;
- strict filters sha256:
  `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`.
