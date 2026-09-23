# Fast86 — audit de conflito direcional no Fast80

Resposta objetiva: filtros simples de direção não salvam o Fast80.

Fast86 testa, de forma diagnóstica, se o fracasso do Fast85 poderia ser
resolvido por filtros óbvios:

- operar só BUY;
- operar só SELL;
- remover timestamps com BUY e SELL simultâneos;
- escolher só a maior probabilidade por timestamp.

Resultado: todos os caminhos continuam não positivos em custo de stress.

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
  backend/scripts/research/run_fast86_directional_conflict_audit.py \
  --output-root backend/data/microstructure_v1/reports/fast86-directional-conflict-audit
```

Input:

`backend/data/microstructure_v1/reports/fast85-fast80-historical-outcome-audit/independent_settlements.jsonl`

## Resultado

- status: `FAST86_DIRECTIONAL_CONFLICT_FILTERS_FAILED`;
- settlements de entrada: `14`;
- timestamps com conflito BUY/SELL: `6`;
- nenhum variante teve `stress_total_bps > 0`.

| Variante | Trades | Expected total | Stress total | Initial stops | Breakeven stops |
|---|---:|---:|---:|---:|---:|
| all independent | 14 | `-42.62029995349127` | `-56.62029995349127` | 5 | 9 |
| BUY only | 6 | `-15.047583581519488` | `-21.04758358151949` | 2 | 4 |
| SELL only | 8 | `-27.57271637197178` | `-35.57271637197178` | 3 | 5 |
| conflict rows | 12 | `-44.60299482067093` | `-56.60299482067093` | 5 | 7 |
| non-conflict rows | 2 | `1.982694867179666` | `-0.01730513282033397` | 0 | 2 |
| top probability per timestamp | 8 | `-13.064888714339823` | `-21.064888714339823` | 2 | 6 |
| conflict-veto + top probability | 2 | `1.982694867179666` | `-0.01730513282033397` | 0 | 2 |

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast80 deve ser descartado como estratégia direta. O aprendizado aproveitável é:

1. regime de volatilidade alta existe e produz sinais fortes;
2. o modelo atual não resolve direção nesse regime;
3. conflitos BUY/SELL carregam quase toda a perda;
4. vetar conflito deixa amostra pequena demais e ainda negativa em stress.

Próximo caminho precisa ser nova família de sinal, não remendo do Fast80:

- regime de volatilidade alta como contexto;
- direção decidida por um filtro/modelo separado;
- veto pré-registrado de conflito;
- avaliação sempre em one-position + stress cost.

## Arquivos e hashes

- Fast86 script:
  `cbcadcfeb24e12782493209b36cf11887d0117e75ebeb682d1e4e811a4b619ea`;
- Fast86 tests:
  `d1958d6d9a313b678b23de8eab80ab99085968b6d33e9370add4c47bdf9f29f3`;
- Fast86 report:
  `815585ba9b8b013d97b1a98d3e21b8775cfbe72db3fdadf5ddbb2b91068bc2e6`;
- variant rows:
  `a1deb215853e707654dc2ba74f131da03d78614f5bd84aabeb958f9bd160838a`.

## Validação

```bash
PYTHONPATH=backend ./backend/.venv/bin/pytest \
  backend/tests/unit/test_fast86_directional_conflict_audit.py -q --no-cov
```

Resultado: `5 passed`.

```bash
./backend/.venv/bin/ruff check \
  backend/scripts/research/run_fast86_directional_conflict_audit.py \
  backend/tests/unit/test_fast86_directional_conflict_audit.py
```

Resultado: `All checks passed!`.
