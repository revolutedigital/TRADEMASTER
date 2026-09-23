# Fast92 — broad universe scout

Resposta objetiva: ampliar muito o universo de entrada também não criou
estratégia. O Fast89 isolado parecia positivo em one-position, mas o histórico
e o leave-one-label-out desmontaram esse sinal.

Fast92 reutiliza os runners Fast90/Fast91 com margem extrema
`threshold - 0.20`. Como o threshold congelado é `0.90195311025804`, isso
inclui praticamente todo candidato com probabilidade acima de `0.70195311025804`
e testa uma família muito mais ampla que o top-p rígido.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Fast89 broad

Comando:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast90_near_threshold_scout.py \
  --input-path backend/data/microstructure_v1/reports/fast84-top-p-silence-diagnostic-fast89/all_scored_decisions.jsonl \
  --wal-roots fast89=backend/data/microstructure_v1/prospective-wal-fast89-30m \
  --output-root backend/data/microstructure_v1/reports/fast92-broad-universe-scout-fast89 \
  --probability-margins 0.20
```

Resultado:

- status: `FAST90_NO_NEAR_THRESHOLD_EDGE_FOUND`;
- scored rows: `666`;
- candidate universe: `642`;
- complete outcomes: `526`.

Top one-position no Fast89:

| Variante | Trades | Stress total |
|---|---:|---:|
| `near200_all_top_probability` | 5 | `+15.961764581233997` |
| `near200_flow5_positive_top_probability` | 5 | `+11.89415528446625` |
| `near200_return5_positive_top_probability` | 5 | `+11.89415528446625` |
| `near200_all_veto` | 2 | `+3.764563932474882` |

Isso parece bom em uma janela, mas é amostra pequena. Por isso o Fast92 exige
histórico e LOLO.

## Histórico broad

Comando:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast90_near_threshold_scout.py \
  --input-path backend/data/microstructure_v1/reports/fast84-top-p-silence-diagnostic-all-local-wals/all_scored_decisions.jsonl \
  --output-root backend/data/microstructure_v1/reports/fast92-broad-universe-scout-all-local \
  --probability-margins 0.20
```

Resultado:

- status: `FAST90_NO_NEAR_THRESHOLD_EDGE_FOUND`;
- scored rows: `8.026`;
- candidate universe: `7.968`;
- complete outcomes: `6.579`.

Top one-position histórico:

| Variante | Trades | Labels | Stress total | Labels negativos | Pior label |
|---|---:|---:|---:|---:|---:|
| `near200_vol15_flow5_positive_top_probability` | 10 | 3 | `+10.970372066877642` | 1 | `-16.30108043008083` |
| `near200_all_veto` | 2 | 2 | `-14.513189764877325` | 1 | `-18.008251577987437` |
| `near200_flow5_positive_top_probability` | 82 | 12 | `-152.79306316450095` | 9 | `-49.93753843472706` |
| `near200_return5_and_flow5_positive_top_probability` | 88 | 12 | `-204.21913926602696` | 11 | `-41.85577411925271` |
| `near200_all_top_probability` | 81 | 12 | `-224.9358617737434` | 8 | `-92.52061780046319` |

O único agregado positivo no histórico voltou a ser o bolso já conhecido
`vol15 + flow5`, ainda com label negativo. Ou seja: ampliar universo não revelou
uma família nova robusta.

## LOLO broad

Comando:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast91_lolo_gate_audit.py \
  --input-path backend/data/microstructure_v1/reports/fast92-broad-universe-scout-all-local/independent_settlements.jsonl \
  --output-root backend/data/microstructure_v1/reports/fast92-broad-universe-lolo-audit \
  --probability-margins 0.20
```

Resultado:

- status: `FAST91_LOLO_GATE_FAILED`;
- holdouts: `12`;
- positivos: `0`;
- negativos: `3`;
- insuficientes: `9`;
- holdout trades: `30`;
- holdout stress total: `-46.02338249898462` bps;
- pior holdout: `-18.5003294563108` bps.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast92 elimina outro caminho:

- o positivo do Fast89 broad é ruído de janela pequena;
- operar universo amplo destrói resultado quando há frequência;
- gates LOLO em cima do universo amplo não produzem holdouts positivos;
- o caminho curto `microstructure + trailing` ainda não tem edge robusto.

## Implicação

Não vale continuar tentando “mais entrada” dentro desse mesmo modelo/feature set
sem uma mudança real de hipótese. Os dados estão dizendo:

1. top-p rígido fica mudo;
2. near-threshold piora;
3. gate simples não generaliza;
4. broad universe gera custo/ruído.

Próxima direção com maior chance: abandonar essa família curta e procurar
operações menos frequentes/maiores, ou criar labels/modelo novos voltados para
payoff líquido do trailing, não para probabilidade de toque/entrada.

## Arquivos e hashes

- Fast92 Fast89 broad report:
  `2ada72449d2feeaebf1b4df3d62cb2a211a712e9a0a648fd44ada6c48b56e0ff`;
- Fast92 all-local broad report:
  `a821de31654cf573f574e63f7adefd69e7ce4ec988337b9de100b5ced05b85cb`;
- Fast92 LOLO report:
  `06be2265249f604ef7c24c5ec05ce98111f31c2b36117e9a1ad8f1fd46a980f7`;
- Fast92 LOLO holdout results:
  `1de437586fc7caecc16d12ae6a6495d92c618b837da3bfbe02dd534cc794b35f`.
