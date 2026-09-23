# Fast90 — scout near-threshold

Resposta objetiva: baixar o threshold do top-p não cria estratégia. No Fast89,
as poucas entradas que surgiriam com `threshold - 0.05` foram negativas após
custo de stress. No histórico local, o relaxamento gera muitos trades e o
resultado agregado também não fica robusto.

Fast90 é diagnóstico pós-outcome, não pré-registro de estratégia.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Pergunta testada

Depois do Fast89 ficar sem top-p, a hipótese natural era:

> Talvez o threshold `0.90195311025804` esteja alto demais. Se aceitarmos
> decisões perto do threshold e mantivermos trailing/flow direcional, talvez
> apareça edge.

Fast90 testou margens:

- `threshold - 0.02`;
- `threshold - 0.05`.

Variantes avaliadas:

- sem filtro direcional;
- `directed_flow_imbalance_5s > 0`;
- `directed_return_5s_bps > 0`;
- `directed_return_5s_bps > 0` + `directed_flow_imbalance_5s > 0`;
- `realized_vol_30s_bps >= 15` + `directed_flow_imbalance_5s > 0`;
- conflito por maior probabilidade e veto.

Todas usam a mesma política de replay/trailing `micro_8_3`, latência `100ms`,
custo esperado `1.0` bps e custo stress `2.0` bps.

## Validação do runner

Teste unitário:

```bash
PYTHONPATH=backend ./backend/.venv/bin/pytest \
  backend/tests/unit/test_fast90_near_threshold_scout.py -q --no-cov
```

Resultado: `7 passed`.

Lint:

```bash
./backend/.venv/bin/ruff check \
  backend/scripts/research/run_fast90_near_threshold_scout.py \
  backend/tests/unit/test_fast90_near_threshold_scout.py
```

Resultado: `All checks passed!`.

## Fast89 — janela prospectiva recém-fechada

Comando:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast90_near_threshold_scout.py \
  --input-path backend/data/microstructure_v1/reports/fast84-top-p-silence-diagnostic-fast89/all_scored_decisions.jsonl \
  --wal-roots fast89=backend/data/microstructure_v1/prospective-wal-fast89-30m \
  --output-root backend/data/microstructure_v1/reports/fast90-near-threshold-scout-fast89 \
  --probability-margins 0.02,0.05
```

Resultado:

- status: `FAST90_NO_NEAR_THRESHOLD_EDGE_FOUND`;
- scored rows: `666`;
- candidate universe: `5`;
- complete outcomes: `4`;
- incomplete outcomes: `1`;
- melhores variantes com `threshold - 0.05` ainda foram negativas:
  - `near050_all_top_probability`: `2` trades, stress `-3.33493676930327`
    bps;
  - `near050_flow5_positive_top_probability`: `2` trades, stress
    `-3.33493676930327` bps;
  - `near050_return5_and_flow5_positive_top_probability`: `1` trade, stress
    `-2.5575852053916295` bps;
  - `near050_vol15_flow5_positive_top_probability`: `0` trades.

Interpretação: no Fast89, baixar o threshold teria colocado o bot em trades
ruins ou insuficientes. Não havia edge escondido logo abaixo do top-p.

## Histórico local

Comando:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast90_near_threshold_scout.py \
  --input-path backend/data/microstructure_v1/reports/fast84-top-p-silence-diagnostic-all-local-wals/all_scored_decisions.jsonl \
  --output-root backend/data/microstructure_v1/reports/fast90-near-threshold-scout-all-local \
  --probability-margins 0.02,0.05
```

Resultado:

- status: `FAST90_NO_NEAR_THRESHOLD_EDGE_FOUND`;
- scored rows: `8.026`;
- candidate universe: `3.085`;
- complete outcomes: `2.517`;
- incomplete outcomes: `568`.

Top variantes one-position:

| Variante | Trades | Labels | Stress total | Stress médio | Labels negativos | Pior label |
|---|---:|---:|---:|---:|---:|---:|
| `near020_vol15_flow5_positive_top_probability` | 10 | 3 | `10.970372066877642` | `1.0970372066877643` | 1 | `-16.30108043008083` |
| `near050_vol15_flow5_positive_top_probability` | 10 | 3 | `10.970372066877642` | `1.0970372066877643` | 1 | `-16.30108043008083` |
| `near020_all_veto` | 29 | 10 | `-0.660848673509375` | `-0.022787885293426725` | 5 | `-26.27513141089876` |
| `near020_flow5_positive_top_probability` | 48 | 10 | `-6.280777977224613` | `-0.13084954119217942` | 4 | `-38.60913391670956` |
| `near050_all_veto` | 46 | 12 | `-83.65840473170562` | `-1.8186609724283829` | 6 | `-44.55683256221854` |

O único agregado positivo é o mesmo bolso `vol15 + flow5 positivo` já visto no
Fast87/Fast88. Ele continua frágil porque tem label negativo (`fast20`) e não
aparece no Fast89.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast90 mata um caminho tentador:

- não basta baixar top-p;
- não basta operar near-threshold;
- filtros simples de flow/return sem regime de volatilidade viram custo;
- o único bolso positivo continua raro e instável.

## Implicação

O próximo caminho não deve ser “ajustar threshold para operar mais”. O próximo
caminho precisa ser uma família de entrada nova ou um gate causal mais forte
para separar as poucas janelas boas das ruins antes do trade.

## Arquivos e hashes

- Fast90 runner:
  `3ff94e6cf0d1e857f8dfdb4c7c38e303c6b4bdd3b9e1e959be97ea8f18cd41a8`;
- Fast90 tests:
  `0308f26d9fe03fea7ffba8f53a43aa309459546973f4122455d803e4786ceed5`;
- Fast90 Fast89 report:
  `10368db2b972f8aa62f863e99bde88411dd22a9c6461fa97bfdc6ae19cdf9ba2`;
- Fast90 all-local report:
  `966ce6d6869cbd41db3df4a1aabbf0dc407d769259d715b714384726375daa03`;
- Fast89 independent settlements:
  `6942bb30c3ca42d92e50584ab9e13130270757f0051d39d0e0d4c2f299a1d5e1`;
- all-local independent settlements:
  `f62eaf6f337eb9a525433934460dbc799f5d955e0a9c6f40aa44ce9a5f2673c8`.
