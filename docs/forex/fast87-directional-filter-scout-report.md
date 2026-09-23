# Fast87 — scout de filtro direcional para nova família

Resposta objetiva: apareceu uma pista melhor que Fast80, mas ainda não é
estratégia.

Fast87 parte do aprendizado anterior:

- Fast80 direto falhou;
- conflitos BUY/SELL mataram o resultado;
- alta volatilidade sozinha não basta;
- precisamos de direção side-aligned.

O scout liquidou todos os top-p locais e testou variantes com:

- `realized_vol_30s_bps`;
- `directed_return_*`;
- `directed_flow_imbalance_*`;
- veto/top-probability em conflitos BUY/SELL;
- avaliação one-position + stress cost.

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
  backend/scripts/research/run_fast87_directional_filter_scout.py \
  --output-root backend/data/microstructure_v1/reports/fast87-directional-filter-scout
```

Input:

`backend/data/microstructure_v1/reports/fast84-top-p-silence-diagnostic-all-local-wals/all_scored_decisions.jsonl`

## Resultado

- status: `FAST87_NO_ROBUST_DIRECTIONAL_FILTER_FOUND`;
- scored rows: `8.026`;
- top-p candidates: `550`;
- top-p outcomes completos: `333`;
- melhor família diagnóstica:
  `vol15_directed_flow_5s_positive`;
- one-position stress total do melhor agregado:
  `+10.970372066877642` bps;
- trade count: `10`;
- labels positivos: `2`;
- labels negativos: `1`;
- pior label:
  `-16.30108043008083` bps.

## Top variante

`vol15_directed_flow_5s_positive`

Regra diagnóstica:

- top-p congelado;
- `realized_vol_30s_bps >= 15`;
- `directed_flow_imbalance_5s > 0`.

One-position:

| Métrica | Valor |
|---|---:|
| trades | `10` |
| expected total | `20.970372066877644` bps |
| stress total | `10.970372066877642` bps |
| stress mean | `1.0970372066877643` bps |
| expected win rate | `0.7` |
| max drawdown | `34.046413984737846` bps |
| positive labels | `2` |
| negative labels | `1` |

Por label:

| Label | Trades | Stress total |
|---|---:|---:|
| fast20 | 1 | `-16.30108043008083` |
| fast28 | 1 | `11.699169137121372` |
| fast36 | 8 | `15.572283359837101` |

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast87 achou uma pista melhor que Fast80, mas ainda insuficiente:

1. o agregado ficou positivo em stress;
2. o resultado depende de poucos labels;
3. existe uma janela negativa relevante;
4. o scout é pós-outcome, então qualquer variante positiva exige novo
   pré-registro antes de ser levada a sério.

Próximo caminho real: pré-registrar uma nova observação baseada nessa família
somente se a regra for simples e congelada antes de novos WALs:

`top-p + vol30>=15 + directed_flow_imbalance_5s>0 + one-position + stress cost`.

Mas ela ainda precisa provar:

- frequência prospectiva;
- stress positivo em janelas novas;
- nenhuma concentração absurda em um único label;
- robustez contra uma janela negativa.

## Arquivos e hashes

- Fast87 script:
  `9c8ad9214c3cd466b06ff0a388d507bc7a0aa954e8f873e06eb79bb3206b655f`;
- Fast87 tests:
  `20597873f5168b661dd426453d2d7d302eff59acadc2db6d16768a6f85838345`;
- Fast87 report:
  `571599fc88b594928af6fbec2f1d068f30f70d0b49ec92c98a9f86c98284cb28`;
- variant rows:
  `24d85f78f133b570f15d533083aa40778f28f7c138f34df800ccc740eb14b09b`;
- all top-p independent settlements:
  `5ff05eb43627feb3693eec77743c01bd3af0fb68c109aaee2a703929d8a0970b`.

## Validação

```bash
PYTHONPATH=backend ./backend/.venv/bin/pytest \
  backend/tests/unit/test_fast87_directional_filter_scout.py -q --no-cov
```

Resultado: `7 passed`.

```bash
./backend/.venv/bin/ruff check \
  backend/scripts/research/run_fast87_directional_filter_scout.py \
  backend/tests/unit/test_fast87_directional_filter_scout.py
```

Resultado: `All checks passed!`.
