# Fast84 — diagnóstico do silêncio top-p no Fast82

Resposta objetiva: o Fast82 não gerou trade porque ficou levemente abaixo do
threshold top-p e porque a regra Fast80 de vol+flow não apareceu na janela.

Este diagnóstico não muda threshold, não recalibra modelo e não aprova
estratégia. Ele apenas reprocessa o mesmo WAL Fast82 exportando também os
`non-entries`, para entender quão longe o modelo ficou de entrar.

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
  backend/scripts/research/run_fast84_top_p_silence_diagnostic.py \
  --wal-roots fast82=backend/data/microstructure_v1/prospective-wal-fast82-30m \
  --output-root backend/data/microstructure_v1/reports/fast84-top-p-silence-diagnostic-fast82
```

## Resultado

- status: `FAST84_TOP_P_NEAR_THRESHOLD_RESEARCH_ONLY`;
- scored rows: `678`;
- threshold congelado: `0.90195311025804`;
- entradas top-p: `0`;
- maior probabilidade observada: `0.89571932347643`;
- distância até o threshold: `0.006233786781610018`;
- rows dentro de `0.01` do threshold: `1`;
- rows dentro de `0.02` do threshold: `6`;
- rows dentro de `0.05` do threshold: `89`;
- rows dentro de `0.10` do threshold: `513`.

Distribuição:

| Métrica | Probabilidade |
|---|---:|
| min | `0.7400701519307765` |
| p25 | `0.8031368691678789` |
| p50 | `0.8238494556219971` |
| p75 | `0.8405775880392728` |
| p90 | `0.8555976472535604` |
| p95 | `0.862527582319635` |
| p99 | `0.8816864726946919` |
| max | `0.89571932347643` |
| mean | `0.8226037520326764` |

## Diagnóstico Fast80

- rows com features exigidas: `678`;
- rule matches Fast80: `0`;
- rule matches que também entrariam top-p: `0`;
- maior probabilidade entre rule matches: `null`.

As top linhas chegaram perto do threshold, mas não no regime Fast80. Exemplo:

- melhor row: `2026-09-23T19:07:35Z`, `SELL`, probabilidade
  `0.89571932347643`;
- `realized_vol_30s_bps=0.7674571944414029`;
- `flow_imbalance_5s=-0.4606763427648517`;
- Fast80 exige `realized_vol_30s_bps >= 45.38807193200911` e
  `flow_imbalance_5s <= -0.2193072079643347`.

Ou seja: o fluxo até ajudou, mas a volatilidade realizada ficou muito longe do
regime congelado.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Este diagnóstico muda a leitura do Fast82:

1. o modelo top-p não está completamente morto — ele ficou perto do threshold;
2. a regra Fast80 é extremamente dependente de regime de volatilidade;
3. a próxima validação precisa acumular mais janelas pós-pré-registro e separar
   explicitamente dois gargalos:
   - frequência do top-p;
   - frequência do regime Fast80 vol+flow.

## Arquivos e hashes

- Fast84 script:
  `a79d5c620b8afe40a9562b12b861dd5d958cb90669f4daa576f44f27daf362b6`;
- Fast84 tests:
  `b0f8ce7f6e174458df58243c4b3eefd565f5d81f27b98c218de141975a9e4167`;
- Fast84 report:
  `4b322ee63040d580a7190eba1faf27f8d119ce215c10d9d9dd3b05437214bdba`;
- all scored decisions:
  `bb696932b429c8f11e239a15500bd6cfe44f04a7adb842e127c32ceee3bcef71`;
- top scored decisions:
  `122d19489c05f82dfab613a71c2b90a9b7159443a72d620020b6cce3bfcc05ec`.

## Validação

```bash
PYTHONPATH=backend ./backend/.venv/bin/pytest \
  backend/tests/unit/test_fast84_top_p_silence_diagnostic.py -q --no-cov
```

Resultado: `5 passed`.

```bash
./backend/.venv/bin/ruff check \
  backend/scripts/research/run_fast84_top_p_silence_diagnostic.py \
  backend/tests/unit/test_fast84_top_p_silence_diagnostic.py
```

Resultado: `All checks passed!`.
