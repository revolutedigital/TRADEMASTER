# Fast108 — auditoria por par das políticas FX Fast6

Resposta objetiva: ainda não temos estratégia. A hipótese de que o agregado do
Fast6 escondia um par específico bom também falhou.

Fast6 rejeitou a família FX Q2 com trailing global. Fast108 pergunta uma coisa
menor e útil: se olharmos cada par individualmente, alguma política vira positiva
em base e stress com estabilidade mensal?

## Safety boundary

Mantido em todos os passos:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `approved_strategy=false`;
- `dry_run=true`;
- sem Demo;
- sem Testnet;
- sem LIVE;
- sem envio de ordem.

## Dados auditados

Fonte:

`backend/data/lab_fast6/policies/*-trades.parquet`

Escopo:

- políticas avaliadas: `18`;
- combinações par+política: `72`;
- janela: Q2 2021, já materializada pelo Fast6;
- métricas: `base_r`, `stress_r`, estabilidade mensal.

Critério strict:

- mínimo `10` trades;
- mínimo `2` meses;
- média base positiva;
- média stress positiva;
- todos os meses com média stress positiva.

Critério weak:

- mínimo `10` trades;
- mínimo `2` meses;
- média base positiva;
- média stress positiva.

## Resultado

Status:

`FAST108_PAIR_SPECIFIC_FAILED_SCREEN`

Resumo:

| Métrica | Valor |
|---|---:|
| Políticas | `18` |
| Combinações par+política | `72` |
| Weak candidates | `0` |
| Strict candidates | `0` |
| Estratégia aprovada | `false` |

Nem o critério fraco encontrou um par positivo em base e stress.

## Melhores diagnósticos

O melhor por média stress ainda foi negativo:

| Política | Par | Trades | Média base R | Média stress R | Meses stress positivos |
|---|---|---:|---:|---:|---:|
| `global-top0p25-t0p5` | EURUSD | `55` | `-0.09953211449931024` | `-0.21899509428787628` | `2/3` |
| `global-top0p5-t0p5` | EURUSD | `55` | `-0.09953211449931024` | `-0.21899509428787628` | `2/3` |
| `global-top1p0-t0p5` | EURUSD | `55` | `-0.09953211449931024` | `-0.21899509428787628` | `2/3` |
| `global-top5p0-t1p5` | EURJPY | `30` | `+0.007299703659973818` | `-0.2592187230712224` | `1/3` |
| `global-top5p0-t1p0` | EURJPY | `33` | `-0.042153314268916484` | `-0.2860945340112499` | `1/3` |

Mesmo onde a base quase melhora, o stress continua negativo.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast108 fecha uma lacuna importante: o problema do Fast6 não era apenas misturar
pares ruins com pares bons. Nos trades materializados do Fast6, nenhum par
isolado sobrevive.

## O que isso ensina

- O edge não está escondido em um par específico dentro dessa família.
- O custo/stress continua destruindo a média mesmo quando filtramos por par.
- Não vale reabrir Fast6 com ajuste cosmético de top-p/trailing.
- A próxima mudança precisa ser estrutural: outro tipo de sinal, outro regime,
  ou observação prospectiva das poucas pistas macro/vol já isoladas.

## Comando executado

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast108_fx_fast6_pair_audit.py
```

## Verificações

```bash
./backend/.venv/bin/python -m pytest \
  backend/tests/unit/test_fast108_fx_fast6_pair_audit.py -q --no-cov

./backend/.venv/bin/python -m ruff check \
  backend/scripts/research/run_fast108_fx_fast6_pair_audit.py \
  backend/tests/unit/test_fast108_fx_fast6_pair_audit.py
```

Resultado:

- `2 passed`;
- `All checks passed!`.

## Arquivos e hashes

- report:
  `backend/data/lab_fast108/fx-fast6-pair-audit/report.json`;
- report sha256:
  `b2ecceeff1629993b65e5f8575883404fd7dcdc75d955a1c01e5ccbb4fa51104`;
- avaliações par+política sha256:
  `2f14e44389942f8c01431a2f4f53bae0b70b2c0882c582336525e573957f0754`;
- strict candidates sha256:
  `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`;
- top diagnostics sha256:
  `89f29d6fc458c6dd1555adc6deecf9b383a9d6337ae1317a013cb2f7c7a54935`.
