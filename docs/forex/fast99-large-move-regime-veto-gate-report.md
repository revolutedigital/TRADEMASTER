# Fast99 — gate causal no near-edge de operações maiores

Resposta objetiva: o gate melhorou a leitura do ramo de operações maiores, mas
ainda não gerou estratégia.

Fast98 encontrou um candidato raro, porém interessante:

- `hist_gradient_payoff`;
- target `10 bps`;
- trailing distance `15 bps`;
- top fraction `5%`;
- positivo em `9/10` janelas a `1 bps`;
- positivo em `8/10` janelas a `2 bps`;
- mas com apenas `28` execuções one-position no total.

Fast99 testou se um gate causal conseguiria filtrar melhor esse candidato.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `approved_strategy=false`;
- sem Demo;
- sem Testnet;
- sem LIVE;
- sem envio de ordem.

## Configuração

Runner:

`backend/scripts/research/run_fast96_fast95_regime_gate_audit.py`

Comando:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast96_fast95_regime_gate_audit.py \
  --output-root backend/data/microstructure_v1/reports/fast99-large-move-regime-veto-gate-audit \
  --horizon-seconds 300 \
  --target-bps 10 \
  --trailing-distance-bps 15 \
  --top-fraction 0.05 \
  --minimum-train-trades 10 \
  --minimum-train-labels 7 \
  --minimum-holdout-trades 1 \
  --maximum-train-negative-labels 0
```

Observação: `minimum_holdout_trades=1` é permissivo. Foi usado porque o
candidato maior é raro. Com o mínimo operacional antigo de `5` execuções por
janela, nenhum holdout positivo deste run seria suficiente.

## Status

`FAST96_FAST95_REGIME_GATE_FAILED_REPLICATION`

Motivos:

- `profit_positive_holdouts:7/10`;
- `stress_positive_holdouts:5/10`;
- `profit_negative_holdouts:1`;
- `no_strict_training_survivors:2`;
- `do_not_promote_to_strategy`.

## Resultado agregado

| Métrica | Valor |
|---|---:|
| Holdouts | `10` |
| Training survivors | `8` |
| Sem survivor no treino | `2` |
| Profit-positive @ 1 bps | `7/10` |
| Stress-positive @ 2 bps | `5/10` |
| Profit-negative | `1` |
| Holdout trades avaliados | `15` |
| Holdout net total @ 1 bps | `+66.24309268597169` bps |
| Holdout net total @ 2 bps | `+51.24309268597169` bps |
| Pior holdout @ 1 bps | `-1.0` bps |

## Por janela

| Holdout | Status | Execuções | Net @ 1 bps | Net @ 2 bps |
|---|---:|---:|---:|---:|
| fast16 | stress-positive | `1` | `+4.343264375832244` | `+3.343264375832244` |
| fast20 | sem survivor | `0` | `0` | `0` |
| fast21 | sem survivor | `0` | `0` | `0` |
| fast22 | stress-positive | `2` | `+9.135633336806277` | `+7.135633336806277` |
| fast24 | stress-fragile | `1` | `+0.9824347129324904` | `-0.017565287067509594` |
| fast28 | stress-positive | `2` | `+15.671566115933096` | `+13.671566115933096` |
| fast29 | stress-fragile | `2` | `+1.0151562378284398` | `-0.9848437621715602` |
| fast32 | stress-positive | `2` | `+7.88944617723504` | `+5.88944617723504` |
| fast35 | stress-positive | `4` | `+28.205591729404098` | `+24.205591729404098` |
| fast36 | negativo | `1` | `-1.0` | `-2.0` |

## Leitura

Fast99 reforça a tese de que operações maiores são mais promissoras que o
micro-alvo de `3 bps`, mas a base atual não sustenta operação:

- o gate achou filtros causais em `8/10` holdouts;
- o pior prejuízo foi pequeno (`-1 bps`), mas isso veio com amostra minúscula;
- nenhum holdout positivo chegou a `5` execuções;
- fast20 e fast21 ficaram sem survivor;
- fast36 quebrou.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast99 é um diagnóstico promissor/frágil, não uma estratégia. O próximo teste
mais sensato, se continuarmos neste ramo, é usar o candidato maior com top mais
aberto (`top 20%`) para dar mais amostra ao gate, ou baixar/coletar mais janelas
antes de exigir conclusão operacional.

## Arquivos e hashes

- report:
  `backend/data/microstructure_v1/reports/fast99-large-move-regime-veto-gate-audit/report.json`;
- report sha256:
  `14d120bf452ffa3dbdd973b11327c07f2437297dc25dc8ed144322f1530068ea`;
- holdout results:
  `backend/data/microstructure_v1/reports/fast99-large-move-regime-veto-gate-audit/holdout_results.jsonl`;
- holdout results sha256:
  `f2d374f2286ba091c5aca863e1a9e3e18cf5eda4f6ff9a0b828f42c54d4111f6`.
