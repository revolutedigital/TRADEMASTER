# Fast57 — runner prospectivo do gate fixo

Data: 2026-09-23  
Status: runner implementado e validado em amostra ainda insuficiente  
Escopo: research-only.

## O que foi feito

Foi criado um runner específico para o pré-registro Fast57:

`backend/scripts/research/run_fast57_fixed_density_prospective_gate.py`

Ele executa:

1. candidato congelado com gate `recent_300s_lte_0`;
2. baseline sem gate;
3. comparação contra os critérios pré-registrados;
4. status fail-closed enquanto não houver amostra mínima.

O runner sempre mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `approved_strategy=false`.

## Validação em 2026-09-21 a 2026-09-22

Fonte:

`backend/data/microstructure_v1/reports/fast57-fixed-density-prospective-gate-sep21-sep22/report.json`

Hash:

`1cf968f8aaedd626aa05fe37c47cb608843d5c82cea75ef15232f6822610ffdf`

Status:

`FAST57_INSUFFICIENT_PROSPECTIVE_SAMPLE`

Motivos:

- `complete_days_below_minimum:2<10`;
- `trade_count_below_minimum:28<100`.

## Resultado observado até agora

| Métrica | Gate `5min≤0` | Sem gate |
| --- | ---: | ---: |
| Dias completos | 2 | 2 |
| Trades | 28 | 121 |
| Média 1 bps | +3,4290 | +0,8962 |
| Média 2 bps | +2,4290 | -0,1038 |
| Total 2 bps | +68,0117 | -12,5609 |
| Max DD | 31,2116 | 123,5421 |

Critérios já verdadeiros:

- stress total positivo;
- stress médio positivo;
- 100% dos dias com trades positivos;
- drawdown menor que lucro total;
- gate melhor que sem gate.

Critérios ainda não verdadeiros:

- mínimo de 10 dias completos;
- mínimo de 100 trades.

## Decisão

Nada aprovado.

O runner confirma que o candidato continua promissor, mas ainda está travado por
amostra insuficiente. A próxima ação válida é acumular novos dias completos até
atingir 10 dias e 100 trades, sem alterar modelo/gate/trailing.

Sem Testnet/LIVE.

