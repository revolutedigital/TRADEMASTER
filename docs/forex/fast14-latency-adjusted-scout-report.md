# Rodada 14 — top-p com label ajustado por latência

Data: 2026-09-23  
Mercado: BTCUSDT futures, microestrutura local `research-v1`, 7 dias  
Resultado: hipótese de latência isolada rejeitada no scout

## Pergunta

A rodada top-p anterior separava bem o evento binário “andou o bastante antes do stop”, mas o replay
com breakeven/trailing continuava negativo. Uma suspeita plausível era mismatch de execução: o label
era calculado no preço do timestamp de decisão, enquanto o replay entrava depois de 100 ms de latência.

Esta rodada reconstruiu o target mantendo as features causais no timestamp original, mas recalculando
o caminho de entrada a partir de `decision_time + 100 ms`. O teste continua sendo apenas scout de
desenvolvimento, porque a amostra local tem só 7 dias e não contém book/depth prospectivo completo.

## O que foi rodado

Comando:

`PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_latency_adjusted_top_p_scout.py --target stress --latency-ms 100`

O script:

- lê `backend/data/microstructure_v1/research-v1`;
- infere o contrato de label dos manifests legados;
- recalcula `latency_paid_stress_before_stop`;
- roda walk-forward calibrado em `flow`, `flow_price` e `flow_price_session`, horizontes 120 e 300s;
- seleciona o melhor candidato top-p monotônico;
- passa esse candidato pelo replay de portfólio com uma posição por vez e políticas de trailing v1.

## Resultado

O modelo continuou conseguindo ordenar probabilidade do target binário. O candidato selecionado foi:

| Campo | Valor |
|---|---:|
| Feature set | `flow_price_session` |
| Horizonte | 300s |
| ROC AUC médio | 0,7316 |
| Top-p monotônico | sim |
| Eventos selecionados no top 1% | 4.453 |
| Taxa do target no top 1% | 26,00% |

Mas isso ainda não virou PnL. A melhor política de replay foi `wide`, top 5%, e ficou negativa:

| Métrica | Valor |
|---|---:|
| Trades | 258 |
| Dias distintos | 3 |
| Média expected | -10,903 bps |
| Média stress | -22,903 bps |
| Win rate expected | 17,05% |
| Drawdown máximo | 2.833 bps |

Todas as 20 combinações de política/trailing ficaram negativas em expected e stress.

## Decisão

Esta hipótese não encontrou estratégia. O problema não era apenas o label ignorar 100 ms de latência.
Mesmo quando o target é ajustado para o caminho executável depois da latência, selecionar “maior chance
de tocar o alvo antes do stop” ainda não produz payoff líquido no replay com trailing.

O aprendizado novo é mais específico: o próximo teste não deve otimizar só a probabilidade de toque.
Ele precisa modelar payoff executável diretamente — por exemplo, ranking por retorno líquido esperado
sob a própria política de execução — ou então esperar a base prospectiva com book/depth, microprice,
spread real e spot-perp para trocar o tipo de informação usada na entrada.

Nenhuma ordem foi enviada e este resultado não autoriza Demo, Testnet ou LIVE.

## Artefatos

- Relatório JSON ignorado pelo Git: `backend/data/microstructure_v1/reports/latency-adjusted-scout-stress/report.json`
  (`7462723379a8d28ca3e2f6172bae4a95c060e35fd997f43881d0e19b59d5852d`)
- Script: `backend/scripts/research/run_latency_adjusted_top_p_scout.py`
  (`5f1d2df4a9b9cfc05f3f245f45c4e2df038fdad582895fc8a95852368f06c54f`)
- Teste: `backend/tests/unit/test_latency_adjusted_top_p_scout.py`
  (`66b3dae1cc600709af1fcadcaecfd276da62f6f248fd616b9864ec6c817f881e`)
