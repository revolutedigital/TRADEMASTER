# Fast80 — runner de observação vol+flow

Resposta objetiva: ainda não temos estratégia, e o runner confirmou que o dado
atual não é scoreável para a regra Fast80.

Fast80 implementa o pré-registro do setup raro:

`realized_vol_30s_bps >= 45.38807193200911 AND flow_imbalance_5s <= -0.2193072079643347`

Ele é fail-closed: se uma feature exigida não existir, o runner não tenta
aproximar, não substitui por outra coluna e não avalia o sinal.

## Safety boundary

O runner mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`;
- `testnet_allowed=false`;
- `live_allowed=false`.

## Execução inicial

Input usado:

`backend/data/microstructure_v1/reports/fast61-fast60-shadow-settlement/one_position_settlements.jsonl`

Hash:

`b33a1bae5f2a28c32a43250993a5538f28e820624a39d5100c835958157f6bd8`

Resultado:

- status: `FAST80_OBSERVATION_NOT_SCOREABLE_MISSING_FEATURES`;
- registros de input: `8`;
- registros pós-pré-registro: `0`;
- observações elegíveis: `0`;
- ordens enviadas: `0`;
- Testnet: não permitido;
- LIVE: não permitido.

Feature audit:

- `realized_vol_30s_bps`: ausente em `8/8` registros;
- `flow_imbalance_5s`: ausente em `8/8` registros.

Isso é o comportamento correto. O settlement atual foi suficiente para avaliar
Fast61, mas não carrega o vetor causal necessário para o Fast80.

## Gates implementados

### Falha precoce

A hipótese falha se:

- após os primeiros `20` trades one-position prospectivos, `stress_total <= -20`
  bps; ou
- qualquer bloco prospectivo com pelo menos `10` trades fechar com
  `stress_total <= -10` bps; ou
- qualquer feature exigida precisar ser aproximada, imputada ou substituída.

### Mínimo para candidato de shadow offline

Mesmo se os próximos dados parecerem bons, só vira candidato para revisão de
shadow offline se:

- houver pelo menos `100` trades one-position prospectivos;
- houver pelo menos `10` dias distintos;
- stress médio > `+1.5` bps/trade;
- expected médio > `+2.5` bps/trade;
- lower confidence bound unilateral de stress, sem correção, > `0`;
- nenhum bloco prospectivo suficiente negativo;
- nenhuma regra alterada desde o pré-registro.

Passar esse gate ainda não libera Testnet.

## Arquivos

- runner: `backend/scripts/research/run_fast80_rare_vol_flow_observation.py`;
- teste: `backend/tests/unit/test_fast80_rare_vol_flow_observation.py`;
- relatório JSON:
  `backend/data/microstructure_v1/reports/fast80-rare-vol-flow-observation/report.json`;
- observações JSONL:
  `backend/data/microstructure_v1/reports/fast80-rare-vol-flow-observation/observations.jsonl`.

Hashes:

- report: `4e8a5de839a3751c95946d0961ca0bffe0d85c040807386763af4a78626a5496`;
- observations:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

## Decisão

Status:

`FAST80_OBSERVATION_NOT_SCOREABLE_MISSING_FEATURES`

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Próximo passo técnico: corrigir a captura/settlement prospectivo para gravar
`realized_vol_30s_bps` e `flow_imbalance_5s` junto do sinal top-p. Sem isso, a
hipótese Fast80 não deve ser avaliada.

## Correção de captura implementada

Depois da execução inicial, o gravador de decisões Fast60/Fast69 foi ajustado
para decisões futuras exportarem:

- `causal_feature_values`, com o vetor causal calculado no momento da decisão;
- `realized_vol_30s_bps` no topo da linha, quando existir no vetor;
- `flow_imbalance_5s` no topo da linha, quando existir no vetor;
- aliases já usados por gates anteriores, como `quote_volume_30s` e
  `directed_flow_imbalance_30s`.

Isso não altera scoring, não recalibra regra e não torna o dado antigo
scoreável. Só garante que as próximas decisões top-p tenham evidência causal
suficiente para o runner Fast80 aceitar ou rejeitar a hipótese sem proxy.
