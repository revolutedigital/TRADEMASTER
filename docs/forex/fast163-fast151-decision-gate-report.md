# Fast163 — Fast151 decision gate

Fast163 consolidou as auditorias Fast152, Fast156, Fast157, Fast159, Fast161 e
Fast162 em um gate único para a política operacional Fast151:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only:max_per_pair_1:dd800_cooldown180d`

Status:

`FAST163_FAST151_CANDIDATE_BLOCKED_BY_FORWARD_DATA_AND_BROKER_SWAP`

## Resposta curta

Temos uma candidata histórica. Não temos uma estratégia aprovada.

O gate final está fail-closed:

- `shadow_allowed=false`;
- `testnet_allowed=false`;
- `live_allowed=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `approved_strategy=false`.

## Evidência que sustenta a candidata

### Edge histórico

| Amostra | Trades | Média | LCB | Max DD | Total |
|---|---:|---:|---:|---:|---:|
| Pré-2019 | `51` | `+67.4106` | `+32.2878` | `-900.2085` | `+3437.9421` |
| 2019–2026 | `86` | `+33.6173` | `+9.6335` | `-616.6221` | `+2891.0872` |

### Estrutura escolhida

- Cesta: `AUDUSD+EURUSD+GBPUSD+NZDUSD`;
- Gestão: `stop2_fixed_1r`;
- Overlay: `max_per_pair_1`;
- Circuit breaker: `dd800_cooldown180d`;
- Trailing: rejeitado pelo Fast162; melhor trailing ficou pior que `fixed_1r`.

### Sizing

Pela cauda p05 em R:

- pior cauda: `-12.6900R`;
- risco `0.25%` por trade projeta DD de cauda em `3.1725%`;
- risco `0.50%` por trade projeta DD de cauda em `6.3450%`;
- risco `1.00%` por trade projeta DD de cauda em `12.6900%`;
- para budget de DD de `10%`, risco máximo estimado: `0.7880%` por trade.

## Bloqueios duros

1. `post_preregistration_m1_data_missing`
   - M1 atual termina em `2026-09-18T20:59:00+00:00`;
   - Fast151 foi pré-registrado depois disso;
   - ainda não existe dado M1 pós-pré-registro para observação prospectiva.

2. `no_prospective_operational_observation_yet`
   - observações elegíveis: `0`;
   - status Fast151: `FAST151_OBSERVATION_NO_ELIGIBLE_OPERATIONAL_TRADES_YET`.

3. `broker_swap_table_missing`
   - Fast159 mostrou break-even current em `0.9589` pip/rollover;
   - falta tabela real de swap da corretora por par/lado.

4. `explicit_execution_authorization_missing`
   - mesmo se os gates passarem, qualquer shadow/Testnet precisa de autorização
     explícita posterior.

## Próximas ações objetivas

1. Atualizar ou importar M1 pós-pré-registro e rerodar:

   `backend/.venv/bin/python backend/scripts/research/run_fast152_fast151_readiness_monitor.py`

   Critério de pronto:

   - status deixa de ser `FAST152_WAITING_FOR_POST_PREREGISTRATION_M1_DATA`.

2. Preencher a tabela real de swap da corretora:

   `backend/data/lab_fast159/fast151-exact-swap-rollover-audit/broker-swap-template.csv`

   Depois rerodar:

   `backend/.venv/bin/python backend/scripts/research/run_fast159_fast151_exact_swap_rollover_audit.py --broker-swap-path <filled.csv>`

   Critério de pronto:

   - status vira `FAST159_FAST151_BROKER_SWAP_TABLE_PASSES_REQUIRES_PROSPECTIVE_OBSERVATION`.

3. Continuar fail-closed:

   - sem shadow;
   - sem Testnet;
   - sem LIVE;
   - sem ledger;
   - sem ordens.

## Interpretação

Essa é a resposta mais limpa até agora:

- sim, existe uma candidata com edge histórico;
- não, ela ainda não é estratégia aprovada;
- não adianta mais trocar trailing, tirar par, ou inventar filtro simples sem
  nova hipótese/preregistro;
- os próximos bloqueios reais são dados forward e swap real de corretora.

## Segurança

Fast163 é gate de decisão offline:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`.

Nenhum trade foi enviado. Nenhum ledger foi alterado. O resultado não autoriza
Demo, Testnet nem LIVE.

## Artefatos

- runner:
  `backend/scripts/research/run_fast163_fast151_decision_gate.py`;
- teste:
  `backend/tests/unit/test_fast163_fast151_decision_gate.py`;
- report JSON:
  `backend/data/lab_fast163/fast151-decision-gate/report.json`;
- resumo:
  `backend/data/lab_fast163/fast151-decision-gate/decision-gate-summary.json`.

Hashes:

- report:
  `d7bf98297ed74e4ad35ec1948545f73b83b5acd5411bea432e25b669d432aad5`;
- runner:
  `a70a1f64ff3feff141f8e2dc352e42abd9ac0890b3c0e07281b0b2370ad6f8b6`;
- teste:
  `167e0401957c342d2dc681a29d7c8a947cd7644ba2b9a13d18fe3fd06cfa8d91`;
- resumo:
  `eb8310ce3ab44f36011d2d636d7bf1fa9dda5b23620391b7c406adec8e2779a8`.
