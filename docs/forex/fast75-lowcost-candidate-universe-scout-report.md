# Fast75 — scout no universo amplo lowcost

Resposta objetiva: ainda não temos estratégia.

Fast75 testou uma mudança estrutural em relação ao Fast74: parar de herdar o
stream top-p antigo e procurar edge no universo amplo de `research_rows`
lowcost.

Este teste é deliberadamente só scout. O label usa MFE/terminal/first-touch para
aproximar payoff de path e é otimista para alvos acima do touch nativo do
dataset, porque não temos a ordem exata de toque para esses alvos.

## Safety boundary

Fast75 mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Setup

- universo: `research-lowcost-fast42` até `research-lowcost-fast53`;
- extensão: `research-lowcost-fast56-holdout-new`;
- features causais: `35`;
- modelo: `Ridge(alpha=10)`;
- custos explícitos: `1 bps`;
- stop inicial: `20 bps`;
- horizontes testados: `30s`, `120s`, `300s`;
- alvos aproximados: `1`, `2`, `3`, `5`, `8` bps;
- top fractions: `0.2%`, `0.5%`, `1%`, `2%`, `5%`, `10%`;
- screen barato: amostra a cada `300s`;
- validação: amostra a cada `60s`;
- one-position: não entra em novo sinal enquanto a posição anterior está aberta.

## Screen barato

Foram testadas `90` combinações.

Só uma passou o screen com histórico e Fast56 positivos:

- horizonte: `300s`;
- alvo aproximado: `8 bps`;
- top fraction: `1%`;
- histórico screen: `386` trades, `+99.06892194676057` bps;
- Fast56 screen: `10` trades, `+4.887106199284151` bps.

Esse resultado parecia promissor o suficiente para validar em amostragem mais
densa.

## Validação densa

Na validação a mesma candidata quebrou:

- histórico validado: `1821` trades, `-1688.8399620864152` bps;
- média: `-0.9274244712171418` bps/trade;
- blocos suficientes: `11`;
- blocos positivos: `0`;
- blocos negativos: `11`;
- pior bloco: `-306.3527129822034` bps;
- Fast56 validado: `81` trades, `-269.39530019858205` bps.

Todos os blocos históricos ficaram negativos:

- `research-lowcost-fast42-holdout`: `-306.3527129822034` bps;
- `research-lowcost-fast44-holdout2`: `-175.43522531892955` bps;
- `research-lowcost-fast45-holdout3`: `-38.99598210266297` bps;
- `research-lowcost-fast46-holdout`: `-236.51184283416393` bps;
- `research-lowcost-fast47-holdout`: `-227.88734552823647` bps;
- `research-lowcost-fast48-holdout`: `-196.6725943667268` bps;
- `research-lowcost-fast49-holdout`: `-155.81297952655171` bps;
- `research-lowcost-fast50-holdout`: `-114.03339747033735` bps;
- `research-lowcost-fast51-holdout`: `-32.78882114120552` bps;
- `research-lowcost-fast52-holdout`: `-92.25420631236187` bps;
- `research-lowcost-fast53-holdout`: `-112.09485450303546` bps.

## Arquivos

- script: `backend/scripts/research/run_fast75_lowcost_candidate_universe_scout.py`;
- relatório JSON: `backend/data/microstructure_v1/reports/fast75-lowcost-candidate-universe-scout/report.json`;
- screen: `backend/data/microstructure_v1/reports/fast75-lowcost-candidate-universe-scout/screen_results.json`;
- validações: `backend/data/microstructure_v1/reports/fast75-lowcost-candidate-universe-scout/validations.json`.

Hashes:

- report: `a357328dbb3bd591b784d8433e0a915c430736d305022a1e73aa352ac39f9369`;
- screen: `7ed2764d5f842a1dc3d345c3571afaa3e129b2e111a47c00ab5ef945a088f1f0`;
- validations: `9d2da96f13f1e2398ea1e62a7dfbc8485121a4a7bbf2e21d00cca7683ea09254`.

## Decisão

Status:

`FAST75_LOWCOST_UNIVERSE_FAILED_VALIDATION`

Não aprovar estratégia. Não Testnet. Não LIVE.

Interpretação: ampliar o universo para `research_rows` lowcost não resgatou o
edge. O screen barato criou falso positivo; quando a frequência de decisão
aumentou, o custo/ruído destruiu tudo. Isso reforça que a base sólida não está
em "mais top-p" nem em "mais amostra do mesmo micro-horizonte" com esse label.
