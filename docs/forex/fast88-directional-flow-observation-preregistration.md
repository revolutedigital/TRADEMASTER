# Fast88 — pré-registro de observação directional-flow

Registrado em `2026-09-23T20:18:35Z`, depois dos audits Fast85, Fast86 e
Fast87. Este documento congela uma nova observação, não uma estratégia.

Fast88 não autoriza Demo, Testnet, LIVE, envio de ordem, commit de ledger ou
operação manual.

## Origem da hipótese

Fast80 direto foi rejeitado:

- Fast85 liquidou os matches Fast80 históricos e falhou em stress;
- Fast86 mostrou que filtros simples de conflito direcional não salvam Fast80;
- Fast87 testou filtros directionais side-aligned e encontrou uma pista
  positiva, porém não robusta.

Pista Fast87:

- variante: `vol15_directed_flow_5s_positive`;
- regra diagnóstica: top-p + `realized_vol_30s_bps >= 15` +
  `directed_flow_imbalance_5s > 0`;
- one-position trades: `10`;
- one-position stress total: `+10.970372066877642` bps;
- labels positivos: `2`;
- labels negativos: `1`;
- pior label: `-16.30108043008083` bps.

Conclusão de partida: observável, não estratégia.

## Safety boundary

Durante todo o Fast88:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

Se qualquer implementação futura tentar enviar ordem ou gravar ledger real com
base neste pré-registro, isso é defeito de segurança.

## Hipótese congelada

Base obrigatória:

- mesmo top-p/modelo congelado da família Fast57/Fast69;
- modelo SHA-256:
  `9878346a6d225cbc29a5cd17c70e039ebeea0e4831e93111243eded42122ba68`;
- mercado: `BTCUSDT` perpetual;
- horizonte: `300s`;
- avaliação: one-position;
- custo primário de decisão: `stress_net_bps`;
- política de replay/trailing: `micro_8_3`, igual aos audits Fast61/Fast85.

Regra congelada:

1. `would_enter == true` pelo top-p congelado;
2. `realized_vol_30s_bps >= 15.0`;
3. `directed_flow_imbalance_5s > 0.0`.

Conflito congelado:

- se `BUY` e `SELL` passarem no mesmo timestamp/label, manter apenas o lado com
  maior `probability`;
- depois disso, aplicar uma posição por vez usando `exit_time_ms`.

Nenhum outro threshold, feature, lado, sessão, densidade, horizonte, top-p ou
trailing entra no Fast88.

## Requisito de captura

O WAL/runner prospectivo precisa registrar, no momento da decisão:

- `realized_vol_30s_bps`;
- `directed_flow_imbalance_5s`;
- vetor causal completo do top-p;
- `decision_time_ms`;
- `side`;
- `probability`;
- `threshold`;
- settlement offline com `expected_net_bps`, `stress_net_bps`, `exit_time_ms`
  e `exit_reason`.

Se `directed_flow_imbalance_5s` ou `realized_vol_30s_bps` não estiverem
disponíveis, o runner deve falhar fechado e não aproximar por outra feature.

## Janela prospectiva

São elegíveis apenas decisões com `decision_time > 2026-09-23T20:18:35Z`.

Qualquer decisão anterior a esse timestamp fica fora, mesmo que seja
materializada depois.

A avaliação prospectiva não pode usar decisões futuras para alterar:

- modelo top-p;
- thresholds;
- features;
- custo;
- trailing;
- tratamento de conflito;
- regra de one-position.

## Gates

### Gate de falha precoce

A hipótese é encerrada como falha se qualquer condição ocorrer:

- após os primeiros `20` trades one-position prospectivos,
  `stress_total <= -20` bps;
- qualquer bloco prospectivo com pelo menos `10` trades fechar com
  `stress_total <= -10` bps;
- qualquer feature exigida precisar ser aproximada, imputada ou substituída;
- qualquer implementação usar dado anterior a `2026-09-23T20:18:35Z` como
  confirmação prospectiva.

### Gate mínimo para candidato de shadow offline

Mesmo que pareça bom antes, não há promoção antes de:

- pelo menos `100` trades one-position prospectivos;
- pelo menos `10` dias distintos;
- stress médio > `+1.5` bps/trade;
- expected médio > `+2.5` bps/trade;
- lower confidence bound unilateral de stress, sem correção, > `0`;
- nenhum bloco prospectivo suficiente negativo;
- nenhuma regra alterada desde este documento.

Passar esse gate não aprova Testnet. Passar esse gate só autoriza escrever um
novo relatório propondo shadow offline, ainda `research_only`.

### Gate para Testnet

Fast88 não tem gate de Testnet. Testnet exigiria novo relatório, novo
pré-registro de execução, auditoria de book/freshness/fill/slippage e
autorização explícita do Igor.

## O que NÃO será feito

- Baixar o threshold `realized_vol_30s_bps >= 15.0` depois de ver dado novo.
- Trocar `directed_flow_imbalance_5s > 0` por outra janela de flow.
- Usar Fast20/Fast28/Fast36 como confirmação prospectiva.
- Ignorar a label negativa vista no Fast87.
- Promover variante pelo agregado positivo sem confirmação prospectiva.
- Chamar Fast88 de estratégia antes dos gates acima.

## Próxima implementação

O próximo passo técnico é criar um runner Fast88 que:

- leia decisões top-p com vetor causal completo;
- recuse execução se as features exigidas faltarem;
- aplique a regra congelada;
- resolva conflito por maior `probability`;
- liquide offline;
- aplique one-position;
- calcule gates;
- mantenha `orders_sent=false` sempre.
