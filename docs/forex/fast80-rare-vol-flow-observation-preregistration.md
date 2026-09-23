# Fast80 — pré-registro de observação do setup raro vol+flow

Registrado em 2026-09-23, depois do Fast79 e antes de qualquer nova janela ser
usada para avaliar esta hipótese. Este documento congela uma única regra fraca
de microestrutura para observação prospectiva. Ele não autoriza Demo, Testnet,
LIVE, envio de ordem, commit de ledger ou operação manual.

## Origem da hipótese

Fast79 encontrou weak survivors em pares de condições causais. A regra escolhida
para observação é a que teve melhor stress na extensão Fast56, mesmo com a
ressalva de amostra pequena:

`realized_vol_30s_bps>=q0.80:45.38807193 AND flow_imbalance_5s<=q0.33:-0.219307208`

Histórico Fast43–Fast53:

- one-position trades: `138`;
- stress total: `+418.2238564179386` bps;
- stress médio: `+3.0306076552024535` bps;
- blocos suficientes positivos: `10/10`;
- pior bloco suficiente: `+9.334692784221659` bps.

Extensão Fast56:

- one-position trades: `3`;
- stress total: `+28.635546270573894` bps;
- stress médio: `+9.545182090191298` bps;
- dias distintos: `1`.

Conclusão de partida: hipótese observável, não estratégia.

## Safety boundary

Durante todo o Fast80:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`;

Se qualquer implementação futura tentar enviar ordem ou gravar ledger real com
base neste pré-registro, isso é defeito de segurança.

## Hipótese congelada

Base obrigatória:

- mesmo top-p/modelo congelado da família Fast57/Fast69;
- mercado: `BTCUSDT` perpetual;
- horizonte: `300s`;
- avaliação: one-position;
- custo primário de decisão: `stress_net_bps`;
- política de replay/trailing: a mesma usada nos frozen top-p trades
  Fast43–Fast56.

Regra congelada:

- `realized_vol_30s_bps >= 45.38807193200911`;
- `flow_imbalance_5s <= -0.2193072079643347`.

Nenhum outro threshold, feature, lado, sessão, densidade, horizonte, top-p ou
trailing entra no Fast80.

## Requisito de captura

O WAL/runner prospectivo precisa registrar, no momento da decisão, pelo menos:

- `realized_vol_30s_bps`;
- `flow_imbalance_5s`;
- todos os campos necessários para reproduzir o top-p congelado;
- timestamps de decisão/entrada/saída;
- `expected_net_bps` e `stress_net_bps` de settlement offline;
- razão de saída.

Se `flow_imbalance_5s` não estiver disponível, o runner deve falhar fechado com
`not_scoreable_missing_flow_imbalance_5s`, e não tentar aproximar por outra
feature.

## Janela prospectiva

São elegíveis apenas decisões geradas depois do commit deste documento. Qualquer
decisão anterior ao commit do Fast80 fica fora, mesmo que seja materializada
depois.

A avaliação prospectiva não pode usar decisões futuras para alterar:

- modelo top-p;
- thresholds;
- features;
- custo;
- trailing;
- gate de aprovação;
- regra de one-position.

## Gates

### Gate de falha precoce

A hipótese é encerrada como falha se qualquer condição ocorrer:

- após os primeiros `20` trades one-position prospectivos, `stress_total <= -20`
  bps;
- qualquer bloco prospectivo com pelo menos `10` trades fechar com
  `stress_total <= -10` bps;
- qualquer feature exigida precisar ser aproximada, imputada ou substituída.

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

Fast80 não tem gate de Testnet. Testnet exigiria novo relatório, novo
pré-registro de execução, auditoria de book/freshness/fill/slippage e
autorização explícita do Igor.

## O que NÃO será feito

- Trocar `flow_imbalance_5s` por `directed_flow_imbalance_30s`.
- Usar a regra alternativa de 11 trades do Fast79 como fallback.
- Recalibrar o threshold de volatilidade ou de flow depois de ver dado novo.
- Chamar o resultado de estratégia antes de passar os gates prospectivos.
- Usar Fast56 ou Sep23 como se fossem dados prospectivos do Fast80.

## Próxima implementação

O próximo passo técnico é criar um runner Fast80 que:

- leia WAL/decisões top-p com o vetor causal completo;
- recuse execução se `flow_imbalance_5s` não existir;
- grave observações locais;
- liquide offline;
- calcule os gates acima;
- mantenha `orders_sent=false` sempre.
