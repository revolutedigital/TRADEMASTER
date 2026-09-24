# Fast151 — pré-registro operacional USD-quoted basket

Registrado em `2026-09-24T06:18:19-03:00`, depois dos audits Fast148,
Fast149 e Fast150.

Fast151 não é estratégia aprovada. É o congelamento prospectivo offline da
primeira candidata operacional completa encontrada nesta linha de pesquisa:
entrada + exposição + circuit breaker.

Fast151 não autoriza Demo, Testnet, LIVE, envio de ordem, commit de ledger ou
operação manual.

## Identidade da política congelada

Policy ID operacional:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only:max_per_pair_1:dd800_cooldown180d`

Componentes:

- cesta: `usd_quoted_majors`;
- pares: `AUDUSD`, `EURUSD`, `GBPUSD`, `NZDUSD`;
- timeframe de sinal: `D1`;
- dados de execução: M1 local;
- entrada: `d1_donchian_fade_ch100_hold20`;
- risco/saída: `stop2_fixed_1r`;
- lado: apenas `SHORT`;
- overlay de exposição: `max_per_pair_1`;
- circuit breaker: `dd800_cooldown180d`;
- cenário primário de decisão: `STRESS`.

## Origem da hipótese

Fast146 encontrou a cesta `usd_quoted_majors` como candidata histórica:

- pré-2019: `89` trades, média stress `+37.8272`, LCB `+9.7967`;
- 2019–2026: `143` trades, média stress `+42.3340`, LCB `+24.6986`.

Fast148 mostrou que a cesta bruta tinha edge, mas risco operacional ruim:

- pré-2019: max drawdown `-2230.5874`, max concorrência `6`, dois anos
  negativos consecutivos;
- 2019–2026: max drawdown `-1125.7067`, max concorrência `10`.

Fast149 selecionou o overlay `max_per_pair_1`:

- pré-2019: `58` trades, média stress `+52.0126`, LCB `+18.1781`,
  max drawdown `-1193.7464`, max concorrência `4`;
- 2019–2026: `86` trades, média stress `+33.6173`, LCB `+9.6335`,
  max drawdown `-616.6221`, max concorrência `4`;
- ainda restava o problema de anos negativos consecutivos no pré-2019.

Fast150 selecionou o circuit breaker `dd800_cooldown180d` em cima do
`max_per_pair_1`:

- pré-2019: `51` trades, média stress `+67.4106`, LCB `+32.2878`,
  max drawdown `-900.2085`, max concorrência `4`, sem anos negativos
  consecutivos;
- 2019–2026: `86` trades, média stress `+33.6173`, LCB `+9.6335`,
  max drawdown `-616.6221`, max concorrência `4`, sem anos negativos
  consecutivos.

## Regra congelada de entrada

- Calcular canal Donchian de `100` dias fechados, de forma causal, por par.
- Trade elegível: fechamento D1 acima da máxima causal dos `100` dias
  anteriores.
- Direção: fade vendido contra rompimento superior.
- `LONG` é proibido.
- Entrada: primeiro minuto M1 disponível posterior ao fechamento da barra D1 de
  sinal.
- ATR: ATR(14) diário em pips, deslocado um D1 para trás, por par.
- Stop inicial: `2.0 * ATR(14)` contra a entrada.
- Alvo: `1R`.
- Saída: primeiro evento entre stop, alvo ou expiração de `20` dias.
- Gaps de fim de semana: usar o primeiro preço M1 disponível depois do gap e
  aplicar stop/target de forma conservadora quando a reabertura ultrapassar o
  nível.

Nenhum filtro adicional de sessão, weekday, mês, macro, spread, distância de
rompimento, volatilidade, top-p, trailing, remoção de par, par adicional,
calibração de modelo ou mudança de target está autorizado neste pré-registro.

## Regra congelada de exposição

Overlay `max_per_pair_1`:

- antes de aceitar nova entrada, verificar posições ainda abertas da própria
  política;
- se já houver uma posição aberta no mesmo par, pular o novo trade;
- se não houver posição aberta no mesmo par, o trade pode ser aceito;
- posições abertas são consideradas abertas até o `exit_stress_timestamp`
  offline ou até o evento equivalente no simulador prospectivo;
- o overlay não muda preço de entrada, stop, alvo, hold nem custos.

## Regra congelada de circuit breaker

Breaker `dd800_cooldown180d`:

- usar apenas equity de trades aceitos e já fechados;
- estado inicial prospectivo em `2026-09-24T06:18:19-03:00`:
  - `closed_equity_pips=0`;
  - `closed_peak_pips=0`;
  - `pause_until=2026-09-24T06:18:19-03:00`;
  - nenhuma posição herdada;
- antes de cada nova entrada prospectiva, processar apenas fechamentos com
  `exit_stress_timestamp <= entry_timestamp`;
- se `closed_equity_pips - closed_peak_pips <= -800`, pausar novas entradas por
  `180` dias a partir do fechamento que disparou o breaker;
- fechamentos adicionais durante drawdown podem estender a pausa;
- trades já abertos antes do gatilho não são encerrados à força pelo breaker;
- o breaker só bloqueia novas entradas.

## Janela prospectiva

São elegíveis apenas trades aceitos com:

```text
entry_timestamp > 2026-09-24T06:18:19-03:00
```

Qualquer trade com `entry_timestamp <= 2026-09-24T06:18:19-03:00` deve ser
excluído da confirmação prospectiva.

Dados históricos antes da âncora podem ser usados apenas para reconstruir
indicadores causais. Eles não podem contar como validação prospectiva e não
inicializam lucro/prejuízo do breaker.

## Gate de falha precoce

A hipótese deve ser marcada como falha se qualquer condição ocorrer:

- depois de pelo menos `20` trades prospectivos aceitos, soma de
  `net_stress_pips` menor ou igual a `-250` pips;
- depois de pelo menos `20` trades prospectivos aceitos, max drawdown fechado
  pior que `-1000` pips;
- dois anos prospectivos consecutivos com média `STRESS` negativa;
- qualquer trade `LONG` entrar na observação;
- qualquer par fora de `AUDUSD`, `EURUSD`, `GBPUSD`, `NZDUSD` entrar na
  observação;
- qualquer runner aproximar, imputar ou substituir ATR, canal Donchian, custo,
  entry timestamp, stop, target, regra de exposição ou breaker;
- qualquer regra acima for alterada depois deste documento;
- qualquer dado com `entry_timestamp <= 2026-09-24T06:18:19-03:00` entrar como
  confirmação prospectiva.

## Gate mínimo para candidato shadow offline

Mesmo que pareça bom antes, não há promoção antes de:

- pelo menos `40` trades prospectivos aceitos;
- pelo menos `2` anos prospectivos distintos com observações aceitas;
- média de `net_stress_pips` maior que `+25` pips/trade;
- limite inferior unilateral de 95% para `net_stress_pips` maior que `0`;
- média de `net_base_pips` positiva;
- pelo menos `2/3` dos anos prospectivos com média `STRESS` positiva;
- pelo menos `3` dos `4` pares congelados com média `STRESS` positiva ou neutra
  (`>= 0`) no conjunto prospectivo;
- nenhuma regra alterada desde este pré-registro;
- `100%` das observações com `side=SHORT`;
- `100%` das observações dentro dos `4` pares congelados;
- max drawdown fechado prospectivo não pior que `-1000` pips;
- max concorrência prospectiva não maior que `4`;
- nenhum ano negativo consecutivo no conjunto prospectivo.

Passar esse gate não aprova Testnet. Passar esse gate só autoriza escrever um
novo relatório propondo shadow offline, ainda `research_only`.

## Gate para Testnet

Fast151 não tem gate automático de Testnet.

Testnet exigiria novo relatório, revisão de execução, análise de gaps, sizing,
risco por trade, risco por par, risco agregado da cesta, limite de correlação,
auditoria do executor, kill-switch, reconciliação de ledger e autorização
explícita do Igor.

## Safety boundary

Durante todo o Fast151:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`.

Se qualquer implementação futura tentar enviar ordem, chamar corretora ou
gravar ledger real com base neste pré-registro, isso é defeito de segurança.

## Fontes congeladas

- commit fonte Fast150:
  `2e1c13f3487bb2472cae93eb53e4f0167a025ef8`;
- Fast150 report JSON:
  `d3d3f98c60ffc74335ecaa8db8eeb79603dc9dc68574e14a998d362755f0019e`;
- Fast150 report markdown:
  `5abe3a0a649a9112f3917173c74221384ba98d3105634f79177ee8bb0ee55756`;
- Fast150 registry:
  `0ca54d414e1f2139fe0e3af5873776456b82465571fd9dd3929e3ad9b99dd60b`;
- Fast150 runner:
  `f3e7aa839add88e91c72fd0ac48b4727dd7c08a632e2f74be92e00f0e873ce52`;
- Fast150 test:
  `1a57e465b65fe208c8471844cde840c8b4b3e89a3048d03ab4f38fbda394b0ba`;
- Fast150 metrics:
  `77bb1d195e7dd27424d5b4bef9ea9a4922fbc4f760f93752aa2c56e55bd4aaab`;
- Fast150 selected pre trades:
  `e619dc3e28f50523a3b8440c15078ab4b97c0ffc4aaebc7eafef0e44c54598e5`;
- Fast150 selected current trades:
  `437af9f105fa677ae8caea6ed9088666f9fbc64a14d15459f61716659efd85a0`;
- Fast150 circuit breaker summary:
  `94a08c6f73ce7360f13b402eb71408a922b6d682666eb9a4aa40f6de4d2880c9`.

## O que NÃO será feito

- Chamar Fast151 de estratégia aprovada antes de dados prospectivos suficientes.
- Usar Fast150 histórico como aprovação para Testnet.
- Alterar `dd800_cooldown180d` depois de ver resultado novo.
- Alterar `max_per_pair_1` depois de ver resultado novo.
- Trocar `SHORT` por `LONG` ou `all_sides`.
- Trocar `fixed_1r` por trailing, stop-only ou outro target.
- Alterar canal `100`, hold `20`, stop `2 ATR` ou alvo `1R`.
- Remover par perdedor ou adicionar par novo.
- Reduzir custo de stress para fazer passar.
- Ligar Testnet, Demo ou LIVE com base neste pré-registro.

## Próxima implementação

O próximo passo técnico é criar um runner Fast151 que:

- valide o hash deste pré-registro e o evento do registro;
- reconstrua sinais e ATR de forma causal por par;
- aplique `max_per_pair_1` somente dentro da janela prospectiva;
- aplique `dd800_cooldown180d` com estado inicial zerado na âncora;
- descarte todo trade aceito com
  `entry_timestamp <= 2026-09-24T06:18:19-03:00`;
- descarte qualquer linha com `side != SHORT`;
- descarte qualquer par fora de `AUDUSD`, `EURUSD`, `GBPUSD`, `NZDUSD`;
- liquide offline em M1 com lógica conservadora de gaps;
- calcule os gates de falha e promoção;
- mantenha `orders_sent=false` sempre.
