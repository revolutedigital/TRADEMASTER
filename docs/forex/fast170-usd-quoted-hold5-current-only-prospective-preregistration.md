# Fast170 — pré-registro prospectivo USD-quoted `hold5` current-only

Registrado em `2026-09-24T08:08:53-03:00`, depois do Fast168 e Fast169.

Fast170 não é estratégia aprovada. É uma observação prospectiva offline de uma
hipótese que ficou forte apenas no período current 2019-2026 e falhou no bloco
pré-2019. Justamente por isso, ela precisa ser congelada antes de qualquer
evidência futura.

Fast170 não autoriza Demo, Testnet, LIVE, envio de ordem, commit de ledger ou
operação manual.

## Identidade da política congelada

Policy ID operacional:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold5:stop2_fixed_1r:short_only:max_per_pair_1:dd800_cooldown180d`

Componentes:

- cesta: `usd_quoted_majors`;
- pares: `AUDUSD`, `EURUSD`, `GBPUSD`, `NZDUSD`;
- timeframe de sinal: `D1`;
- dados de execução: M1 local;
- entrada: `d1_donchian_fade_ch100_hold5`;
- risco/saída: `stop2_fixed_1r`;
- lado: apenas `SHORT`;
- overlay de exposição: `max_per_pair_1`;
- circuit breaker: `dd800_cooldown180d`;
- cenário primário de decisão: `STRESS`.

## Origem da hipótese

Fast168 mostrou que `hold5` parece muito melhor no período current, mas falha
no pré-2019:

- pré LCB: `-8.5652`;
- pré positive-year share: `0.625`;
- pré dois anos negativos consecutivos: `true`;
- falhas: `lcb_non_positive`,
  `two_consecutive_negative_years`,
  `positive_year_share_below_minimum`;
- current LCB: `+19.1220`;
- current mean: `+35.2947`;
- current swap break-even: `5.6610` pips por rollover/dia;
- current drawdown: `-238.8375`.

Fast169 auditou 135 combinações de basket/canal/hold na família USD-quoted.
Apenas Fast151 (`ch100_hold20`) passou pré e current. Mesmo assim, o melhor
diagnóstico current-only foi esta política `ch100_hold5`.

Interpretação:

- `hold5` não pode substituir Fast151;
- `hold5` não pode ser chamado de estratégia;
- `hold5` só pode ser observado prospectivamente como uma nova hipótese
  congelada.

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
- Saída: primeiro evento entre stop, alvo ou expiração de `5` dias.
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
- estado inicial prospectivo em `2026-09-24T08:08:53-03:00`:
  - `closed_equity_pips=0`;
  - `closed_peak_pips=0`;
  - `pause_until=2026-09-24T08:08:53-03:00`;
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
entry_timestamp > 2026-09-24T08:08:53-03:00
```

Qualquer trade com `entry_timestamp <= 2026-09-24T08:08:53-03:00` deve ser
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
- qualquer dado com `entry_timestamp <= 2026-09-24T08:08:53-03:00` entrar como
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

Fast170 não tem gate automático de Testnet.

Testnet exigiria novo relatório, revisão de execução, análise de gaps, sizing,
risco por trade, risco por par, risco agregado da cesta, limite de correlação,
auditoria do executor, kill-switch, reconciliação de ledger e autorização
explícita do Igor.

## Safety boundary

Durante todo o Fast170:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`.

Nenhuma rotina Fast170 pode enviar ordens, tocar credenciais, escrever ledger de
execução, ativar estratégia ou simular autorização do Igor.

## Próximo passo permitido

O próximo passo permitido é criar/rodar um observador offline Fast170 que:

1. valide o hash deste pré-registro;
2. leia apenas M1 local/independente;
3. reconstrua a regra congelada;
4. filtre apenas entradas prospectivas posteriores à âncora;
5. aplique `max_per_pair_1` e `dd800_cooldown180d`;
6. escreva evidência offline;
7. mantenha `approved_strategy=false` e `orders_sent=false`.
