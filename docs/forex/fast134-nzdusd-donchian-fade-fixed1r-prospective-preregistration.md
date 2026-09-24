# Fast134 — pré-registro prospectivo NZDUSD Donchian fade fixed 1R

Registrado em `2026-09-24T04:04:40-03:00`, depois do audit global Fast133.

Fast134 não é estratégia aprovada. É uma observação prospectiva offline para
testar se o melhor candidato do Fast133 continua funcionando em dados
posteriores ao pré-registro, sem trocar parâmetros depois de ver o resultado.

Fast134 não autoriza Demo, Testnet, LIVE, envio de ordem, commit de ledger ou
operação manual.

## Origem da hipótese

Fast129 encontrou uma única família D1 Donchian fade promissora:

`NZDUSD:d1_donchian_fade_ch100_hold20`

Fast130 auditou essa família com stop/target/trailing e escolheu inicialmente
`stop2_fixed_2r`. Esse candidato foi congelado no Fast131 e continua válido como
observação separada.

Fast133 ampliou a busca: testou todos os pares, canais, holds e gestões de
risco da família D1 Donchian fade. Com Bonferroni sobre `100` finalistas, o
único candidato aprovado foi:

`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r`

Resultado Fast133 do candidato congelado:

- seleção: `44` trades, média stress `+34.5342` pips/trade, LCB stress
  `+1.3261`, anos positivos `4/5`;
- audit: `21` trades, média stress `+67.7507` pips/trade, LCB stress
  Bonferroni `+20.0959`, anos positivos `3/3`, `pilot_gate=true`.

Conclusão de partida: `fixed_1r` é a variante mais robusta no audit global. Ela
não substitui retroativamente o Fast131, mas é o melhor congelamento para nova
observação prospectiva.

## Safety boundary

Durante todo o Fast134:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`.

Se qualquer implementação futura tentar enviar ordem, chamar corretora ou
gravar ledger real com base neste pré-registro, isso é defeito de segurança.

## Hipótese congelada

Configuração de entrada:

- par: `NZDUSD`;
- timeframe de sinal: `D1`;
- dados de execução: M1 local;
- `entry_policy_id`: `d1_donchian_fade_ch100_hold20`;
- `risk_policy_id`: `stop2_fixed_1r`;
- canal Donchian: `100` dias fechados, calculado de forma causal;
- breakout long do canal: fechamento D1 acima da máxima causal dos `100` dias
  anteriores;
- breakout short do canal: fechamento D1 abaixo da mínima causal dos `100` dias
  anteriores;
- direção do trade: fade/contra o rompimento;
  - se fechar acima da máxima causal: abrir `SHORT`;
  - se fechar abaixo da mínima causal: abrir `LONG`;
- entrada: primeiro minuto M1 disponível posterior ao fechamento da barra D1 de
  sinal;
- hold máximo: `20` dias;
- base de risco: ATR(14) diário em pips, deslocado um D1 para trás;
- stop inicial: `2.0 * ATR(14)` contra a entrada;
- alvo: `1R`;
- gestão: saída no primeiro evento entre stop, alvo ou expiração de `20` dias;
- gaps de fim de semana: respeitar o primeiro preço M1 disponível depois do gap
  e aplicar stop/target de forma conservadora quando o preço reabrir além do
  nível;
- custos: cenário base `FUSION_ZERO` e cenário `STRESS` do Fast133; o cenário
  primário de decisão é `STRESS`.

Nenhum filtro adicional de sessão, weekday, mês, volatilidade, macro, spread,
distância de rompimento, regime, correlação, trailing, top-p, par alternativo ou
calibração posterior está autorizado no Fast134.

## Janela prospectiva

São elegíveis apenas trades com:

```text
entry_timestamp > 2026-09-24T04:04:40-03:00
```

Qualquer trade com `entry_timestamp <= 2026-09-24T04:04:40-03:00` deve ser
excluído, mesmo que seja lucrativo.

Os arquivos históricos usados em Fast129/Fast130/Fast133 podem ser usados apenas
para reconstruir estado causal até a âncora de pré-registro. Eles não podem
contar como confirmação prospectiva.

## Gates

### Gate de falha precoce

A hipótese deve ser encerrada como falha se qualquer condição ocorrer:

- depois de pelo menos `8` trades prospectivos, soma de `net_stress_pips`
  menor ou igual a `-100` pips;
- dois anos prospectivos consecutivos com média `STRESS` negativa;
- qualquer runner precisar aproximar, imputar ou substituir ATR, canal
  Donchian, custo, entry timestamp, stop ou target;
- qualquer regra acima for alterada depois deste documento;
- qualquer dado com `entry_timestamp <= 2026-09-24T04:04:40-03:00` entrar como
  confirmação prospectiva.

### Gate mínimo para candidato shadow offline

Mesmo que pareça bom antes, não há promoção antes de:

- pelo menos `20` trades prospectivos;
- pelo menos `2` anos prospectivos distintos com observações;
- média de `net_stress_pips` maior que `+20` pips/trade;
- limite inferior unilateral de 95% para `net_stress_pips` maior que `0`;
- média de `net_base_pips` positiva;
- pelo menos `2/3` dos anos prospectivos com média `STRESS` positiva;
- nenhuma regra alterada desde este pré-registro.

Passar esse gate não aprova Testnet. Passar esse gate só autoriza escrever um
novo relatório propondo shadow offline, ainda `research_only`.

### Gate para Testnet

Fast134 não tem gate automático de Testnet.

Testnet exigiria novo relatório, revisão de execução, análise de gaps, tamanho
de posição, risco por trade, risco por par, circuito de drawdown e autorização
explícita do Igor.

## Fontes congeladas

- commit fonte Fast133: `579a16ab72809dea03018d8d1c87195f1297efa8`;
- timestamp do commit fonte: `2026-09-24T04:04:33-03:00`;
- Fast133 report JSON:
  `9d482ba974a6061f26a8b1a1e6f5dc702d93094f7424496af88871e7f18ca6fd`;
- Fast133 report markdown:
  `618e9925d8b1f02c6454a586fc6246e1381f19d4285cc6bdc4ad75f85816624e`;
- Fast133 trades:
  `13bad134cd1da482eb69e4ad081daa255f5e0d2a685b94589a0fa1e5750d41da`;
- Fast133 signal counts:
  `59b55e287b175263504ed9060cd9c81e7562b075973aa44318986f3240ead9b5`;
- Fast133 selection:
  `dea15246a4f1d0a6c365cd7730ddfae133c981f592817e72ce4fbca90458f2b8`;
- Fast133 audit:
  `a53f99e81ecdce2eba6cc438b3fb293c75381e234a06244ece3f36e2544c5d2d`;
- Fast133 pair quality:
  `1dc11165fc3ea86d9e930dc51b14ec02df56075b62e7e5e973059c34582dd119`.

## O que NÃO será feito

- Chamar Fast134 de estratégia aprovada antes de dados prospectivos suficientes.
- Trocar `fixed_1r` por `fixed_2r`, trailing ou stop-only depois de ver dado
  novo.
- Trocar NZDUSD por outro par depois de ver resultado novo.
- Alterar canal `100`, hold `20`, stop `2 ATR` ou alvo `1R`.
- Reduzir custo de stress para fazer o resultado passar.
- Usar trade histórico como validação prospectiva.
- Ligar Testnet, Demo ou LIVE com base neste pré-registro.

## Próxima implementação

O próximo passo técnico é criar um runner Fast134 que:

- valide o hash deste pré-registro e o evento do registro;
- reconstrua sinais e ATR de forma causal;
- descarte todo trade com `entry_timestamp <= 2026-09-24T04:04:40-03:00`;
- use apenas a regra `NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r`;
- liquide offline em M1 com lógica conservadora de gaps;
- calcule os gates;
- mantenha `orders_sent=false` sempre.
