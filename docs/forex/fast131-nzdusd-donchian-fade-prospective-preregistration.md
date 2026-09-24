# Fast131 — pré-registro prospectivo NZDUSD Donchian fade

Registrado em `2026-09-24T02:50:09-03:00`, depois do scout Fast129 e do audit
de risco Fast130.

Fast131 não é estratégia aprovada. É uma observação prospectiva offline para
testar se a melhor pista do Fast130 continua funcionando em dados posteriores ao
pré-registro, sem trocar parâmetros depois de ver o resultado.

Fast131 não autoriza Demo, Testnet, LIVE, envio de ordem, commit de ledger ou
operação manual.

## Origem da hipótese

Fast129 testou fade/reversão de rompimentos Donchian em D1 em dez pares FX com
saída por tempo. Apenas um candidato sobreviveu ao audit:

- candidato: `NZDUSD:d1_donchian_fade_ch100_hold20`;
- seleção: `44` trades, média stress `+40.5638` pips/trade, anos positivos
  `4/5`;
- audit: `21` trades, média stress `+61.3837` pips/trade, LCB Bonferroni
  `+1.3387`, anos positivos `3/3`.

Fast130 congelou esse candidato e testou gestão de risco com hard stop,
take-profit e trailing. O melhor candidato por ranking de seleção/audit foi:

`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_2r`

Resultado Fast130:

- sinais de entrada: `79`;
- políticas de risco testadas: `12`;
- trades simulados: `780`;
- finalistas elegíveis na seleção: `10`;
- candidatos aprovados no audit: `9`;
- seleção do candidato preferencial: `44` trades, média stress `+40.9656`
  pips/trade, LCB stress `+2.8862`, anos positivos `4/5`;
- audit do candidato preferencial: `21` trades, média stress `+58.9756`
  pips/trade, LCB stress Bonferroni `+13.1163`, anos positivos `3/3`,
  `pilot_gate=true`.

Conclusão de partida: esta é a primeira família recente da pesquisa FX que
sobreviveu a seleção, audit independente, custo de stress e hard stop. Ainda
assim, ela foi descoberta por busca histórica e precisa de observação
prospectiva antes de qualquer decisão de execução.

## Safety boundary

Durante todo o Fast131:

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
- `risk_policy_id`: `stop2_fixed_2r`;
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
- alvo: `2R`;
- gestão: saída no primeiro evento entre stop, alvo ou expiração de `20` dias;
- gaps de fim de semana: respeitar o primeiro preço M1 disponível depois do gap
  e aplicar stop/target de forma conservadora quando o preço reabrir além do
  nível;
- custos: cenário base `FUSION_ZERO` e cenário `STRESS` do Fast130; o cenário
  primário de decisão é `STRESS`.

Nenhum filtro adicional de sessão, weekday, mês, volatilidade, macro, spread,
distância de rompimento, regime, correlação, trailing, top-p, par alternativo ou
calibração posterior está autorizado no Fast131.

## Janela prospectiva

São elegíveis apenas trades com:

```text
entry_timestamp > 2026-09-24T02:50:09-03:00
```

Qualquer trade com `entry_timestamp <= 2026-09-24T02:50:09-03:00` deve ser
excluído, mesmo que seja lucrativo.

Os arquivos históricos usados em Fast129/Fast130 podem ser usados apenas para
reconstruir estado causal até a âncora de pré-registro. Eles não podem contar
como confirmação prospectiva.

## Gates

### Gate de falha precoce

A hipótese deve ser encerrada como falha se qualquer condição ocorrer:

- depois de pelo menos `8` trades prospectivos, soma de `net_stress_pips`
  menor ou igual a `-100` pips;
- dois anos prospectivos consecutivos com média `STRESS` negativa;
- qualquer runner precisar aproximar, imputar ou substituir ATR, canal
  Donchian, custo, entry timestamp, stop ou target;
- qualquer regra acima for alterada depois deste documento;
- qualquer dado com `entry_timestamp <= 2026-09-24T02:50:09-03:00` entrar como
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

Fast131 não tem gate automático de Testnet.

Testnet exigiria novo relatório, revisão de execução, análise de gaps, tamanho
de posição, risco por trade, risco por par, circuito de drawdown e autorização
explícita do Igor.

## Fontes congeladas

- commit fonte Fast129/Fast130: `faa915dcd3ab929605c0781e87f0048e39a05f2c`;
- timestamp do commit fonte: `2026-09-24T02:49:59-03:00`;
- Fast129 report JSON:
  `d6ae2769b745be9d72ead8bd0f24aa6ddfde19092e2a31b919931a1a5c94051f`;
- Fast129 trades:
  `886158b88d7cf3c0131de9960258595a45838eb93b5fc94db10a97b0b6895157`;
- Fast129 selection:
  `273a451240fefa39301cbb701cb91ba95eab5428224951f134bfdf233e2fda3b`;
- Fast129 audit:
  `b9a41e40d80dd57dcc265ee071d3012ced96da81709a307999ec767c184cac14`;
- Fast130 report JSON:
  `b6c473330d22aaf8f1f2a01671ade72bb09a3664cb817169acd9fa7a919007e3`;
- Fast130 signals:
  `565aefb881e831b0551e9e7040e9c2e13e0ad7b3ca243b3cfcbe0c6efa6b248a`;
- Fast130 trades:
  `9b5049138cd4ac3fe27099a4dfe0a5890487ebd6521c5a530dcf7e7243438ce6`;
- Fast130 selection:
  `2ca0b42fea20428fa5ccd0b8c14ae1fd96b3aa11bb53abc2e14ee655a6962c0c`;
- Fast130 audit:
  `816ce331c515205d7ab2bf3af90103956f3e45ed1ea5cf4be06f04316d180b5b`;
- Fast130 pair quality:
  `2f1596c51b37aa93a5502b40ee6b9c16dfc24827c96b930162c6649897a14a5d`.

## O que NÃO será feito

- Chamar Fast131 de estratégia aprovada antes de dados prospectivos suficientes.
- Trocar NZDUSD por outro par depois de ver resultado novo.
- Alterar canal `100`, hold `20`, stop `2 ATR` ou alvo `2R`.
- Adicionar trailing porque pareceu melhor em dado novo.
- Reduzir custo de stress para fazer o resultado passar.
- Usar trade histórico como validação prospectiva.
- Ligar Testnet, Demo ou LIVE com base neste pré-registro.

## Próxima implementação

O próximo passo técnico é criar um runner Fast131 que:

- valide o hash deste pré-registro e o evento do registro;
- reconstrua sinais e ATR de forma causal;
- descarte todo trade com `entry_timestamp <= 2026-09-24T02:50:09-03:00`;
- use apenas a regra `NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_2r`;
- liquide offline em M1 com lógica conservadora de gaps;
- calcule os gates;
- mantenha `orders_sent=false` sempre.
