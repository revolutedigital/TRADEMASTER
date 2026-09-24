# Fast143 — pré-registro prospectivo NZDUSD Donchian fade short-only

Registrado em `2026-09-24T05:25:14-03:00`, depois do audit cronológico
Fast142.

Fast143 não é estratégia aprovada. É uma observação prospectiva offline para
testar se a variante mais limpa do Fast142 continua funcionando em dados
posteriores ao pré-registro, sem trocar regra depois de ver o resultado.

Fast143 não autoriza Demo, Testnet, LIVE, envio de ordem, commit de ledger ou
operação manual.

## Origem da hipótese

Fast134 congelou a melhor candidata Fast133:

`NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r`

Fast135 mostrou que o candidato bruto tinha estabilidade temporal mista:

- `current_2019_2026`: média stress positiva e LCB positivo;
- `pre_2011_2018`: média stress positiva, mas LCB negativo.

Fast142 fez um audit cronológico de side-filter usando somente `pre_2011_2018`
para escolher entre `all_sides`, `long_only` e `short_only`, depois auditando em
`current_2019_2026`. O único filtro que passou foi:

`short_only`

Resultado Fast142:

- seleção pré-2019:
  - `19` trades;
  - média stress `+98.9480` pips/trade;
  - LCB stress `+64.0934`;
  - anos positivos `5/5`;
- audit 2019–2026:
  - `29` trades;
  - média stress `+53.9022` pips/trade;
  - LCB stress `+25.2283`;
  - anos positivos `6/7`.

Conclusão de partida: `short_only` é a variante histórica mais limpa da linha
NZDUSD Donchian fade. Ainda assim, ela nasceu de pesquisa histórica e precisa de
observação prospectiva nova antes de qualquer shadow/Testnet.

## Safety boundary

Durante todo o Fast143:

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
- `side_filter`: `SHORT`;
- canal Donchian: `100` dias fechados, calculado de forma causal;
- trade elegível: somente fechamento D1 acima da máxima causal dos `100` dias
  anteriores;
- direção do trade: fade vendido contra rompimento superior;
- trade não elegível: rompimento inferior que geraria `LONG`;
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
- custos: cenário base `FUSION_ZERO` e cenário `STRESS`; o cenário primário de
  decisão é `STRESS`.

Nenhum filtro adicional de sessão, weekday, mês, volatilidade, macro, spread,
distância de rompimento, regime, correlação, trailing, top-p, par alternativo ou
calibração posterior está autorizado no Fast143.

## Janela prospectiva

São elegíveis apenas trades com:

```text
entry_timestamp > 2026-09-24T05:25:14-03:00
```

Qualquer trade com `entry_timestamp <= 2026-09-24T05:25:14-03:00` deve ser
excluído, mesmo que seja lucrativo.

Os arquivos históricos usados em Fast133/Fast135/Fast142 podem ser usados apenas
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
- qualquer dado com `entry_timestamp <= 2026-09-24T05:25:14-03:00` entrar como
  confirmação prospectiva;
- qualquer trade `LONG` entrar na observação.

### Gate mínimo para candidato shadow offline

Mesmo que pareça bom antes, não há promoção antes de:

- pelo menos `12` trades prospectivos;
- pelo menos `2` anos prospectivos distintos com observações;
- média de `net_stress_pips` maior que `+30` pips/trade;
- limite inferior unilateral de 95% para `net_stress_pips` maior que `0`;
- média de `net_base_pips` positiva;
- pelo menos `2/3` dos anos prospectivos com média `STRESS` positiva;
- nenhuma regra alterada desde este pré-registro;
- `100%` das observações com `side=SHORT`.

Passar esse gate não aprova Testnet. Passar esse gate só autoriza escrever um
novo relatório propondo shadow offline, ainda `research_only`.

### Gate para Testnet

Fast143 não tem gate automático de Testnet.

Testnet exigiria novo relatório, revisão de execução, análise de gaps, tamanho
de posição, risco por trade, risco por par, circuito de drawdown e autorização
explícita do Igor.

## Fontes congeladas

- commit fonte Fast142:
  `82156f51f431bd6e7a4944b3d70c15b776fe1e91`;
- timestamp do commit fonte: `2026-09-24T05:24:37-03:00`;
- Fast142 report JSON:
  `b0f0cb45b717ec22d4caee6ed570d7cf0e110c222002800ad8a50abeb5d714f6`;
- Fast142 report markdown:
  `c44a8175e694a28ec5ccfc2020248d475c04fc54c1a91acde1218ae247d8aa5c`;
- Fast142 registry:
  `dab947d61da9541cd58fb41fb62f719e036f9eaa1c2cbffcf6396135ac450106`;
- Fast142 side-filter evaluations:
  `41d353edd491f6eee93c6e6c989ac7f8f6dc6dd21c447a3ca65bf1720f6508f8`;
- Fast142 passed filters:
  `c07c1fc17ee7e9ea7d402966bd18a304d1e54b51efe540df2d4b0e1f00bd2d87`;
- Fast135 temporal audit trades:
  `a4de5c3a9cd0502c814cf423aee65239deb2fb01aae0654b1ae58f47fe5f63a6`.

## O que NÃO será feito

- Chamar Fast143 de estratégia aprovada antes de dados prospectivos suficientes.
- Operar `LONG`.
- Trocar `short_only` por `all_sides` depois de ver dado novo.
- Trocar `fixed_1r` por `fixed_2r`, trailing ou stop-only depois de ver dado
  novo.
- Trocar NZDUSD por outro par depois de ver resultado novo.
- Alterar canal `100`, hold `20`, stop `2 ATR` ou alvo `1R`.
- Reduzir custo de stress para fazer o resultado passar.
- Usar trade histórico como validação prospectiva.
- Ligar Testnet, Demo ou LIVE com base neste pré-registro.

## Próxima implementação

O próximo passo técnico é criar um runner Fast143 que:

- valide o hash deste pré-registro e o evento do registro;
- reconstrua sinais e ATR de forma causal;
- descarte todo trade com `entry_timestamp <= 2026-09-24T05:25:14-03:00`;
- use apenas a regra
  `NZDUSD:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only`;
- descarte qualquer linha com `side != SHORT`;
- liquide offline em M1 com lógica conservadora de gaps;
- calcule os gates;
- mantenha `orders_sent=false` sempre.
