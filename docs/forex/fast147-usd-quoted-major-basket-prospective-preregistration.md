# Fast147 — pré-registro prospectivo USD-quoted major basket

Registrado em `2026-09-24T05:49:20-03:00`, depois do audit cronológico
Fast146.

Fast147 não é estratégia aprovada. É uma observação prospectiva offline para
testar se a cesta USD-quoted major continua funcionando em dados posteriores ao
pré-registro, sem trocar regra depois de ver o resultado.

Fast147 não autoriza Demo, Testnet, LIVE, envio de ordem, commit de ledger ou
operação manual.

## Origem da hipótese

Fast145 mostrou que a regra exata de `NZDUSD` não replica par-a-par em outros
ativos, embora `AUDUSD` tenha aparecido como lead current-only.

Fast146 testou grupos causais pré-definidos usando seleção no pré-2019 e audit
no 2019–2026. O único grupo que passou os dois períodos foi:

`usd_quoted_majors`

Pares congelados:

- `AUDUSD`;
- `EURUSD`;
- `GBPUSD`;
- `NZDUSD`.

Política congelada:

`basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only`

Resultado Fast146:

- seleção pré-2019:
  - `89` trades;
  - `4` pares;
  - média stress `+37.8272` pips/trade;
  - total stress `+3366.6223` pips;
  - LCB stress `+9.7967`;
  - anos positivos `6/8`;
- audit 2019–2026:
  - `143` trades;
  - `4` pares;
  - média stress `+42.3340` pips/trade;
  - total stress `+6053.7689` pips;
  - LCB stress `+24.6986`;
  - anos positivos `7/8`.

Conclusão de partida: a cesta USD-quoted major é a candidata histórica mais
forte da linha Donchian fade short-only. Ainda assim, ela nasceu de pesquisa
histórica e precisa de observação prospectiva nova antes de qualquer
shadow/Testnet.

## Safety boundary

Durante todo o Fast147:

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

- grupo: `usd_quoted_majors`;
- pares elegíveis: `AUDUSD`, `EURUSD`, `GBPUSD`, `NZDUSD`;
- timeframe de sinal: `D1`;
- dados de execução: M1 local;
- `entry_policy_id`: `d1_donchian_fade_ch100_hold20`;
- `risk_policy_id`: `stop2_fixed_1r`;
- `side_filter`: `SHORT`;
- canal Donchian: `100` dias fechados, calculado de forma causal por par;
- trade elegível: somente fechamento D1 acima da máxima causal dos `100` dias
  anteriores do mesmo par;
- direção do trade: fade vendido contra rompimento superior;
- trade não elegível: rompimento inferior que geraria `LONG`;
- entrada: primeiro minuto M1 disponível posterior ao fechamento da barra D1 de
  sinal;
- hold máximo: `20` dias;
- base de risco: ATR(14) diário em pips, deslocado um D1 para trás, por par;
- stop inicial: `2.0 * ATR(14)` contra a entrada;
- alvo: `1R`;
- gestão: saída no primeiro evento entre stop, alvo ou expiração de `20` dias;
- gaps de fim de semana: respeitar o primeiro preço M1 disponível depois do gap
  e aplicar stop/target de forma conservadora quando o preço reabrir além do
  nível;
- custos: cenário base `FUSION_ZERO` e cenário `STRESS`; o cenário primário de
  decisão é `STRESS`.

Nenhum filtro adicional de sessão, weekday, mês, volatilidade, macro, spread,
distância de rompimento, regime, correlação, trailing, top-p, par alternativo,
remoção de par perdedor ou calibração posterior está autorizado no Fast147.

## Janela prospectiva

São elegíveis apenas trades com:

```text
entry_timestamp > 2026-09-24T05:49:20-03:00
```

Qualquer trade com `entry_timestamp <= 2026-09-24T05:49:20-03:00` deve ser
excluído, mesmo que seja lucrativo.

Os arquivos históricos usados em Fast133/Fast137/Fast146 podem ser usados apenas
para reconstruir estado causal até a âncora de pré-registro. Eles não podem
contar como confirmação prospectiva.

## Gates

### Gate de falha precoce

A hipótese deve ser encerrada como falha se qualquer condição ocorrer:

- depois de pelo menos `20` trades prospectivos, soma de `net_stress_pips`
  menor ou igual a `-250` pips;
- dois anos prospectivos consecutivos com média `STRESS` negativa;
- qualquer runner precisar aproximar, imputar ou substituir ATR, canal
  Donchian, custo, entry timestamp, stop ou target;
- qualquer regra acima for alterada depois deste documento;
- qualquer dado com `entry_timestamp <= 2026-09-24T05:49:20-03:00` entrar como
  confirmação prospectiva;
- qualquer trade `LONG` entrar na observação;
- qualquer par fora de `AUDUSD`, `EURUSD`, `GBPUSD`, `NZDUSD` entrar na
  observação.

### Gate mínimo para candidato shadow offline

Mesmo que pareça bom antes, não há promoção antes de:

- pelo menos `40` trades prospectivos;
- pelo menos `2` anos prospectivos distintos com observações;
- média de `net_stress_pips` maior que `+25` pips/trade;
- limite inferior unilateral de 95% para `net_stress_pips` maior que `0`;
- média de `net_base_pips` positiva;
- pelo menos `2/3` dos anos prospectivos com média `STRESS` positiva;
- pelo menos `3` dos `4` pares congelados com média `STRESS` positiva ou neutra
  (`>= 0`) no conjunto prospectivo;
- nenhuma regra alterada desde este pré-registro;
- `100%` das observações com `side=SHORT`;
- `100%` das observações dentro dos `4` pares congelados.

Passar esse gate não aprova Testnet. Passar esse gate só autoriza escrever um
novo relatório propondo shadow offline, ainda `research_only`.

### Gate para Testnet

Fast147 não tem gate automático de Testnet.

Testnet exigiria novo relatório, revisão de execução, análise de gaps, tamanho
de posição, risco por trade, risco por par, risco agregado da cesta, limite de
correlação, circuito de drawdown e autorização explícita do Igor.

## Fontes congeladas

- commit fonte Fast146:
  `3297e8b862f998211d943ea4becfc1cce8627718`;
- timestamp do commit fonte:
  `2026-09-24T05:49:06-03:00`;
- Fast146 report JSON:
  `450971147b555a83151ec99bf9e5d50f26d99c68868a3257ff79170468cad6a2`;
- Fast146 report markdown:
  `5b09cdfb7a228c3d8167ee814f8e7a1acebb10f1ed13b4dddf8bacb0030b467d`;
- Fast146 registry:
  `673ca60a32eb955a8bdc2583818b50311558728264e0daabc77e256e4bff685c`;
- Fast146 runner:
  `054c20a0d114da453f72f4f0f8fa101bfa37a44a86e3451683455a6983a21909`;
- Fast146 test:
  `5a70c6c4b87c4b8aa3fc6675a7b7e6ac3f0a794d2739f0bee0e8ce09f836be00`;
- Fast146 metrics:
  `e01e67c7758707592ac49de8926e0c47c2df1eb254dae063f733b8b578e0870b`;
- Fast146 passed groups:
  `8b5718a3165c36d0c71edf11c7a558a0b3f0166ec87d10b5e0575b9b9cf3ecde`;
- Fast146 group summary:
  `6138a80846af96418c9f59616366528945d2c157c6a8e8c49c82c052e14e367f`;
- Fast146 evaluations:
  `086509c1d61b6826328ebf094990346a379074b51c60cd42c2f58660f762a465`;
- Fast137 pre/current materialized trades:
  `2ea0b3b650035529b0601f3f2d0559e25a1a1554a2da2bef98f28ac80b423bdb`;
- Fast133 current full-grid trades:
  `13bad134cd1da482eb69e4ad081daa255f5e0d2a685b94589a0fa1e5750d41da`.

## O que NÃO será feito

- Chamar Fast147 de estratégia aprovada antes de dados prospectivos suficientes.
- Operar `LONG`.
- Trocar `short_only` por `all_sides` depois de ver dado novo.
- Trocar `fixed_1r` por `fixed_2r`, trailing ou stop-only depois de ver dado
  novo.
- Remover `EURUSD` por ter sido negativo no pré-2019.
- Adicionar `USDCAD`, `USDCHF`, `USDJPY`, `EURGBP`, `EURJPY` ou `GBPJPY` depois
  de ver resultado novo.
- Alterar canal `100`, hold `20`, stop `2 ATR` ou alvo `1R`.
- Reduzir custo de stress para fazer o resultado passar.
- Usar trade histórico como validação prospectiva.
- Ligar Testnet, Demo ou LIVE com base neste pré-registro.

## Próxima implementação

O próximo passo técnico é criar um runner Fast147 que:

- valide o hash deste pré-registro e o evento do registro;
- reconstrua sinais e ATR de forma causal por par;
- descarte todo trade com `entry_timestamp <= 2026-09-24T05:49:20-03:00`;
- use apenas a regra
  `basket:usd_quoted_majors:d1_donchian_fade_ch100_hold20:stop2_fixed_1r:short_only`;
- descarte qualquer linha com `side != SHORT`;
- descarte qualquer par fora de `AUDUSD`, `EURUSD`, `GBPUSD`, `NZDUSD`;
- liquide offline em M1 com lógica conservadora de gaps;
- calcule os gates de falha e promoção;
- mantenha `orders_sent=false` sempre.
