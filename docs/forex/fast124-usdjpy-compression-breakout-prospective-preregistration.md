# Fast124 — pré-registro prospectivo USDJPY H1 compression breakout

Fast124 congela a melhor pista encontrada no Fast121/Fast122 e a transforma em
observação prospectiva offline. O objetivo é descobrir se o edge continua
existindo em dados novos, sem alterar parâmetros depois de ver o resultado.

Este artefato não aprova estratégia, não autoriza Testnet, não autoriza Demo e
não autoriza LIVE.

## Regra congelada

- par: `USDJPY`;
- timeframe de sinal: `1h`;
- `policy_id`: `tf1h_ch20_cq0.4_atr1.5_hold720_fixed_3r`;
- canal causal: 20 barras fechadas;
- compressão: média de true range de 20 barras abaixo ou igual ao quantil
  histórico causal `0.40` calculado em janela móvel de 500 barras, com mínimo
  de 250 barras;
- breakout long: fechamento H1 acima da máxima causal do canal;
- breakout short: fechamento H1 abaixo da mínima causal do canal;
- entrada: minuto M1 imediatamente posterior ao fechamento da barra de sinal;
- stop inicial: `1.5 * ATR(14)` em pips, calculado somente com barras já
  fechadas antes do sinal;
- alvo/saída: fixo em `3R`, sem trailing;
- expiração: `720` minutos;
- execução simulada: M1 bid/ask local;
- custos: cenário base `FUSION_ZERO` e cenário `STRESS`, com spread ampliado,
  comissão e slippage conforme runners Fast121/Fast122.

Nenhum filtro adicional de sessão, direção, weekday, mês, ATR, compressão,
distância de breakout, regime ou volatilidade está autorizado no Fast124.

## Âncora prospectiva

Somente trades com `entry_timestamp` estritamente maior que o timestamp do
commit de pré-registro Fast124 são elegíveis.

O runner Fast124 deve registrar explicitamente:

- `preregistration_commit`;
- `preregistration_timestamp`;
- `preregistration_sha256`;
- hash dos relatórios fonte Fast121, Fast122 e Fast123;
- hash dos arquivos M1 usados;
- quantidade de trades materializados;
- quantidade de observações elegíveis;
- flags de segurança.

Qualquer trade com `entry_timestamp <= preregistration_timestamp` deve ser
excluído, mesmo que seja lucrativo.

## Gates prospectivos

Fast124 é observação offline. Mesmo se os gates abaixo passarem, o resultado é
apenas candidato para revisão humana posterior.

Critério mínimo para status de candidato shadow:

- pelo menos `100` trades prospectivos no cenário `STRESS`;
- média de `net_stress_pips` maior que `1.0` pip/trade;
- limite inferior unilateral de 95% para `net_stress_pips` maior que `0`;
- média de `net_base_pips` positiva;
- pelo menos `2` anos prospectivos com observações;
- participação de anos positivos em `STRESS` maior ou igual a `2/3`.

Critério de falha antecipada:

- com pelo menos `50` trades prospectivos, média de `net_stress_pips` menor ou
  igual a `-2.0` pips/trade; ou
- dois anos prospectivos consecutivos com média `STRESS` negativa.

## Segurança

Durante todo o Fast124:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`.

O Fast124 não pode chamar trading engine, corretora, ledger, Demo, Testnet ou
LIVE. Se o runner encontrar um candidato shadow, o próximo passo obrigatório é
um novo relatório de revisão, não execução.

## Fontes históricas

- Fast121 selecionou a pista em scout histórico H1/H4, mas falhou audit formal:
  `FAST121_FX_COMPRESSION_BREAKOUT_FAILED_AUDIT`;
- Fast122 mostrou resultado recente positivo, mas estabilidade temporal fraca:
  `FAST122_USDJPY_COMPRESSION_BREAKOUT_WEAK_TEMPORAL_STABILITY_REJECTED`;
- Fast123 tentou filtros de regime simples e não encontrou filtro robusto:
  `FAST123_USDJPY_COMPRESSION_REGIME_FILTER_FAILED`.

Fast124 existe porque a pista é boa o suficiente para observação prospectiva,
mas não boa o suficiente para operação.
