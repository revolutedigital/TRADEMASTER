# Fast176 — pré-registro prospectivo do gate diagnóstico Fast175

Registrado em `2026-09-24T08:53:07-03:00`, depois do Fast174 e Fast175.

Fast176 não é estratégia aprovada. É uma observação prospectiva offline de uma
pista diagnóstica que apareceu depois de olhar a extensão histórica do Fast175:

`ETHUSDT_qv_30m >= 586337978.3184608`

Justamente por ser pós-hoc, ela não pode ser promovida pelo histórico. Ela só
pode ser congelada e medida em dados novos posteriores a este pré-registro.

Fast176 não autoriza Demo, Testnet, LIVE, envio de ordem, commit de ledger ou
operação manual.

## Origem da hipótese

Fast174 testou lead-lag cross-asset usando BTCUSDT, ETHUSDT e SOLUSDT para
prever ETHUSDT/SOLUSDT.

O clue que passou treino e validação foi:

- target: `ETHUSDT`;
- horizonte: `900s`;
- top fraction histórica: `0.02`;
- treino: `+893.4026` bps;
- validação: `+29.1748` bps;
- extensão: `-272.3073` bps.

Fast175 testou gates simples de regime em cima desse clue. O gate escolhido
honestamente só pelo treino falhou. Mas o melhor gate diagnóstico pós-hoc foi:

- gate: `ETHUSDT_qv_30m >= 586337978.3184608`;
- treino: `+508.9846` bps;
- validação: `+66.2637` bps;
- extensão: `+30.6811` bps.

Interpretação:

- isso não aprova estratégia;
- isso sugere que lead-lag melhora em regime de alta liquidez/quote volume de
  ETHUSDT;
- o único uso honesto agora é observação prospectiva congelada.

## Hipótese congelada

Mercado/símbolos:

- `BTCUSDT` USD-M perpetual public aggTrades;
- `ETHUSDT` USD-M perpetual public aggTrades;
- `SOLUSDT` USD-M perpetual public aggTrades.

Target negociado:

- `ETHUSDT`.

Horizonte:

- `900` segundos.

Decisão:

- frequência de decisão: `5` minutos;
- usar apenas dados fechados até o minuto da decisão;
- nenhum dado futuro pode entrar em feature.

Features congeladas:

- retornos fechados de `1`, `5`, `15`, `30` minutos;
- volatilidade realizada de `5`, `15`, `30` minutos;
- taker flow médio de `1`, `5`, `15`, `30` minutos;
- quote volume de `1`, `5`, `15`, `30` minutos;
- `hour_sin` e `hour_cos`;
- features calculadas para BTCUSDT, ETHUSDT e SOLUSDT.

Modelo congelado:

- `Ridge(alpha=10.0)`;
- `SimpleImputer(strategy="constant", fill_value=0.0)`;
- `StandardScaler`;
- treinar somente nos blocos históricos:
  - `fast50`: `2026-08-30`, `2026-08-31`;
  - `fast51`: `2026-09-06`, `2026-09-07`;
  - `fast52`: `2026-09-13`, `2026-09-14`.

Score/entrada congelados:

- `prediction = model.predict(features)`;
- direção: `LONG` se `prediction > 0`, `SHORT` se `prediction < 0`;
- `prediction == 0` não gera decisão;
- edge mínimo congelado, derivado do top `2%` do treino Fast174:

```text
abs(prediction) >= 17.039926929459345
```

Gate de regime congelado:

```text
ETHUSDT_qv_30m >= 586337978.3184608
```

Custos:

- custo explícito offline: `2.0` bps;
- net observado: `actual_bps * side - 2.0`.

Não há trailing, stop dinâmico, uma-posição, retreinamento, recalibração de
threshold, troca de target, troca de símbolo, troca de horizonte, troca de
feature, nem filtro adicional neste pré-registro.

## Janela prospectiva

São elegíveis apenas decisões com:

```text
decision_time > 2026-09-24T08:53:07-03:00
```

Qualquer decisão com `decision_time <= 2026-09-24T08:53:07-03:00` deve ser
excluída da confirmação prospectiva.

Dados anteriores à âncora podem ser usados apenas para treinar o modelo
congelado e calcular features causais. Eles não podem contar como observação
Fast176.

## Gate de falha precoce

A hipótese deve ser marcada como falha se qualquer condição ocorrer:

- após pelo menos `20` decisões prospectivas, `net_total_bps <= -100`;
- qualquer dia prospectivo com pelo menos `5` decisões fechar com
  `net_total_bps <= -60`;
- `ETHUSDT_qv_30m` precisar ser aproximado, imputado ou substituído;
- qualquer decisão anterior ou igual à âncora entrar como confirmação;
- qualquer regra acima for alterada depois deste pré-registro;
- qualquer rotina enviar ordem, tocar ledger ou simular autorização.

## Gate mínimo para candidato shadow offline

Mesmo que pareça bom antes, não há promoção antes de:

- pelo menos `60` decisões prospectivas;
- pelo menos `5` dias prospectivos distintos;
- média prospectiva `> +3.0` bps/decisão;
- net total prospectivo positivo em custo `2.0` bps;
- pelo menos `60%` dos dias suficientes com net total positivo;
- nenhum dia suficiente com `net_total_bps <= -60`;
- lower confidence bound unilateral de 95% do net por decisão maior que `0`;
- nenhuma regra alterada desde este pré-registro.

Passar esse gate não aprova Testnet. Passar esse gate só autoriza escrever um
novo relatório propondo shadow offline, ainda `research_only`.

## Gate para Testnet

Fast176 não tem gate automático de Testnet.

Testnet exigiria novo relatório, revisão de execução, slippage/fill/book,
one-position/portfolio risk, kill-switch, reconciliação de ledger e autorização
explícita do Igor.

## Safety boundary

Durante todo o Fast176:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`.

Nenhuma rotina Fast176 pode enviar ordens, tocar credenciais, escrever ledger de
execução, ativar estratégia ou simular autorização do Igor.

## Próximo passo permitido

O próximo passo permitido é criar/rodar um observador offline Fast176 que:

1. valide o hash deste pré-registro;
2. treine somente nos blocos congelados;
3. pontue apenas decisões disponíveis em dados públicos locais;
4. filtre apenas decisões prospectivas posteriores à âncora;
5. aplique edge mínimo e gate de regime congelados;
6. escreva evidência offline;
7. mantenha `approved_strategy=false` e `orders_sent=false`.
