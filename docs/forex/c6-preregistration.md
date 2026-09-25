# Pré-registro C6: a "banca" (meta-labeling) em cima das 14 estratégias publicadas

Registrado em 2026-09-25, antes de existir o código da banca e antes de qualquer resultado dela.

## Ideia

As 14 estratégias de C3 (regras e parâmetros inalterados, `fx_c3_lab.py`) "levantam a mão". Um modelo rápido (gradient boosting, a banca) olha o contexto de cada sinal e decide **entrar ou pular**. É o meta-labeling de Marcos López de Prado: a estratégia dá a direção; a banca decide se aquele sinal, naquele contexto, vale o custo. A banca não muda entrada, stop, alvo nem saída das estratégias: só filtra.

## Divulgação

As médias por estratégia dos 10 pares (C3, 2012-2026) e de RSI2_D1 e TSMOM nos 31 mercados (C4) já foram vistas. Nenhum resultado condicional por contexto foi visto. As outras 12 estratégias nunca rodaram nos 31 mercados do grupo B.

## Dados e divisão no tempo

- Os 41 instrumentos de C3, manifesto `docs/forex/c3-data-manifest.csv`, custos e execução de C3.
- **Treino:** todas as operações das 14 estratégias nos 41 instrumentos com entrada até 2025-08-31.
- **Teste:** entradas de 2025-09-01 a 2026-08-31 (o último ano), uma única vez.
- **Prova principal:** o teste restrito ao grupo B (31 mercados) e às 12 estratégias que nunca rodaram nele (todas menos RSI2_D1 e TSMOM). O teste completo (41 mercados, 14 estratégias) é informativo.

## Os avaliadores da banca (fixos)

Calculados só com barras fechadas antes da entrada:

- **Estratégia e lado:** qual das 14, comprado ou vendido, classe do instrumento (major, cruzado, metal, índice, energia, índice do dólar).
- **Custo:** custo da operação em ATR da estratégia.
- **Volatilidade:** ATR(14) diário ÷ ATR(100) diário; ATR(14) de H1 ÷ ATR(100) de H1.
- **Tendência:** (fechamento − média de 50 dias) ÷ ATR diário; (fechamento − média de 200 dias) ÷ ATR diário; inclinação da média de 50 dias em 5 dias ÷ ATR; os três multiplicados pelo lado (a favor ou contra a tendência); RSI(14) diário.
- **Curto prazo:** retorno das últimas 4 horas ÷ ATR de H1, multiplicado pelo lado.
- **Horário e dia:** hora de Nova York da entrada, dia da semana.
- **Momento da própria estratégia:** média do resultado das últimas 10 operações já fechadas da mesma estratégia no mesmo instrumento; das últimas 50 da mesma estratégia em todos os instrumentos.

## Modelo e regra de decisão (fixos)

- `xgboost.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=50, random_state=20260925)`, alvo = resultado líquido em ATR no custo base.
- **Entra se a previsão for > 0.** Nada é ajustado olhando o teste.

## Critério (prova principal)

Aprovada se, nas operações que a banca aceita no teste principal: média > 0 no custo base e no estresse; valor-p unilateral do t por cluster de dia de entrada ≤ 0,05; ao menos 100 operações aceitas; média das aceitas maior que a média de todas as operações do teste principal (a banca precisa melhorar o resultado sem filtro).

Se aprovada: demo com o modelo congelado. Se não: paro e levo os números ao Igor.

## Palpite antes do dado

A banca costuma melhorar a média quando existe um pedaço lucrativo reconhecível; C3 e C4 sugerem pouco material bruto. Palpite: 15% a 25% de chance de passar.

## Leitura fixada no código antes do resultado

- O treino usa só operações **fechadas** antes de 2025-09-01 (mais conservador que "entrada até 2025-08": nenhum resultado usado no treino acontece dentro do ano de teste).
