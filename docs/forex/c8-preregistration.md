# Pré-registro C8: a banca de saída (realizar agora ou segurar), operação por operação

Registrado em 2026-09-25, antes do código e de qualquer resultado.

## Pergunta

Um modelo que decide, a cada fechamento diário e para cada posição no lucro, entre **realizar agora** e **segurar até a saída original da estratégia**, melhora o resultado das estratégias que carregam posição?

## Operações

As de C3 (regras e parâmetros inalterados) das estratégias que carregam posição por dias: TSMOM, TURTLE20, TURTLE55, SQUEEZE_D1, NR7_D1, RSI2_D1, CHANDELIER_D1, CHANDELIER_H4. Os 37 instrumentos da banca (sem ouro, prata, WTI e Brent). Custos e execução de C3.

## Pontos de decisão

Cada fechamento diário (dia FX) depois da entrada e antes da saída original em que a posição está no lucro depois do custo. Avaliadores (só dado fechado): estratégia, lado, classe do mercado, lucro atual em ATR, melhor lucro já visto em ATR, quanto já devolveu do melhor, dias em posição, volatilidade (ATR14/ATR100 diário), distância às médias de 50 e 200 dias a favor do lado, inclinação da média de 50 a favor do lado, RSI(14) diário a favor do lado. Alvo do modelo: quanto a operação ainda ganha (ou perde) se segurar até a saída original, em ATR.

## Modelo e regra (fixos)

O mesmo `XGBRegressor` de C6 (mesmos parâmetros). **Realiza no primeiro ponto de decisão em que a previsão for < 0**; senão, segura até a saída original. Treino: operações fechadas antes de 2025-09-01. Teste, uma vez: operações com entrada de 2025-09-01 a 2026-08-31.

## Critério

Aprovada se, no teste: a média por operação com a banca de saída é > 0 no custo base e no estresse; a melhora média por operação em relação à saída original é > 0 com valor-p unilateral ≤ 0,05 (t por cluster de dia de entrada); ao menos 100 operações.

## Emenda antes de qualquer resultado (pedido do Igor): entrada e saída juntas

Além da banca de saída sozinha, o mesmo teste avalia **as duas bancas juntas**: nas operações de teste, só as que a banca de entrada do C6 (mesmo código, mesmo modelo, treinado com operações fechadas antes de 2025-09-01 nas 14 estratégias e 41 mercados) aceita, e nelas a banca de saída decide quando realizar. Critério da combinação: média > 0 no custo base e no estresse, valor-p unilateral do t por cluster de dia ≤ 0,05 e ao menos 100 operações.
