# Pré-registro C9: as duas bancas com avaliadores novos, ajustadas só por walk-forward, julgadas na demo

Registrado em 2026-09-25, antes do código e de qualquer resultado.

## O que muda em relação a C6 e C8

**Avaliadores novos** (só barras diárias fechadas), somados aos de C6 (entrada) e C8 (saída):

- **Sessão** da entrada (Ásia, Londres, sobreposição Londres–Nova York, tarde de Nova York).
- **Força da tendência:** ADX(14) diário. **Duração:** dias seguidos com o fechamento do lado da operação em relação à média de 50.
- **Compressão de volatilidade:** percentil da largura das Bandas de Bollinger (20, 2) nos últimos 125 dias.
- **Contexto do dólar:** tendência do índice do dólar (UDXUSD: distância à média de 50 em ATR) e retorno de 5 dias em ATR, com o sinal da exposição da operação ao dólar; correlação de 60 dias dos retornos do mercado com o índice do dólar.
- **Sequência da própria estratégia:** vitórias ou derrotas seguidas nas últimas operações fechadas da mesma estratégia no mesmo mercado.
- **Distância ao stop e ao alvo** em ATR: na entrada, a distância inicial; na saída, a distância do preço atual.
- **Lucro em aberto já devolvido** (saída): já existia em C8 e continua.

## Ajuste do modelo (a única escolha, e sem olhar o último ano)

Grade fixa de 8 combinações do `XGBRegressor`: profundidade {3, 5} × árvores {200, 500} × peso mínimo por folha {50, 300}; o resto igual a C6. O mesmo conjunto de combinações vale para as duas bancas.

**Walk-forward:** para cada ano de validação de 2019 a 2025 (2025 só até agosto), o modelo treina com as operações fechadas antes do início do ano e decide as operações que entram naquele ano. A combinação escolhida é a de maior **média por operação das duas bancas juntas** (entrada filtra, saída realiza), somando os 7 anos de validação. A saída não se aplica ao TSMOM (em C8 ela piorou a estratégia que vive de carregar posição; decisão registrada antes deste teste).

O último ano (2025-09 a 2026-08) já julgou C6, C7 e C8 e **não participa da escolha**; o resultado nele entra no relatório só como informação.

## Juiz

O modelo escolhido é treinado de novo com todas as operações fechadas até 2026-08-31, congelado (hash no relatório) e colocado na demo 10139135. **A conclusão sai da demo, daqui para a frente.** Referência antes de rodar: se a média das duas bancas no walk-forward não for positiva, o relatório diz isso, e a demo é um teste sem expectativa de lucro.

Estratégias, mercados (os 37, sem ouro, prata, WTI e Brent), custos e execução: os de C3/C8.
