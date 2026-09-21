# Rodada 11 — fluxo executado centralizado

Data: 2026-09-21  
Mercado: BTCUSDT spot, Binance Vision, barras de 1 segundo  
Resultado: zero hipótese aprovada

## Dados e separação temporal

Os sete arquivos diários de 2026-09-13 a 2026-09-19 foram validados contra os checksums oficiais.
O teste usou quatro dias para treino, um para seleção e dois dias intocados para auditoria. A decisão
ocorre a cada cinco segundos e cada evento existe nos dois lados, sem reutilizar segundos ausentes.

| Horizonte | Treino | Seleção | Auditoria |
|---:|---:|---:|---:|
| 5 s | 138.042 | 34.530 | 69.088 |
| 30 s | 138.042 | 34.530 | 69.078 |
| 120 s | 138.042 | 34.530 | 69.042 |

## Cobertura

Foram geradas 1.945 hipóteses sobre retorno, volatilidade, range, volume negociado, negócios,
agressão taker-buy, acelerações de fluxo, posição no range e distância do VWAP.

| Horizonte | Caudas | Interseções | Árvores | Clusters | Modelos | Total |
|---:|---:|---:|---:|---:|---:|---:|
| 5 s | 156 | 417 | 26 | 28 | 30 | 657 |
| 30 s | 156 | 415 | 23 | 28 | 30 | 652 |
| 120 s | 156 | 404 | 18 | 28 | 30 | 636 |

As 25 melhores por horizonte chegaram à auditoria. A correção de Bonferroni foi aplicada às 75
finalistas em conjunto.

## Resultado fora da amostra

Nenhuma das 75 regras teve média positiva sequer no hurdle otimista de 5 bps. Consequentemente,
nenhuma sobreviveu ao hurdle severo de 20 bps ou ao limite inferior corrigido.

| Horizonte | Melhor média com 5 bps | Mesma regra com 20 bps | Eventos | Família |
|---:|---:|---:|---:|---|
| 5 s | -4,524 bps | -19,524 bps | 965 | HistGradient top 2% |
| 30 s | -3,529 bps | -18,529 bps | 389 | Extra Trees top 1% |
| 120 s | -3,305 bps | -18,305 bps | 235 | Interseção fluxo + retorno |

Isso significa que o melhor recorte encontrou apenas cerca de 0,5 a 1,7 bps brutos por operação,
abaixo até do custo otimista. O fluxo executado contém alguma ordenação fraca, mas não um edge
economicamente negociável nesse formato de saída terminal.

## Decisão

O piloto terminal está rejeitado. Isso não encerra a hipótese específica de Igor — entrada por
probabilidade com stop, breakeven e trailing — porque o target desta rodada foi o retorno no fim de
5, 30 ou 120 segundos. O próximo teste deve congelar as entradas e procurar a combinação de gestão
de caminho em uma amostra diária ainda não aberta. Só esse teste separa corretamente “a entrada
antecipa algum movimento” de “o preço termina acima após N segundos”. Nenhum resultado desta rodada
autoriza Demo ou LIVE.

Artefato auditado: `97bceae1678a5d4931218bdc1befa1540613d6a913b2e40f3f17f73cfab5bec9`.
