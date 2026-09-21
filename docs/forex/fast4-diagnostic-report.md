# Rodada 4 — microestrutura de cotações subminuto

Data: 2026-09-21. Estado: **desenvolvimento rejeitado; nenhum modelo travado; 2022, D4, S3 e
semana da corretora fechados**.

## O que foi testado

- 7.394.199 eventos causais, em 10 pares, materializados de 2019 a 2022 a partir de ticks bid/ask.
- 51 colunas de features e 45 colunas de outcomes por evento, com decisões a cada 128 atualizações
  depois de um aquecimento de 512 atualizações.
- Entradas no primeiro tick posterior, compra no ask e venda no bid, com saídas em 30, 120 e 600
  segundos. Todos os resultados pagam spread observado, comissão e slippage; o cenário de estresse
  dobra o spread e aumenta o slippage.
- B0 sem operação, B1 por média histórica de par, lado, sessão e quintil de intensidade, uma ridge e
  oito XGBoost rasos. Cada modelo treinável foi avaliado nos quatro cortes pré-registrados, totalizando
  108 combinações modelo × threshold.
- Treino até 2020, early stopping em 2021-H1 e seleção integral em 2021-H2. O treino dos modelos
  treináveis usou uma amostra determinística de 1/8 dos eventos; calibração e seleção não foram
  amostradas.

O feed local permite medir microestrutura da **cotação top-of-book**, não volume negociado, agressão,
absorção ou profundidade do book. O campo de volume é zero e não entrou no modelo.

## Controles

B0 produz zero operação por definição. B1 também não encontrou nenhuma entrada no corte mínimo de
0,03R. O melhor valor esperado conservador de qualquer grupo histórico continuou negativo:

| horizonte | maior score B1 |
|---:|---:|
| 30s | -0,433R |
| 120s | -0,416R |
| 600s | -0,312R |

Isso confirma que simplesmente condicionar o retorno por par, lado, sessão e intensidade de updates
não paga o custo de execução.

## Grid treinável

| horizonte | linhas treino | calibração | seleção | tentativas | maior score conservador | operações emitidas |
|---:|---:|---:|---:|---:|---:|---:|
| 30s | 829.765 | 1.277.864 | 1.219.542 | 36 | -0,266R | 0 |
| 120s | 825.563 | 1.269.288 | 1.211.504 | 36 | -0,266R | 0 |
| 600s | 819.762 | 1.256.296 | 1.197.566 | 36 | -0,049R | 0 |

Os oito XGBoost escolheram `best_iteration = 0` tanto para custo-base quanto para estresse em todos
os horizontes: na calibração, acrescentar árvores piorou o erro imediatamente. A ridge foi o modelo
com o maior score máximo nos três horizontes, mas nem ela atingiu o corte mínimo de +0,03R.

Uma execução diagnóstica anterior, ainda sem o early stopping exigido pelo pré-registro, chegou a
emitir quatro operações no melhor recorte de 600s. Elas perderam, em média, -0,126R no custo-base e
-0,338R no estresse. Essa execução não é o resultado final; ela foi substituída pela repetição
conforme o protocolo, mas serve como evidência adicional de que os picos previstos eram falsos
positivos raros.

Como nenhum candidato passou o gate elementar, não há política para comparar com B1, calcular
PBO/DSR, abrir 2022, travar modelo ou testar trailing. Fazer essas etapas sem uma entrada positiva
seria procurar uma saída que maquiasse um sinal inexistente.

## Defeitos encontrados e fechados

1. Timestamps pandas com resolução diferente quebravam o piloto real; a fábrica passou a normalizar
   explicitamente para milissegundos e ganhou teste de regressão.
2. Índices de fill acima de 16,7 milhões perdiam inteiros ao serem compactados como `float32`; eles
   agora permanecem em `float64` e o painel foi rematerializado.
3. Pares de maior volume excediam 30 GB de memória; a materialização passou a usar partições anuais
   com contexto dos meses adjacentes e validação de paridade.
4. O primeiro grid fixou 300/700 árvores sem aplicar o early stopping declarado. O código ganhou
   teste explícito, o grid inteiro foi reexecutado com 2021-H1 como calibração e os artefatos antigos
   foram substituídos.

## Conclusão

O resultado não prova que o mercado não possui padrões. Ele demonstra algo específico e útil:
**neste banco, interseções de duração, intensidade, direção de bid/ask, persistência, spread,
volatilidade e sessão não superaram o custo de varejo em 30 segundos, 2 minutos ou 10 minutos** com
os modelos e gates declarados.

Portanto, não há estratégia confirmada para baixar a semana da Fusion/cTrader. Fazer isso agora
violaria a ordem combinada e transformaria a semana da corretora em dado de ajuste. Ela continua
reservada para uma futura política que primeiro passe desenvolvimento, D4 e S3.

O próximo experimento só é justificável se trouxer informação nova — negócios executados, agressão e
profundidade L2 de uma venue centralizada — ou uma hipótese econômica nova pré-registrada. Ajustar
mais thresholds, indicadores ou trailing sobre este mesmo top-of-book seria overfitting, não avanço.
