# Pré-registro, rodada 10: torneio exaustivo barato de hipóteses

Registrado em 2026-09-21 antes de executar o torneio. A rodada usa apenas os payoffs EURUSD já
materializados; não abre Q2, H2, 2022 nem envia ordens.

## Objetivo e limite honesto

Não existe um conjunto finito de “todas as estratégias possíveis”. Esta rodada cobre
exaustivamente as famílias que o banco atual consegue expressar sem criar dado novo e sem novo
replay de ticks. O objetivo é procurar uma região aprendível antes de gastar computação em mais
pares.

## Divisão temporal

- geração/ajuste: 2019–2020;
- seleção: 2021-01-01 a 2021-03-15;
- auditoria final: 2021-03-15 a 2021-04-01;
- purge de seis horas nas fronteiras.

O alvo é sempre o retorno líquido exato do gestor congelado na rodada 9, em base e stress. Compra
e venda são linhas separadas e as features direcionais são orientadas pelo lado.

## Universo de hipóteses

1. **Caudas univariadas:** para cada feature, valores abaixo de P10/P25 e acima de P75/P90.
2. **Interseções:** AND entre pares das 30 melhores caudas no treino, sem repetir a mesma feature.
3. **Regimes supervisionados:** folhas de árvores rasas com profundidade `{2; 3; 4}` treinadas no
   menor retorno entre base e stress.
4. **Regimes não supervisionados:** clusters MiniBatch K-Means com `k={4; 8; 16}` após escala
   robusta ajustada somente no treino.
5. **Rankings de modelo:** Ridge, Extra Trees, HistGradientBoosting, XGBoost regressivo, XGBoost
   para retorno positivo e XGBoost para retorno acima de +0,1R. Cada score testa os top
   `{1%; 2%; 5%; 10%; 20%}` definidos no treino.

Todas as configurações acima são fixas. Não haverá ajuste manual depois de observar a auditoria.

## Seleção e correção por procura

Hipóteses precisam de pelo menos 500 linhas no treino e 100 na seleção. A seleção ordena pelo menor
retorno médio entre base e stress e leva no máximo 25 regras distintas à auditoria.

Na auditoria, a incerteza é agrupada por dia FX. Cada uma das até 25 regras recebe intervalo
unilateral de 95% com correção de Bonferroni para o número efetivamente auditado. Uma hipótese só
sobrevive se tiver:

- pelo menos 50 eventos e oito dias FX;
- média base e stress positivas;
- limites inferiores corrigidos positivos em base e stress;
- lift mínimo de +0,05R contra a média incondicional nos dois cenários.

Esse P0 mede separabilidade, não autoriza portfólio. Se nenhuma hipótese passar, a informação atual
é considerada esgotada para operações curtas. Se alguma passar, somente então a regra vencedora
ganha um piloto P1 em EURUSD, GBPUSD e GBPJPY com não sobreposição e custos idênticos.
