# Rodada 10 — torneio barato de hipóteses

Data: 2026-09-21  
Escopo: EURUSD, informação e payoffs já existentes  
Resultado: zero hipótese aprovada

## Cobertura

Foram geradas 702 hipóteses antes da auditoria:

| Família | Hipóteses |
|---|---:|
| Caudas univariadas | 204 |
| Interseções de duas condições | 414 |
| Folhas de árvores rasas | 26 |
| Clusters não supervisionados | 28 |
| Rankings de seis famílias de modelo | 30 |

As 25 melhores na seleção temporal foram auditadas com correção de Bonferroni e erro agrupado por
dia FX. Nenhuma apresentou média positiva, portanto nenhuma chegou perto de satisfazer os limites
inferiores corrigidos.

## Melhores recortes na auditoria

| Hipótese | Eventos | Base | Stress | Lift base | Lift stress |
|---|---:|---:|---:|---:|---:|
| Árvore profundidade 2, folha 6 | 60 | -0,145R | -0,289R | +0,100R | +0,150R |
| Árvore profundidade 3, folha 13 | 54 | -0,175R | -0,303R | +0,070R | +0,136R |
| Extra Trees, top 2% | 121 | -0,135R | -0,358R | +0,110R | +0,081R |
| HistGradient, top 1% | 153 | -0,186R | -0,362R | +0,059R | +0,077R |

Algumas regras reduziram a perda em relação ao incondicional, mas nenhuma atravessou zero. O melhor
limite inferior base ainda foi -0,220R e o stress -0,416R.

## Decisão

O conjunto atual de informação FX — preço, spread, frequência de updates, direção bid/ask,
volatilidade, sessões, regimes e suas interseções — está esgotado para esse gestor e horizonte
curto. Não serão testadas mais combinações sobre o mesmo banco.

O próximo piloto precisa acrescentar informação inexistente neste feed: volume realmente negociado
e lado agressor. Por isso a pesquisa seguinte migra somente o P0 barato para dados públicos de
negócios executados em mercado centralizado. Isso não autoriza operação, conta ou capital real.
