# Rodada 9 — piloto P0 de expectativa do gestor

Data: 2026-09-21  
Escopo: EURUSD, sem Q2  
Resultado: hipótese reprovada no funil barato

## Resultado

O piloto usou 297.057 linhas direcionais de 2019–2021-Q1. A auditoria final continha 12.495 linhas
por cenário e não participou do treino, early stopping ou ajuste dos calibradores.

| Cenário | Melhora de MSE | Spearman | Lift do decil superior | Erro da média | Gate |
|---|---:|---:|---:|---:|---:|
| Base | +0,0597% | 0,0433 | +0,0073R | 0,0056R | Reprovado |
| Stress | +0,0726% | 0,0120 | +0,0167R | 0,0020R | Reprovado |

O pré-registro exigia, nos dois cenários, melhora mínima de 1% no MSE, Spearman acima de 0,02,
lift mínimo de +0,05R e erro de média de até 0,05R. O base falhou em melhora e lift; o stress falhou
em melhora, correlação e lift.

O melhor base misturou 25% do modelo hurdle com 75% da regressão direta e calibração affine. O
melhor stress foi regressão direta sem transformação. A comparação incluiu 15 combinações por
cenário, todas declaradas antes da auditoria.

## Decisão

P1 com três pares não será materializado. O painel de dez pares e o Top P no Q2 também não serão
executados. H2, 2022 e todas as etapas de ativação continuam fechados.

As partições já concluídas de EURUSD e GBPUSD permanecem em cache para eventual pesquisa futura,
mas não justificam expansão. O resultado mostra que, com as features atuais, o payoff completo do
gestor quase não é mais previsível que a média e o ranking não separa uma cauda economicamente útil.

## Processo adotado daqui em diante

Novas hipóteses passam primeiro por um piloto de aprendibilidade de um par e uma janela curta. Só
um sinal material fora do treino autoriza três pares; só três pares aprovados autorizam a bateria
completa. Isso reduz custo computacional e evita consumir tempo em backtests de uma variável-alvo
que o modelo ainda não consegue ordenar.
