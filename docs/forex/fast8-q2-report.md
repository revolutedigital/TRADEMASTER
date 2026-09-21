# Rodada 8 — time stop antes da ativação

Data: 2026-09-21  
Janela aberta: somente 2021-Q2  
Resultado: arquitetura reprovada; H2 e 2022 permaneceram fechados

## Decisão

Nenhum dos três timeouts pré-registrados produziu expectativa positiva. A política menos negativa
foi o corte em 60 segundos, com média de -0,2431R no cenário base e -0,4520R no stress. Os três
meses e os seis pares participantes ficaram negativos.

Com 24 tentativas acumuladas nas rodadas 6, 7 e 8, fica encerrada a arquitetura atual de entrada
por probabilidade de primeiro toque, breakeven e trailing com o conjunto atual de features. Nenhuma
política foi congelada. 2021-H2, 2022, D4, S3, a semana de corretora, Demo e LIVE não foram abertos.

## Resultado do gate Q2

| Timeout antes da ativação | Trades | Média base | Média stress | PF stress | Meses positivos | Pares |
|---:|---:|---:|---:|---:|---:|---:|
| 60 s | 4.455 | -0,2431R | -0,4520R | 0,049 | 0/3 | 6 |
| 120 s | 3.412 | -0,2484R | -0,4528R | 0,079 | 0/3 | 6 |
| 210 s | 2.931 | -0,2526R | -0,4580R | 0,101 | 0/3 | 6 |

O gate exigia simultaneamente médias base e stress positivas, pelo menos dois meses positivos,
PF stress acima de 1,05, no mínimo quatro pares e concentração máxima de 35%. Somente volume e
diversidade foram atendidos.

## Onde a hipótese falhou

O time stop reduziu alguns stops cheios, mas a saída antecipada ainda ocorreu tarde demais em
relação ao custo e ao movimento adverso. No cenário stress:

| Timeout | Ativados | Stops cheios | Timeouts, média | Runners, média |
|---:|---:|---:|---:|---:|
| 60 s | 14,28% | 637 a -1,0217R | 3.182 a -0,4575R | 636 a +0,1463R |
| 120 s | 25,67% | 878 a -1,0221R | 1.658 a -0,4620R | 876 a +0,1350R |
| 210 s | 34,29% | 1.051 a -1,0217R | 875 a -0,4624R | 1.005 a +0,1355R |

Esperar mais aumenta a proporção de ativações, mas também deixa mais operações alcançarem o stop
cheio. Cortar antes diminui o dano unitário, porém cria muitos encerramentos por volta de -0,46R.
O payoff médio dos runners, perto de +0,14R sob stress, não compensa nenhum desses dois grupos.

## Robustez transversal

- Abril, maio e junho foram negativos nos três timeouts, em base e stress.
- EURUSD, GBPUSD, GBPJPY, USDJPY, USDCAD e EURJPY foram negativos individualmente.
- O limite inferior bootstrap de 95% da média base permaneceu negativo em todas as tentativas.
- A mesma agenda de entradas foi usada para base e stress, com no máximo três posições globais e
  uma posição por par.

## Interpretação

A calibração de Top P conseguiu ordenar a probabilidade do evento usado como alvo, mas esse evento
não equivale a expectativa líquida positiva depois de custos e gestão. O problema não é apenas
"proteger mais cedo": a variável-alvo precisa representar o resultado econômico completo do trade,
incluindo magnitude e caminho, em vez de somente a chance de tocar um nível antes do stop.

Uma próxima família de pesquisa, se aberta, deve ser pré-registrada como um modelo direto de
expectativa líquida ou distribuição de payoff. Ela não deve reutilizar Q2 para escolher features ou
parâmetros e não pode abrir os períodos protegidos sem passar novamente pelo gate.
