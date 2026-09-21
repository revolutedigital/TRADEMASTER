# Rodada 7 — proteção antecipada, etapa A

Status: **reprovada em 2021-Q2**. A etapa B não foi executada. H2 e todos os períodos posteriores
continuam fechados.

| Gatilho de breakeven | Trades | Ativação base | Média base | Ativação stress | Média stress | PF stress | Meses positivos |
|---:|---:|---:|---:|---:|---:|---:|---:|
| +0,1R | 2.584 | 64,63% | -0,240R | 46,87% | -0,442R | 0,139 | 0/3 |
| +0,2R | 2.359 | 56,68% | -0,249R | 41,37% | -0,451R | 0,182 | 0/3 |
| +0,3R | 2.228 | 49,78% | -0,253R | 35,50% | -0,460R | 0,215 | 0/3 |

Mover a proteção mais cedo funcionou mecanicamente, mas não economicamente. Em +0,1R, os trades
ativados subiram para 64,63% no base, porém seu ganho médio caiu para +0,145R; no stress, 46,87%
ativaram com +0,138R médio. A estratégia trocou stops por muitos breakevens pequenos sem preservar
cauda suficiente. Nenhuma tentativa teve um mês positivo.

Na tentativa +0,1R, metade dos stops stress ocorreu até 105,8 segundos e 75% até 207,5 segundos.
Isso sustenta uma hipótese distinta e testável: um timeout pré-ativação curto pode encerrar trades
que não respondem antes do stop cheio. Ela não foi testada nesta rodada porque a etapa A falhou.

Artefato: `a5ad88643fd956a81ae2d60d2f1a38db6a25166ce8fd2eff9a23822bf7021170`.
