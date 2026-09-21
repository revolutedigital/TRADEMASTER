# Pré-registro, rodada 7: proteção antecipada dos melhores sinais globais

Registrado em 2026-09-21 depois da reprovação Q2 da rodada 6 e antes de simular qualquer novo nível
de proteção. A rodada 6 mostrou que a entrada probabilística concentra movimentos e que trades
ativados são positivos, mas +0,5R demora demais: na melhor política base, 40,69% ativaram; no
estresse, 29,31%. Para empatar as distribuições observadas seriam necessários aproximadamente 58%
e 61%. H2 e todos os períodos posteriores continuam fechados.

## Elementos congelados

- modelos, calibradores, scores, pisos absolutos e ranking causal da rodada 6;
- Top P global fixado em **5%**, o menor corte da rodada 6 com pelo menos quatro mercados e o melhor
  stress entre os cortes com diversidade elegível;
- no máximo três posições globais e uma por par;
- stop inicial -1R líquido, breakeven verdadeiramente líquido, custos base/estresse e timeout total
  de seis horas;
- mesma agenda comum: uma vaga só libera após a saída mais tardia entre base e estresse;
- mesmos Q2, gates e fonte tick a tick.

A escolha de Top 5% usa o resultado exploratório de Q2 da rodada 6 e, portanto, todas as 18
tentativas anteriores continuam contando em PBO/DSR. Esta rodada não reapresenta Q2 como confirmação;
somente H2 poderá confirmar uma política.

## Etapa A — a proteção é cedo o suficiente?

Com timeout pré-ativação de 600 segundos e trailing de 0,5R fixos, testar somente três gatilhos de
breakeven: `{+0,1R; +0,2R; +0,3R}`. São três novas tentativas. A etapa A só segue se pelo menos uma
combinação tiver 100 trades, médias base e estresse positivas, dois dos três meses positivos, profit
factor stress acima de 1,05, quatro pares participantes e concentração máxima de 35% do lucro
positivo.

Se nenhuma passar, a rodada termina sem testar outros timeouts ou trailings.

## Etapa B — somente se A passar

Congelar o gatilho aprovado de maior média stress e testar timeout pré-ativação `{60; 180; 600}`
segundos × trailing `{0,5R; 1R; 1,5R}`: nove tentativas. A melhor política segue o mesmo ranking e
gates. O total acumulado será 30 tentativas: 18 da rodada 6, 3 da etapa A e 9 da etapa B.

Somente uma política aprovada na etapa B pode abrir 2021-H2 uma vez. 2022, D4, S3, semana da
corretora, Demo e LIVE continuam fechados.
