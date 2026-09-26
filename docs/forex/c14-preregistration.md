# Pré-registro C14: Gotobi, a compra de dólar dos importadores japoneses antes do fix de Tóquio

Registrado em 2026-09-26, antes do código e de qualquer resultado desta rodada, e **antes** de a rodada única de C13 no período 2003-2011 existir. C13 não condiciona nada ao calendário japonês. Hipótese escolhida por um painel de 4 pesquisadores e 2 juízes independentes, dentre 12 propostas.

## Mecanismo e evidência

No Japão, importadores pagam fornecedores nas datas "goto" (5, 10, 15, 20, 25, 30), em dólar, pela taxa TTM que cada banco fixa às 09:55 JST. Os bancos compram dólar antes do fix.

Evidência:

- Ito e Yamada (NBER w22820; Journal of International Economics, 2017; EBS 1999–2013) acham desequilíbrio de compra de USD e EUR antes do fix, mais forte nos dias 5 e 10.
- Bessho, Sugimoto e Suzuki (arXiv 2301.13204, 2023; 2018–2020) mostram o USDJPY subindo das 03:00 às 09:55 JST nos dias gotobi, e revertendo depois.
- Krohn, Mueller e Whelan (Journal of Finance, 2024) mostram o dólar subindo antes dos fixes e caindo depois.

Por que C11 não refuta: C11 testou o fix de Tóquio em **todos** os dias úteis, e só das 08:55 às 09:55. Os dias gotobi são cerca de 27% deles, e a alta vem sobretudo antes das 09:00.

## Regras (fixas)

**Dias gotobi:** os dias 5, 10, 15, 20, 25 e 30 de cada mês. Em fevereiro, o último dia faz o papel do 30; em meses de 31 dias, o 31 não conta. Se a data cai em sábado, domingo, feriado nacional japonês (lista oficial do Cabinet Office, `syukujitsu.csv`) ou entre 31/12 e 03/01, vale o dia útil de Tóquio anterior. Datas que caem no mesmo dia contam uma vez.

**Relógio:** o M1 já em UTC (C3). JST = UTC+9, sem horário de verão.

**Configurações:**

- **A:** compra USDJPY na primeira barra M1 a partir das 03:00 JST do dia gotobi (até 5 minutos depois) e vende no fechamento da barra das 09:54 JST. Em dia gotobi na segunda-feira, 03:00 JST cai com o mercado fechado. Nesse caso a entrada é na primeira barra 60 minutos ou mais depois da abertura da semana.
- **B:** A, mais uma venda de USDJPY na abertura da barra das 09:56 JST, zerada no fechamento da barra das 11:59 JST (a reversão depois do fix). O resultado do dia é a soma das duas pernas.
- **C:** a janela de A em USDJPY e em EURJPY, metade do tamanho em cada: duas operações no dia, agrupadas pelo dia.

A saída exige uma barra até 5 minutos antes do horário nominal; sem ela, o dia não conta.

**Stop de proteção:** 1 ATR diário, o ATR(14) dos dias FX completos antes da entrada. É checado minuto a minuto, primeiro dentro do minuto; o gap executa na abertura. Não há alvo.

**Custo:** o de C3 (USDJPY 0,9 pip; EURJPY 1,4 pip), ida e volta por perna; no estresse, o dobro.

**Swap:** a perna de A atravessa o rollover das 17:00 de Nova York. Cada rollover conta 1 dia, e o da quarta-feira conta 3. O valor segue o método de C13 com os juros do FRED (mês M−2) e markup de 1% ao ano: (r_base − r_cotada − markup) × dias ÷ 365 × preço de entrada, com o sinal do lado. No estresse: markup de 2%, e crédito positivo zerado (só débito conta).

**Unidade:** R = 1 ATR diário, líquido.

## Julgamento

**Descoberta, 2012-01 a 2026-08 (manifesto de C3).** Média de R por dia de operação e t agrupado por dia. Bootstrap estacionário de máximo t sobre as 3 configurações: 5.000 sorteios, bloco médio de 10 dias, semente 20261031. **Passa** se todas:

- p ajustado ≤ 0,05;
- média > 0 com custo normal e no estresse;
- ≥ 60% dos anos positivos;
- média nos dias gotobi **maior** que a da mesma janela nos dias úteis de Tóquio não-gotobi, o que atribui o efeito ao mecanismo;
- média > 0 em 2017-01 a 2026-08, depois da publicação de Ito e Yamada;
- **placebo:** em 20 cópias com o rótulo "gotobi" sorteado entre os dias úteis de Tóquio, na mesma quantidade por mês, no máximo 1 cópia alcança a média real.

**Período intocado, 2003-01 a 2011-12 (manifesto de C10, M1 de USDJPY e EURJPY).** Só as configurações que passarem. Uma rodada. p unilateral pelo t agrupado; Holm entre elas. **Aprovada:** p de Holm ≤ 0,05; média > 0 com custo normal e no estresse; ≥ 5 dos 9 anos positivos.

**Poder, escrito antes:** o esperado é de 2 a 4 pips líquidos por dia, com desvio de 25 a 35 pips, e cerca de 580 dias no período intocado. Mesmo com o efeito real, o t esperado fica perto de 1,5 a 2,5. Uma reprovação lá pode ser falta de poder, e isso será dito.

## Depois

Aprovada vai para a demo 10139135 (0,5% do capital por ATR diário, ou o volume mínimo), de 3 a 6 operações por mês. Sem aprovada, relatório e nada na demo. Emendas só antes de resultado, em `c14-registry.jsonl`.
