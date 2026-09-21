# Pré-registro, rodada 2: calibração das famílias rápidas em três amostras

Registrado em 2026-09-20, **antes** de existir código da rodada 2 e antes de qualquer resultado dela. Este documento e `docs/forex/fast2-grid.json` são commitados primeiro; depois só mudam por **emenda registrada** em `docs/forex/fast2-registry.jsonl`, nunca por causa de resultado.

## Por que existe

A rodada 1 (`docs/forex/fast-preregistration.md`) testou 10 configurações com parâmetros fixos e nenhuma foi aprovada: todas perdem depois do custo (a menos ruim, F1a, deu −0,045 R por trade). O Igor decidiu não parar ali: existe muito passado para testar hipóteses, e **calibrar** cada família (varrer os parâmetros) é o passo natural. Calibrar do jeito ingênuo, ajustando até a amostra ficar bonita e chamando o melhor de estratégia, é o que produziu o resultado do G0 (+0,196 R na descoberta, +0,019 R em dado novo). Esta rodada calibra de forma que o resultado valha: **grade fixada antes, estatística que paga pelo número de combinações, e três amostras em que a última nunca foi vista**.

## O que a rodada 1 já mostrou (divulgação)

Quem escreve esta grade conhece a tabela-resumo da rodada 1 na amostra de 2019-01 a 2024-08 (R médio por configuração; nenhuma aprovada; F1a a menos negativa, os controles muito negativos) e o comportamento geral do mercado nesses anos. Isso pode influenciar sem querer a escolha da grade. **A proteção é estrutural:** a busca desta rodada roda em dado anterior a 2019, que nenhuma hipótese rápida jamais viu, e a amostra de 2019 a 2024 entra só como réplica.

## Amostras e papéis

- **S0, descoberta:** 2014-11 a 2018-11 (49 meses por par; ver a emenda de 2026-09-20), ticks bid/ask do HistData convertidos em M1. É uma **fonte diferente** da usada de 2019 em diante: o relógio segue o horário de verão dos EUA (o espelho de 2019+ segue as datas europeias), e o preço difere do da Dukascopy em cerca de 0,1 pip no fechamento horário (mediana; 94% a 97% das horas dentro de 0,5 pip, medido no EURUSD em seis meses ao redor das trocas de horário), com spread mediano cerca de 0,1 pip maior (0,4 contra 0,3 pip no EURUSD), o que deixa os custos um pouco **mais conservadores** que os da Fusion. O mês 2018-12 fica fora (a fonte muda em 2018-12-16). Os meses de 2011-01 a 2014-10 foram baixados e **não entram**: até 2014-10 a fonte cota spreads fixos de número redondo (2,0 a 5,0 pips, a mesma mediana mês após mês, em todos os pares), que nenhum feed de mercado mostra e que a Fusion não cobra; em 2014-11, nos 10 pares ao mesmo tempo, o spread passa a variar e cai para 0,3 a 1,0 pip. Esses 46 meses ficam reservados, sem uso. O relógio foi conferido por dois meios independentes: contra o oráculo horário da Dukascopy em 2017-2018, e pela hora da abertura semanal do mercado no relógio bruto (17h de Nova York em todos os domingos das 48 amostras de 2011 a 2018).
- **S1, réplica:** 2019-01 a 2024-08, a amostra de descoberta da rodada 1 (mesmo manifesto de meses limpos, `docs/forex/fast-data-manifest.csv`; os 5 meses com buraco, 2023-03 a 2023-07, ficam fora). Ela já foi vista pela rodada 1 (10 hipóteses, todas perdedoras), por isso **não** serve de descoberta aqui.
- **S2, confirmação final, congelada:** 2024-09 a 2026-08. Nenhuma rodada corre nela antes do relatório de réplica estar commitado, e só corre uma vez.
- **Pares (10):** os mesmos da rodada 1. Horários, dia FX, barra ausente e trechos contínuos: como na rodada 1.

### Qualidade do dado de S0 (critério fixo, só de qualidade; decidido antes de rodar qualquer estratégia)

Um par-mês de S0 entra se **todas**: (1) minutos ≥ 85% da mediana do par (sem buraco); (2) nenhum minuto com ask abaixo do bid; (3) mediana do spread de abertura ≤ 3 pips; (4) **relógio:** em todas as semanas do mês a primeira barra de domingo, convertida de UTC para Nova York, abre entre 17:00 e 17:59 (um relógio errado por uma hora abre às 16h ou às 18h; num domingo de feriado ou de mercado fino a primeira cotação pode vir até 17:59); (5) majors nos meses com oráculo horário (2016-09 a 2018-11): ao menos 90% das horas com fechamento dentro de 0,5 pip do oráculo, no bid e no ask; (6) cruzados: erro mediano de triangulação com os majors ≤ 1 pip. O resultado vai para `docs/forex/fast2-data-manifest.csv`, commitado antes de qualquer estratégia rodar em S0.

## Custos (fixos, os da rodada 1)

`FUSION_ZERO` como base e `STRESS` no estresse, com bid e ask reais de cada barra (em S0, os do feed da época), comissão da conta Zero da Fusion, slippage e conversões idênticos aos da rodada 1 (`docs/forex/fast-preregistration.md`, seção "Custos"). O AUDNZD sintético mantém as duas pernas custadas.

## Hipóteses (fixas: a grade inteira e nada fora dela)

`docs/forex/fast2-grid.json` lista as **212 combinações**, geradas por `backend/scripts/research/fx_fast2_grid.py`; o registro guarda o SHA-256 do arquivo. Mecanismo, regra e custos de cada família são os da rodada 1; só variam os parâmetros abaixo.

- **F1a e F1b, rompimento de faixa (M15), 48 combinações cada:** razão retorno/risco do alvo {1,0; 1,5; 2,5} × filtro de largura da faixa em ATR diários {0,3–1,2; 0,3–0,8; 0,6–2,0; sem filtro} × fim da janela de sinal {09:30; 11:00} × saída forçada {12:30; 16:30} (Londres) ou {12:00; 16:00} (Nova York).
- **F2a (antes do fixing, compra de dólar), 18:** entrada {14:30; 15:00; 15:30} × saída {15:55; 16:00} × stop em ATR M5 {2; 3; 5}. **F2b (depois, venda de dólar), 18:** entrada {16:05; 16:15} × saída {16:45; 17:00; 17:30} × stop {2; 3; 5}. Sete pares com dólar.
- **F3, reversão de pico (M5), 32:** limiar {3; 4; 6; 8} × stop em faixas do pico {0,5; 1,0} × retração-alvo {0,50; 0,75} × saída por tempo {12; 24} barras. Corpo mínimo 0,6 do range e spread ≤ 2× a mediana, como na rodada 1.
- **F5a (EURGBP) e F5b (AUDNZD sintético), razão de dois pares (H1), 24 cada:** z de entrada {1,5; 2,0; 2,5; 3,0} × z do stop {3,5; 4,5} × saída por tempo {60; 120; 240} barras, janela de 480 barras.
- **Fora da grade:** os controles C1 e C2 (a calibração com dado sem vantagem já mede a taxa de falsa aprovação do critério, e o C1 sozinho tem mais de um milhão de trades por rodada) e a F4 (sem fonte de calendário).
- **Dentro da grade:** as 8 configurações da rodada 1 (F1a, F1b, F2a, F2b, F3a = limiar 4, F3b = limiar 6, F5a, F5b), com parâmetros idênticos (um teste confere).

**Número de testes: K = 212.**

## Critério (o da rodada 1, com K = 212)

Estatística, unidade (o trade), R e t com erro padrão por cluster de dia FX: como na rodada 1.

- **Chave A:** bootstrap estacionário de dias FX (bloco médio de 10 dias, 5.000 reamostragens, semente 20260922) com o máximo-t entre as configurações avaliadas na amostra.
- **Chave B:** placebo sintético (a cada minuto o retorno troca de sinal com probabilidade 1/2, uma moeda por minuto igual nos 10 pares; máxima, mínima e spreads preservados), **200 conjuntos** por amostra (semente 20260923), valor-p pelo máximo-t entre as configurações avaliadas, piso 1/201. Cada conjunto é simulado na mesma amostra em que o critério é aplicado.
- **Aprovada em S0** se todas: (1) valor-p ajustado das duas chaves ≤ 0,05 (máximo sobre as 212); (2) média de R > 0 no custo base; (3) ≥ 60% dos pares aplicáveis com média > 0; (4) ≥ 300 trades; (5) média de R > 0 no estresse. Inconclusiva (não reprovada) só se não aprovada e com < 300 trades ou efeito mínimo detectável (3,4 × erro padrão) acima de 0,15 R.
- **Replicada em S1:** as m aprovadas em S0, sem mudar nada, com o máximo-t restrito às m: valor-p das duas chaves ≤ 0,05, média de R > 0 (base e estresse), ≥ 60% dos pares positivos, ≥ 300 trades.
- **Confirmada em S2:** as m′ replicadas, uma única vez, mesmas regras com máximo entre as m′ e ≥ 100 trades.
- **Calibração antes do dado real:** os 200 conjuntos sintéticos de S0 também medem a taxa de falsa aprovação do critério inteiro sobre as 212 (cada conjunto avaliado contra os outros 199), esperada ≤ 5%; se passar de 8%, o critério está descalibrado, corrige-se **antes** da primeira execução em dado real (com o motivo commitado) e a calibração é refeita.

## Ordem de execução

1. Este commit: pré-registro, grade, registro append-only com a linha de criação (SHA-256 do documento e da grade) e o código que impõe as regras.
2. Dado de S0 baixado, convertido e validado; manifesto de S0 commitado **antes** de qualquer estratégia rodar nele.
3. Calibração (placebo em S0), relatório commitado.
4. Descoberta em S0, relatório commitado com a lista das aprovadas.
5. Réplica em S1 só das aprovadas, relatório commitado.
6. Confirmação em S2 só das replicadas, relatório commitado.
7. Se nada sobreviver a uma etapa: paro e converso com o Igor; S2 continua intacta.

## Registro e proteção

`docs/forex/fast2-registry.jsonl` é append-only: `registry_created` e `amendment` (com o SHA-256 deste documento e da grade; a emenda exige o motivo), `calibration_report`, `discovery_report`, `replication_report` e `run`. `backend/scripts/research/fx_fast2_registry.py` recusa configuração fora da grade, etapa fora de ordem e configuração que não passou a etapa anterior, e um teste automático falha se este documento ou a grade mudarem sem emenda. Correção de defeito só vale se nascer de um teste que falha mostrando um desvio **deste documento**, escrito antes da correção; a rodada é repetida e registrada com o motivo.

## O que NÃO será feito

- Mudar a grade, os pares, os custos, o critério ou as amostras depois de ver qualquer resultado da rodada.
- Rodar qualquer coisa em S2 antes do relatório de réplica, ou mais de uma vez.
- Somar amostras para "resgatar" uma etapa que falhou; uma estimativa combinada é só informativa.
- Olhar o resultado por configuração antes de o relatório de calibração estar commitado.

## O que se espera, antes de ver o dado (palpite sem medida por trás)

O valor desta rodada é responder "funciona ou não funciona" com rigor, e não achar ouro: a rodada 1 já mostrou que o terreno é hostil (custo de varejo contra vantagens pequenas). Palpite: 2% a 5% de chance de ao menos uma configuração passar por S0, S1 e S2; algo em torno de 10% a 15% de chance de aparecer alguma aprovada em S0 só por regime (2014 a 2018 foi de volatilidade baixa e dólar em tendência), que a réplica em S1 deve derrubar. Se ninguém sobreviver, a pergunta "esta grade de regras de curto prazo em forex de varejo tem vantagem depois do custo?" fica respondida com uma amostra de 4,1 anos virgem, uma de 5,6 anos e uma de 2 anos.

## Emendas

- **2026-09-20, antes de qualquer estratégia rodar em S0 e sem olhar resultado nenhum:** (1) S0 passa de 2011-01 a 2018-11 para **2014-11 a 2018-11**, e (2) a janela do relógio passa de 17:00–17:15 para 17:00–17:59. Motivo: ao construir o manifesto de qualidade, o spread mediano mensal mostrou que a fonte cota spreads fixos de número redondo até 2014-10 (2,0 a 5,0 pips, iguais mês após mês, nos 10 pares) e passa a spread variável de 0,3 a 1,0 pip em 2014-11, nos 10 pares no mesmo mês; o critério (3) (mediana ≤ 3 pips) foi escrito para tirar dado que não representa o custo em teste e, como estava, deixava passar esses spreads fixos. A janela de 15 minutos do relógio reprovava 119 pares-mês só porque a primeira cotação de um domingo de feriado ou de mercado fino veio entre 17:16 e 17:59: em todos os 407 domingos de 2011 a 2018 do EURUSD a abertura cai na hora 17, e um relógio errado por uma hora cairia às 16h ou às 18h. As duas mudanças vêm só de qualidade de dado; a grade, os pares, os custos, o critério estatístico e as outras amostras não mudam.
- **2026-09-20, correção de execução antes de qualquer resultado:** a primeira tentativa de calibração abortou na guarda do simulador, antes de calcular ou gravar resultado, porque o calendário de rollover padrão começava em 2015 e S0 começa em 2014-11. Um teste de regressão reproduziu a falha antes da correção; o calendário passou a começar em 2014. Amostra, grade, custos e critério estatístico não mudaram.
- **2026-09-20, correção do motivo da emenda anterior (só documentação):** a sessão do Codex mostrou que a maior parte dos 119 reprovados no relógio era um artefato da checagem, e não abertura tardia: um mês em UTC que começa numa segunda-feira às 00:00 UTC começa, em Nova York, no domingo às 20:00, depois da abertura real, e a primeira barra "de domingo" do recorte não é a abertura. A checagem passou a ignorar esse domingo parcial (commit 8546747). Só uma minoria foi de abertura tardia de verdade (por exemplo EURUSD 2016-12-25, às 17:30), e a janela de 17:00 a 17:59 cobre esses casos. Nenhuma outra regra muda.

