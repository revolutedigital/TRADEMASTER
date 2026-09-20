# Pré-registro: estratégias rápidas de forex (minutos a horas)

Registrado em 2026-09-20, **antes** de existir código de estratégia rápida e antes de qualquer resultado delas em qualquer dado de M1. Este documento é commitado primeiro. Depois disso ele só muda por **emenda registrada** (seção "Registro e proteção do documento") e nunca por causa de resultado.

## Por que existe

O teste com estratégias lentas mostrou o que acontece sem esta disciplina: a melhor de 12 configurações da descoberta deu +0,196 R por trade e, em 7 anos de dados novos, deu +0,02 R (t = 0,4). Quanto mais variantes se testa, maior o melhor resultado obtido por puro acaso, e com dados de 1 minuto, 10 pares e várias famílias o espaço de variantes cresce rápido. Este registro limita o número de hipóteses, fixa os parâmetros, reserva uma amostra que nenhuma estratégia rápida toca antes da hora e calibra o critério contra dados sem vantagem nenhuma.

## Dados e amostra

- **Pares (10):** EURUSD, GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF, EURJPY, GBPJPY, EURGBP.
- **Fonte:** ticks bid/ask do HistData convertidos em M1 bid/ask e validados contra a Dukascopy (`docs/forex/data-m1-spike.md`). As barras M5, M15 e H1 são agregadas do M1: abertura do primeiro minuto, máxima, mínima e fechamento do último, bid e ask separados.
- **Amostra:** 2019-01-01 a 2026-08-31. Os 5 meses com buraco na fonte (2023-03 a 2023-07) ficam fora em todos os pares. Qualquer outro par-mês que reprove na validação contra a Dukascopy (≥ 99% das horas idênticas e nenhuma cotação cruzada) também fica fora, decidido só por esse critério de qualidade do dado, antes de rodar qualquer estratégia. Posição aberta no fim de um trecho contínuo de dados é fechada na última barra do trecho.
- **Descoberta:** 2019-01 a 2024-08. **Confirmação, congelada:** 2024-09 a 2026-08 (24 meses). Nenhuma estratégia rápida roda nela antes de o relatório de descoberta estar commitado.
- **O que esta amostra já viu (divulgação):** as estratégias lentas do G0 (indicadores clássicos em velas de H4 e diárias, 7 majors) rodaram em 2023-09 a 2026-08 (descoberta) e em 2016-09 a 2023-08 (confirmação), ou seja, em toda esta janela. São hipóteses e resolução diferentes, mas não existe período que nenhum teste do projeto tenha visto. Também rodou sobre M1 real o cruzamento de médias de teste do simulador (regressão e velocidade, no EURUSD), sem análise de resultado financeiro. Além disso, quem escreveu estas hipóteses conhece o comportamento geral do mercado nesses anos, o que pode influenciar sem querer a escolha delas. O que vale: nenhuma das hipóteses abaixo rodou em nenhum dado.
- **Horários** são locais de Londres (Europe/London) e de Nova York (America/New_York), cada um com o seu horário de verão. O dia FX vai de 17:00 a 17:00 de Nova York.
- **Barra ausente:** a decisão é tomada no fechamento de uma barra e a ordem é executada na abertura da barra seguinte que existe. Uma entrada é **cancelada** se a barra em que ela executaria abre mais de um período depois da barra que deu o sinal (buraco de dados, fim de semana): nenhum trade entra em horário diferente do que a regra nomeia. Sem a barra de decisão (a anterior à de entrada) não há sinal e o trade daquele dia não acontece. Uma saída por horário sai na primeira barra existente depois da barra de decisão.

## Custos (fixos)

- **Base** (`FUSION_ZERO` em `backend/app/fx/sim/costs.py`): bid e ask reais de cada barra (compra no ask, venda no bid); comissão da conta Zero da Fusion na cTrader: 2,25 **na moeda base do par** por lote de 100.000 por lado (EURUSD 1 lote = EUR 2,25; AUDUSD = AUD 2,25; fonte: página da Fusion para a cTrader), convertida em dólares com o preço mediano do período e em pips de cada par; no AUDNZD sintético, AUD 2,25 na perna do AUDUSD e NZD 2,25 × preço na do NZDUSD; slippage de 0,1 pip mais 0,05 × o range médio das 20 barras anteriores do mesmo timeframe da estratégia (só barras já fechadas; dobrado no AUDNZD sintético, que tem duas pernas); swap zero.
- **Estresse** (`STRESS`): spread ×2, slippage de 0,3 pip mais 0,10 × o range médio das 20 barras anteriores, mesma comissão.
- **Sensibilidade** (`ADVERSE_SWAP`, só informativa, não entra em nenhum critério): swap adverso de 0,3 pip por dia.
- Toda entrada acontece na abertura da barra seguinte à do sinal. Stop e alvo ficam nos **níveis** que a regra define, medidos a partir do fechamento do mid da barra do sinal (extremo da faixa, extremo do pico, média da janela); uma ordem cujo stop ou alvo já ficou para trás do preço de execução é recusada, como uma corretora faria; o R usa a distância real do preço de execução ao stop.

## Hipóteses (fixas: as únicas)

Cada configuração usa os parâmetros abaixo, **sem varredura e sem ajuste**. Cada variante declarada (a, b) conta como um teste. Os detalhes de implementação que este texto não lista (arredondamento, ordem de eventos dentro de uma barra) seguem a regra mais simples e mais conservadora, são fixados em código e em teste antes da primeira execução em dado real e não podem ser escolhidos olhando resultado.

**F1a. Rompimento da faixa asiática na abertura de Londres (M15).** Mecanismo (hipótese): liquidez e volatilidade sobem na abertura e o preço tende a continuar depois de romper a faixa formada antes. Faixa = máxima e mínima do mid entre 00:00 e 08:00 (Londres); dias com menos de 24 das 32 barras M15 dessa janela são pulados. Sinal: o primeiro fechamento M15 do mid, entre as barras que abrem de 08:00 a 10:45 (Londres), acima da máxima da faixa (compra) ou abaixo da mínima (venda). No máximo um trade por par por dia. Filtro: largura da faixa entre 0,3× e 1,2× o ATR diário (média do true range do mid dos 14 dias FX completos anteriores). Stop: o extremo oposto da faixa. Alvo: 1,5× a distância do stop. Saída forçada às 16:30 (Londres). Pares: os 10.

**F1b. Rompimento na abertura de Nova York (M15).** Faixa = 03:00 a 08:00 (Nova York); dias com menos de 15 das 20 barras M15 dessa janela (os mesmos 75% da F1a) são pulados; sinais nas barras que abrem de 08:00 a 10:45 (Nova York); mesmos filtro de largura, stop e alvo; saída forçada às 16:00 (Nova York). Pares: os 10.

**F2a. Fluxo antes do fixing de Londres (M5).** Mecanismo (hipótese): rebalanceamento de moeda perto do fixing de 16:00 (Londres) empurra o dólar antes e devolve depois. Comprar dólar (vender o par quando o dólar é a moeda cotada; comprar o par quando é a base) na abertura da barra M5 das 15:00 (Londres) e sair na abertura da barra das 15:55. Stop de proteção de 3× o ATR M5 (14 barras). Sem alvo. Pares: os 7 com dólar (EURUSD, GBPUSD, AUDUSD, NZDUSD, USDJPY, USDCAD, USDCHF).

**F2b. Fluxo depois do fixing (M5).** Vender dólar na abertura da barra das 16:05 (Londres) e sair na abertura da barra das 17:00; mesmo stop. Pares: os mesmos 7.

**F3a. Reversão depois de pico de volatilidade, limiar 4 (M5).** Pico: range da barra M5 ≥ 4× a mediana do range das 48 barras M5 anteriores e |fechamento − abertura| ≥ 0,6× o range, com o spread do fechamento ≤ 2× a mediana dos spreads das 48 barras anteriores. Entrada contra a direção do pico. Stop: 0,5× o range do pico além do extremo do pico. Alvo: retração de 50% do range do pico. Saída por tempo após 12 barras. Sem entrada entre 16:55 e 17:15 (Nova York). Uma posição por par. Pares: os 10.

**F3b. Mesma regra com limiar 6.**

**F4. Deriva depois de notícia (M5), condicionada ao calendário do step 5.** F4 entra somente se, antes da execução da descoberta, `docs/forex/calendar-source.md` documentar uma fonte com horário de divulgação verificável para os eventos de alto impacto de 2019-01 a 2026-08 (cobertura ≥ 90% dos eventos de NFP, CPI e decisão de juros de Fed, BCE, BoE e BoJ) e termos de uso compatíveis. A decisão é registrada em commit próprio, antes de qualquer resultado. Se não, F4 sai do escopo. Regra: evento de alto impacto (rótulo da fonte) na moeda de um dos lados do par. Entrada na abertura da barra M5 que começa 10 minutos depois da divulgação, na direção do movimento líquido do mid entre o instante da divulgação e essa abertura, se o movimento for ≥ 10 pips (o mesmo limite para todos os pares) e o spread da barra de entrada for ≤ 2× a mediana do spread do mesmo horário nos 20 dias anteriores. Stop: 0,5× o movimento. Alvo: 1,0× o movimento. Saída por tempo após 60 minutos. Um trade por par por evento.

**F5a. Pares correlacionados, EURGBP (H1).** Mecanismo (hipótese): o EURGBP é exatamente a razão EURUSD ÷ GBPUSD, dois pares muito correlacionados, e oscila em faixa; operá-lo direto equivale ao spread entre os dois com uma perna e um spread só. z = (fechamento do mid − média das 480 barras H1 anteriores) ÷ desvio-padrão dessas 480 barras. Entrada contra o desvio quando |z| ≥ 2,0. Alvo: a média no instante da entrada (z = 0). Stop: média ± 3,5 desvios do instante da entrada. Saída por tempo após 120 barras. Uma posição. Par: EURGBP.

**F5b. Pares correlacionados, AUDNZD sintético (H1).** A razão AUDUSD ÷ NZDUSD construída minuto a minuto com o bid e o ask de cada perna (comprar o sintético = comprar AUDUSD no ask e vender NZDUSD no bid), com o spread, o slippage e a comissão das duas pernas contados. Mesma regra da F5a. A construção exata é fixada em código e em teste antes da primeira execução em dado real.

**C1. Controle negativo, reversão em M1.** Bandas de Bollinger (20, 2) sobre o fechamento do mid em M1; entrada contra o fechamento fora da banda; stop de 3 pips, alvo de 3 pips, saída por tempo após 30 barras; sem entrada entre 16:55 e 17:15 (Nova York); uma posição por par. Pares: os 10.

**C2. Controle negativo, momentum em M5.** Retorno do mid de uma barra M5 (fechamento a fechamento, em pips) com valor absoluto ≥ 2,5 desvios-padrão dos retornos das 48 barras anteriores: entra na direção do movimento; stop de 3 pips, alvo de 3 pips, saída por tempo após 12 barras; mesma exclusão do rollover; uma posição por par. Pares: os 10.

### Leituras fixadas na implementação

Pontos que o texto acima deixa em aberto e que o código resolveu, registrados aqui antes de qualquer resultado real (conferidos por uma auditoria independente do código contra este documento):

- **F5:** não há trade se o z já está em |z| ≥ 3,5 no sinal (o stop ficaria para trás); desvio-padrão populacional (ddof = 0), como no C1; as 480 barras são as 480 existentes, e atravessam fins de semana e buracos.
- **Blackout de F3, C1 e C2 (16:55 a 17:15, Nova York):** vale o instante da entrada (o fechamento da barra do sinal) no intervalo [16:55, 17:15).
- **Máxima e mínima do mid** em M5, M15 e H1: metade da soma da máxima do bid com a máxima do ask (e o mesmo para a mínima), o que deixa a faixa um pouco mais larga.
- **Filtro de largura da F1:** fechado nas duas pontas (0,3× e 1,2×).
- **AUDNZD sintético (F5b):** a máxima e a mínima do sintético em cada minuto vêm da razão na abertura e no fechamento do minuto.
- **Alvo:** executa por toque no preço exato, sem slippage; o stop executa com slippage e, se o mercado abre além dele, no preço da abertura.
- **C2:** o retorno de fechamento a fechamento inclui o de reabertura depois de um fim de semana ou buraco.
- **Trechos contínuos:** cada trecho contínuo de dados é simulado com estado novo, e posição aberta no fim do trecho fecha na última barra.

**Número de testes: K = 10** (F1a, F1b, F2a, F2b, F3a, F3b, F5a, F5b, C1, C2), ou 11 com F4. Os controles esperam falhar: se algum for aprovado, o resultado é tratado como suspeita de defeito de dado ou de custo, e não como achado. Eles ficam na família do critério de propósito (deixam o critério um pouco mais conservador).

## Critério (fixo)

**Unidade:** o trade. R = resultado líquido em pips ÷ distância inicial do stop em pips (do preço de entrada executado ao stop). **Estatística de uma configuração:** t = (média de R) ÷ (erro padrão robusto por cluster, sendo o cluster o dia FX da entrada), sobre todos os trades de todos os pares aplicáveis. Com N_d trades e soma S_d de R no dia d, a média é ΣS ÷ ΣN e o erro padrão é raiz(Σ (S_d − média × N_d)²) ÷ ΣN.

**Chave A, bootstrap de dias com máximo-t** (calibra o acaso de escolher a melhor de K usando a dependência que o dado real tem): 5.000 reamostragens estacionárias de dias FX (bloco médio de 10 dias), a mesma reamostragem para todas as configurações e pares. Em cada uma, o t de cada configuração é recalculado e centrado na média real dela (hipótese nula: média zero). O valor-p ajustado de uma configuração é a fração das reamostragens em que o **máximo** desses t entre as K alcança o t real dela, com o piso de 1/5.001. Semente 20260920.

**Chave B, placebo sintético** (calibra o acaso usando dado que não tem vantagem por construção): 300 conjuntos de dados em que, a cada minuto, o retorno do mid troca de sinal com probabilidade 1/2 (uma moeda por minuto, a mesma nos 10 pares, o que preserva a correlação entre pares e a coerência dos cruzados), a máxima e a mínima do minuto espelham junto, e os spreads originais ficam intocados. Isso zera qualquer tendência, reversão ou deriva por horário e mantém a volatilidade por horário, o agrupamento de volatilidade, os spreads e os buracos reais. O valor-p ajustado de uma configuração é a fração dos conjuntos em que o máximo do t entre as K alcança o t real dela, com o piso de 1/301. Semente 20260921.

**Aprovada na descoberta** se todas: (1) o valor-p ajustado das duas chaves é ≤ 0,05; (2) a média de R é > 0 no custo base; (3) ≥ 60% dos pares aplicáveis têm média de R > 0; (4) ≥ 300 trades; (5) a média de R é > 0 no estresse.

**Inconclusiva, e não reprovada** (vale só para configuração que não foi aprovada): menos de 300 trades, ou efeito mínimo detectável acima de 0,15 R (efeito mínimo detectável = 3,4 × o erro padrão da média de R, que é significância unilateral de 0,5% e poder de 80%). Vira "inconclusiva por falta de amostra" no relatório e não conta como evidência de ausência de vantagem.

**Confirmada** se, na amostra congelada, sem mudar nada e com a família reduzida às m configurações aprovadas: o valor-p ajustado das duas chaves (agora com o máximo-t entre as m) é ≤ 0,05; a média de R é > 0; ≥ 60% dos pares aplicáveis são positivos; ≥ 100 trades; a média de R é > 0 no estresse.

**Calibração antes do dado real:** os 300 conjuntos sintéticos também medem a taxa de falsa aprovação do critério inteiro (cada conjunto é avaliado contra os outros 299), esperada ≤ 5%. Se passar de 8% (mais de 24 de 300), o critério está descalibrado. A correção é feita **antes** da primeira execução em dado real, é descrita e commitada, e a calibração é refeita. O relatório de calibração é commitado antes de qualquer resultado de descoberta.

## Ordem de execução

1. Este commit: o pré-registro, o registro append-only com a linha de criação (com o SHA-256 deste documento) e o código que impõe as regras.
2. Famílias e controles implementados e testados **só com dado sintético** (steps 13 a 18). O código só toca estrutura de dado real (horários, buracos, spreads) pela versão placebo, que não tem sinal.
3. Calibração nos 300 conjuntos sintéticos, relatório commitado.
4. Descoberta na amostra de descoberta, relatório commitado com a lista das aprovadas.
5. Confirmação na amostra congelada, só das aprovadas, relatório commitado.
6. **Se nada for confirmado: paro e converso com o Igor** (decisão 3 do plano). Não abro nova busca por conta própria.

## Registro e proteção do documento

`docs/forex/fast-registry.jsonl` é append-only. Cada linha é um evento: `registry_created` e `amendment` (com o SHA-256 deste documento; a emenda exige o motivo), `calibration_report`, `discovery_report` (com a lista das aprovadas) e `run` (configuração, amostra, commit do código, resultado). `backend/scripts/research/fx_fast_registry.py` recusa configuração que este documento não declara, rodada de descoberta antes do relatório de calibração ou depois do relatório de descoberta, e rodada de confirmação antes do relatório de descoberta ou de configuração que não foi aprovada. Um teste automático falha se este documento mudar sem uma linha `amendment` com o novo hash.

Correção de defeito: só vale se nasce de um teste que falha mostrando um desvio **deste documento** (por exemplo, olhar o futuro), escrito antes da correção; a rodada é repetida e registrada com o motivo, e não conta como um teste novo. Emenda de texto (erro de digitação, contradição interna) só antes de existir qualquer resultado real e nunca muda hipótese, parâmetro, par, custo, amostra ou critério.

## O que se espera, antes de ver o dado

- **Palpite:** 10% a 20% de chance de ao menos uma configuração ser confirmada. É um palpite, sem medida por trás; fica registrado para vermos depois se estava calibrado.
- F1a e F1b são as mais plausíveis. Se existir efeito de fixing (F2), ele deve ser do tamanho do custo. F3 tem risco de cauda assimétrico (contrariar um movimento que continua). F5 tem cauda quando o regime muda. C1 e C2 devem falhar.
- Os avisos de risco das corretoras que pesquisei (`docs/forex/venue-research-2026-09-19.md`) mostram de 60% a 82% das contas de varejo perdendo dinheiro em CFD: o terreno é hostil.
- Uma reprovação geral é o desfecho mais provável e **não é falha do teste**.

## O que NÃO será feito

- Ajustar parâmetros, horários, pares, custos ou critério depois de ver qualquer resultado real.
- Rodar qualquer estratégia na amostra congelada antes de o relatório de descoberta estar commitado.
- Acrescentar famílias ou variantes na mesma amostra depois de ver resultados: ideia nova exige amostra nova.
- Somar as amostras de descoberta e de confirmação para "resgatar" uma confirmação que falhou. Uma estimativa combinada é só informativa.
- Olhar o resultado por configuração antes de o relatório de calibração estar commitado.

## Viabilidade do canário (informativa)

Com conta de US$ 500 e risco de 0,25% por trade, o menor lote (0,01) só comporta stop de até 12,5 pips no EURUSD (18,75 no USDJPY). Para cada configuração aprovada, o relatório traz a fração dos trades cujo stop cabe nesse limite, para o Igor decidir entre manter 0,25%, subir para 0,5% ou aceitar só as configurações de stop curto. Antecipo que F1, F3 e F5 têm stops bem maiores que 12,5 pips na maior parte dos trades; isso não muda o critério de aprovação, só a conta em que o bot poderia rodar.
