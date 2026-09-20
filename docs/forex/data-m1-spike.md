# Step 2: spike de dados de 1 minuto (M1) com bid e ask

Medido em 2026-09-20. Código: `backend/scripts/research/fx_dataset_m1.py` (Dukascopy) e `backend/scripts/research/fx_histdata_check.py` (comparação com o HistData). Testes: `backend/tests/unit/test_fx_dataset_m1.py`.

## Resultado

**Decisão (revista no mesmo dia):** o dado do laboratório é o **tick bid/ask do HistData convertido em M1**, validado contra a Dukascopy. A primeira versão deste relatório dizia o contrário ("HistData só para desenvolvimento, difere nos movimentos bruscos"); isso estava errado e foi corrigido depois de uma pesquisa independente apontar a causa: as diferenças eram um deslocamento de 1 hora do fuso, ver a seção do HistData. A Dukascopy fica como oráculo de validação e para tapar buracos, porque o servidor limita o ritmo (~54 horas para 10 pares × 5 anos em M1) e os termos dela proíbem bot.

## Dukascopy M1 (bid e ask reais)

- Um arquivo por dia e lado (`.../SYMBOL/YYYY/MM/DD/BID_candles_min_1.bi5`), ~10 KB cada. O mês na URL começa em zero e o dia em um. Cada arquivo tem 1.440 registros (um por minuto); minutos com o mercado fechado vêm como candles planos de volume zero, que o carregador descarta.
- **Qualidade (EURUSD, maio/2024, 32.658 minutos ativos):** nenhuma cotação cruzada (ask abaixo do bid), nenhum minuto no sábado, nenhuma pausa maior que 30 minutos dentro da semana, spread mediano **0,2 pip** (p95 = 0,4; p99 = 2,1), só 0,41% dos minutos com spread acima de 3 pips.
- **Ritmo medido: 6,7 arquivos por minuto com 4 conexões e 9,7 com 8** (+45%). O servidor limita o ritmo (na rodada de H1, com 7 processos simultâneos, devolveu 503 em rajada), então mais conexões ajudam pouco e o resultado é uma rampa suave, não linear.
- **Projeção:** 10 pares × 5 anos = 31.286 arquivos = **54 horas** com 8 conexões (78 horas com 4). Os 3 pares prioritários (EURUSD, GBPUSD, USDJPY) levam ~16 horas.
- **Andamento:** o download dos 10 pares (2021-09-01 a 2026-08-31) foi iniciado em 2026-09-20 às 10:20 (reiniciado com 8 conexões às 10:22), em sequência, com retomada automática pelo cache em disco. Exige a máquina ligada.

## HistData (M1 só bid, e ticks com bid e ask)

- **Velocidade:** um ano de M1 de um par em ~6 segundos (10 anos do EURUSD, 3,67 milhões de minutos, em 69 segundos); **um mês de ticks bid/ask em ~10 segundos** (~1,2 milhão de ticks por mês de EURUSD).
- **Fuso, a causa das diferenças:** o relógio do HistData é o horário de Nova York (UTC−5 no inverno, UTC−4 no verão), **mas a troca de horário segue as datas europeias** (último domingo de março e de outubro) em 2019-2025, e não as americanas. Nas ~5 semanas por ano em que os dois calendários divergem, ler o horário com a regra americana desloca tudo 1 hora. Medido com o fechamento horário do EURUSD contra a Dukascopy: regra americana, 5,6% a 7,7% das horas diferem mais de 0,05 pip em 2019-2025; **regra europeia, 0,0% a 0,3%**. Em 2016-2018 nenhuma das duas regras funciona (66% a 81% de diferença); esses anos ficam de fora, e não precisamos deles (usamos 2021 em diante).
- **Validação em ticks (EURUSD, maio/2024):** o M1 bid/ask derivado dos ticks do HistData é **idêntico** ao M1 real da Dukascopy: 0,000% dos minutos diferem mais de 0,05 pip em abertura, máxima, mínima e fechamento, dos dois lados; spread mediano igual (0,20 pip), correlação 1,000. Os 238 minutos que só a Dukascopy tem são as 4 primeiras horas do dia 1, que no fuso de Nova York pertencem ao arquivo do mês anterior.
- **Armadilhas registradas por uma pesquisa independente** (a confirmar durante o download): buracos entre 2023-02 e 2023-07 (até ~30% das horas), a partir de 2026-06-28 os ticks vêm sem milissegundos, e 2017 vem de outra fonte. Os buracos se tapam com a Dukascopy.
- **O que o HistData não resolve:** são cotações top-of-book de uma casa, sem latência, slippage nem last look; o soak em demo (step 25) mede a diferença para a corretora real.

## Limitações e riscos

- Termos de uso: os da Dukascopy proíbem bot e mineração de dados (por isso paramos o download em massa depois de validar a alternativa); confirmar os do HistData antes de qualquer uso além de pesquisa pessoal.
- Buracos conhecidos no HistData (2023-02 a 2023-07) e a perda de milissegundos desde 2026-06-28: tratados no step 6.
- Os termos de uso da Dukascopy e do HistData valem para pesquisa pessoal; conferir antes de qualquer outro uso.
- O feed agregado da Dukascopy não é o preço executável de cada corretora; o soak em demo (step 25) mede essa diferença.

## Teste do step

- O mês de teste passou no controle de qualidade e o relatório traz o ritmo e a projeção (acima). **Step 2 concluído.**

## Resultado do step 6 para o EURUSD (2021-09 a 2026-08)

`backend/scripts/research/fx_histdata_ticks.py` baixa os ticks mensais, resolve o fuso pela regra europeia, monta o M1 bid/ask e valida cada mês contra o H1 da Dukascopy que o projeto já tem.

- **1.811.388 minutos em 60 meses**, 16 minutos de execução (2 conexões), **59 dos 60 meses passam** o critério (≥ 99% das horas com fechamento idêntico ao oráculo, nos dois lados, e nenhuma cotação cruzada).
- **Concordância média: 99,88% (bid) e 99,89% (ask)**; mínimo mensal 98,4% (2023-09, 8 horas de 501 com 1 a 4 pips de diferença, espalhadas: parecem ticks ausentes no HistData, não erro de fuso).
- **Buracos confirmados:** **2023-03 a 2023-07** têm 20.173 a 23.354 minutos (o normal é ~31.000), até 31% das horas do oráculo sem dado. Esses 5 meses (8% da amostra) são tratados como buraco: ficam fora das análises ou são preenchidos com a Dukascopy.
- Os outros 9 pares (GBPUSD, USDJPY, AUDUSD, NZDUSD, USDCAD, USDCHF, EURJPY, GBPJPY, EURGBP) estão sendo processados em segundo plano. Os três cruzados não têm H1 de oráculo; serão validados por triangulação com os majors.

