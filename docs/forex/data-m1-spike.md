# Step 2: spike de dados de 1 minuto (M1) com bid e ask

Medido em 2026-09-20. Código: `backend/scripts/research/fx_dataset_m1.py` (Dukascopy) e `backend/scripts/research/fx_histdata_check.py` (comparação com o HistData). Testes: `backend/tests/unit/test_fx_dataset_m1.py`.

## Resultado

**Decisão:** o dado canônico do laboratório é o M1 da **Dukascopy com bid e ask reais**. O download é lento por limite do servidor (~54 horas para 10 pares × 5 anos com 8 conexões), mas roda em segundo plano e não custa esforço. O **HistData** é 1.000× mais rápido e serve para desenvolver e medir desempenho, **não para decidir**: ele só traz bid e difere da Dukascopy justamente nos movimentos bruscos.

## Dukascopy M1 (bid e ask reais)

- Um arquivo por dia e lado (`.../SYMBOL/YYYY/MM/DD/BID_candles_min_1.bi5`), ~10 KB cada. O mês na URL começa em zero e o dia em um. Cada arquivo tem 1.440 registros (um por minuto); minutos com o mercado fechado vêm como candles planos de volume zero, que o carregador descarta.
- **Qualidade (EURUSD, maio/2024, 32.658 minutos ativos):** nenhuma cotação cruzada (ask abaixo do bid), nenhum minuto no sábado, nenhuma pausa maior que 30 minutos dentro da semana, spread mediano **0,2 pip** (p95 = 0,4; p99 = 2,1), só 0,41% dos minutos com spread acima de 3 pips.
- **Ritmo medido: 6,7 arquivos por minuto com 4 conexões e 9,7 com 8** (+45%). O servidor limita o ritmo (na rodada de H1, com 7 processos simultâneos, devolveu 503 em rajada), então mais conexões ajudam pouco e o resultado é uma rampa suave, não linear.
- **Projeção:** 10 pares × 5 anos = 31.286 arquivos = **54 horas** com 8 conexões (78 horas com 4). Os 3 pares prioritários (EURUSD, GBPUSD, USDJPY) levam ~16 horas.
- **Andamento:** o download dos 10 pares (2021-09-01 a 2026-08-31) foi iniciado em 2026-09-20 às 10:20 (reiniciado com 8 conexões às 10:30), em sequência, com retomada automática pelo cache em disco. Exige a máquina ligada.

## HistData M1 (só bid)

- **Velocidade:** um ano de M1 de um par em ~6 segundos; 10 anos do EURUSD (3,67 milhões de minutos) em 69 segundos.
- **Fuso:** o site diz "EST fixo" (UTC−5), mas os dados seguem o **horário de Nova York com horário de verão**. Lidos como EST fixo, o erro mediano no verão é de 4,3 pips; lidos como Nova York, é 0,000.
- **Comparação com a Dukascopy (EURUSD, 2016-2025, 57.256 horas):** o fechamento de cada hora é idêntico na maioria (mediana 0,000 pip; em maio/2024, todos os 32.658 minutos batem), mas **22,7% das horas diferem mais de 0,05 pip, 3,9% mais de 2 pips, 1,3% mais de 10 pips**, com diferenças de até ~90 pips nos dias mais voláteis (março/2020). As horas que diferem passam de ~20 por ano (2016-2018) para ~300 por ano (2019-2025). Não achei a causa.
- **Por que isso impede o uso para decisão:** as estratégias que vamos testar (rompimento de sessão, reversão de pico, deriva pós-notícia) operam justamente nos minutos bruscos, onde as duas fontes mais divergem. E o HistData não tem o ask, então o spread real teria de ser inferido.
- **Uso permitido:** desenvolvimento do simulador, testes de desempenho e regressão, sempre rotulado como HistData.

## Limitações e riscos

- O limite do servidor da Dukascopy é o gargalo de calendário; não há como acelerar sem outra fonte com bid e ask.
- A máquina precisa ficar ligada por ~2 dias para o conjunto completo.
- Os termos de uso da Dukascopy e do HistData valem para pesquisa pessoal; conferir antes de qualquer outro uso.
- O feed agregado da Dukascopy não é o preço executável de cada corretora; o soak em demo (step 25) mede essa diferença.

## Teste do step

- O mês de teste passou no controle de qualidade e o relatório traz o ritmo e a projeção (acima). **Step 2 concluído.**
