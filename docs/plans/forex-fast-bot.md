# Plano: plataforma de bots de forex de curto prazo

Versão de 2026-09-20, com as respostas do Igor às decisões e o escopo ampliado ("algo mais completo"). Substitui o rascunho do mesmo dia e o plano `forex-rebuild.md` (encerrado no G0: estratégia lenta). As pesquisas de dados M1/tick, corretora/infraestrutura e método de backtest ainda estavam rodando quando este plano foi escrito; os steps 2, 3 e 5 são os que elas podem mudar, e a estimativa é refeita depois do step 3.

## Progresso

- **Step 2 concluído (2026-09-20), com correção no mesmo dia:** o dado do laboratório é o tick bid/ask do HistData convertido em M1 (idêntico à Dukascopy em maio/2024, 1 a 4 horas de download em vez de ~54); a Dukascopy fica como oráculo. A primeira conclusão ("HistData só para desenvolvimento") estava errada: as diferenças eram o fuso, que segue as datas europeias de horário de verão. Ver `docs/forex/data-m1-spike.md`.
- **Step 3 concluído (2026-09-20):** núcleo próprio compilado em numba, 246 milhões de barras/s por núcleo, igual à referência trade a trade; NautilusTrader descartado para a cTrader. Ver `docs/adr/0001-motor-do-bot.md`.
- **Step 6 concluído (2026-09-20):** 10 pares × 60 meses, 18,08 milhões de minutos de M1 bid/ask, validados contra a Dukascopy (majors: 99,87% a 99,89% das horas idênticas; cruzados triangulados a 0,08 a 0,13 pip). Buraco de 5 meses (2023-03 a 07) em todos os pares. Ticks de HistData baixados em 1 a 4 horas em vez das ~54 da Dukascopy. Os buracos de 2023-02 a 2023-07 (as mesmas horas nos 7 pares) se tapam com ~1.750 a 2.500 arquivos diários de M1 da Dukascopy (~2,5 a 4 horas), e o espelho do HistData vale desde 2018-12-16, então a amostra pode se estender até lá. Um segundo cético independente reproduziu o resultado (relógio europeu: 99,84% das horas idênticas; M1 dos ticks igual ao da Dukascopy em 100% de 60.922 minutos).
- **Step 7 concluído (2026-09-20):** modelo de instrumentos (pip, lote, conversão de moeda, tamanho por risco, margem). Com conta de US$ 500 e risco de 0,25%, o menor lote (0,01) limita o stop do EURUSD a **12,5 pips** (18,75 no USDJPY); um stop de 20 pips exigiria risco de 0,4% ou não opera. Decisão futura do Igor: subir o risco por trade para 0,5% ou aceitar só estratégias com stop curto.
- **Step 8 concluído (2026-09-20):** contrato de estratégia compartilhado entre laboratório e bot ao vivo (`app/fx/strategy.py`); mesmas decisões barra a barra e em lote, nenhuma decisão muda quando o futuro muda.
- **Step 9 concluído (2026-09-20):** núcleo do simulador dirigido por estratégia (`app/fx/sim/core.py`), reproduz o núcleo validado trade a trade e roda a 65 milhões de barras/s por núcleo.
- **Step 10 concluído (2026-09-20):** camada de custos do simulador (`app/fx/sim/costs.py`): estresse de spread, slippage por barra que cresce com a volatilidade sem olhar adiante, comissão convertida em pips por par, e swap do rollover das 17h de Nova York (triplo na quarta, zero no fim de semana) calculado de forma vetorizada e igual à implementação de referência em 150 casos aleatórios.
- **Step 11 concluído (2026-09-20):** o núcleo compilado reproduz o simulador em pandas (o do G0) **trade a trade em 10 anos de EURUSD real** (horário, lado, motivo de saída, preços, duração; inclui stops atravessados por gap), a varredura paralela dá 359 milhões de barras/s em 12 núcleos (240 configurações × 3,67 milhões de barras em 2,5 s) e 65 milhões de barras/s em 1 núcleo, contra a meta de 1 milhão. Desvio registrado: em vez de reproduzir os números do G0 refazendo as 6 estratégias antigas, a regressão compara a mecânica do simulador com os mesmos sinais nos dois lados, que é o que precisa continuar igual.
- **Step 12 concluído (2026-09-20):** pré-registro das estratégias rápidas commitado antes de existir código de estratégia (`docs/forex/fast-preregistration.md`): 10 configurações (11 se o calendário do step 5 servir), amostra 2019-01 a 2026-08 (descoberta 2019-01 a 2024-08, confirmação congelada 2024-09 a 2026-08, buraco de 2023-03 a 07 fora), custos fixos (`FUSION_ZERO`, `STRESS`), critério em duas chaves (bootstrap de dias com máximo-t **e** placebo sintético de 300 conjuntos) e "se nada confirmar, paro e converso". O registro de execuções é append-only e o documento fica travado por hash: `backend/scripts/research/fx_fast_registry.py` (28 testes) recusa rodada fora de ordem, configuração não declarada e confirmação antes do relatório de descoberta. Desvios do plano, todos decididos antes de ver dado: (1) a Família 5 virou reversão à média de **razões de dois pares muito correlacionados** (EURGBP real e AUDNZD sintético, com as duas pernas custadas), e não spread com β estimado; (2) a amostra passou de 5 para ~7,7 anos (o espelho do HistData vale desde 2018-12-16); (3) o placebo de embaralhar barras do G0 foi trocado por sinal aleatório do retorno a cada minuto, porque embaralhar barras de M1 destruiria a sazonalidade por horário e não anularia a deriva por horário da família de fixing. Revisão do Igor pendente: se ele discordar de algo, vale emenda registrada enquanto não existir resultado real.
- **Steps 13, 14, 15, 17 e 18 concluídos (2026-09-20):** as famílias F1 (rompimento de sessão, Londres e Nova York), F2 (fluxo do fixing, antes e depois), F3 (reversão de pico, limiares 4 e 6), F5 (reversão de razões de dois pares: EURGBP e AUDNZD sintético, com as duas pernas custadas) e os controles C1 e C2 estão como funções compiladas do contrato do step 8, com a base que elas precisam: relógio de Londres e Nova York com horário de verão de cada um dentro do código compilado, conferido contra o `zoneinfo` em 40 mil instantes aleatórios por cidade e em todas as trocas de 2015 a 2032 (`app/fx/sessions.py`), e agregação de M1 para M5, M15 e H1 igual ao `resample` do pandas com buracos (`app/fx/bars.py`). Cada família é conferida contra uma reimplementação independente da regra do pré-registro em pandas (mesmos sinais, mesmos stops e alvos, atravessando as duas trocas de horário de verão), roda igual barra a barra e em lote, e não muda uma decisão já tomada quando o futuro muda. **Emenda registrada:** a janela de Nova York da F1b tem 20 barras M15, então a exigência de 24 das 32 barras virou 15 das 20 (75%, mesma proporção), corrigida antes de existir qualquer resultado; o registro acusou a mudança do documento e aceitou a emenda com hash novo. Falta o step 16 (F4, notícia), que depende do calendário do step 5.
- **Step 19, parte 1 (2026-09-20):** o laboratório roda as 10 configurações pré-registradas por par e por trecho contínuo, com custo base e de estresse, monta as tabelas diárias e aplica o critério inteiro (chave A: bootstrap de dias com máximo-t; chave B: placebo com sinal aleatório por minuto; regras de aprovação e de "inconclusivo"), grava no registro na ordem obrigatória e recusa rodar com código não commitado (`scripts/research/fx_fast_lab.py`, `fx_fast_stats.py`, `fx_fast_placebo.py`, `fx_fast_manifest.py`; testes de ponta a ponta com dado sintético). Falta rodar em dado real: depende de terminar o download de 2019 a 2021 dos 3 cruzados (a fonte passou a limitar o ritmo), gerar o manifesto de meses limpos e commitá-lo, e só então a calibração (300 conjuntos), a descoberta e a confirmação.
- **Steps 22, 23 e 24, parte contra venue falso (2026-09-20):** núcleo do runner independente da corretora em `app/fx/runner/`: interface mínima de venue, venue falso com as falhas que o runner precisa sobreviver, executor com lote por risco (recusa se o menor lote já arrisca mais que o orçamento), stop no servidor conferido na volta, reenvio sem duplicar ordem (mesmo `client_order_id`), trava de risco (teto de perda diária de 1% por dia FX, teto de posições e de spread, kill-switch gravado em disco que sobrevive a reinício, dia e equity inicial persistidos) e reconciliação no boot (adota órfã com stop, fecha órfã sem stop, recupera stop perdido, registra o que o servidor fechou com o bot desligado). 16 testes. Falta o adaptador real da corretora (step 4, depende do Igor), o feed de barras ao vivo (step 21) e a bateria contra a demo.
- **Step 21, parte independente da corretora (2026-09-20):** construtor de barras ao vivo a partir de cotações (`app/fx/runner/feed.py`), igual à agregação de minutos do laboratório em M1, M5, M15 e H1 (buraco continua buraco), e o laço do bot (`runner/bot.py`) que alimenta a mesma função compilada do laboratório e transforma a decisão em ordem protegida ou fechamento. Falta a fonte real de cotações (adaptador da corretora, step 4) e a conferência contra as barras da corretora na demo.
- **Step 28 concluído em SQLite, falta Postgres (2026-09-20):** migração aditiva `019_fx_bots` com 4 tabelas (bots, trades, snapshots de equity, eventos), sem coluna de credencial, com rollback `alembic downgrade 018`; os testes fazem upgrade e downgrade num banco que já tem dado de cripto e conferem modelos contra colunas migradas e as restrições do banco. **Não foi rodada em Postgres** (não há Postgres local e o Docker desta máquina nega permissão): rodar `upgrade` e `downgrade` num Postgres de desenvolvimento antes de qualquer deploy.
- **Aguardando o Igor:** step 1 (conta demo cTrader Zero e credenciais da Open API). O Igor está abrindo a demo (conta tipo **Zero**, cTrader, USD, US$ 500).

## Visão geral

Construir, no mesmo repositório, um núcleo novo para bots de forex de minutos a horas (24 horas nos dias de mercado): um laboratório rápido que testa famílias de estratégia no passado sem se enganar, um runner que opera numa corretora com API, e uma plataforma com lista de bots, detalhe por bot, gráficos ao vivo, tela do laboratório e botões de ligar, parar e kill-switch. O primeiro dinheiro real é um canário de encanamento de risco mínimo; quais estratégias entram vem do laboratório, não do desejo de operar.

## Respostas do Igor (2026-09-20)

1. Canário: conta de US$ 500, risco por trade de 0,25% (~US$ 1,25) e perda diária máxima de 1% (~US$ 5). **Confirmado.**
2. Corretora: cTrader Open API (Fusion Markets ou equivalente). **Confirmado.**
3. Regra de parada: **não encerramos, conversamos.** Se as famílias fixadas no pré-registro não confirmarem em amostra nunca vista, eu paro e trago os números para decidirmos juntos (ampliar amostra, novas hipóteses em amostra nova, outro mercado). Não abrimos nova busca por conta própria, porque cada tentativa extra aumenta o risco de falso positivo.
4. Parecer jurídico e contábil: **o Igor informou que já existe.** A trava da remessa (antes: "sem parecer não avança") sai do plano. Guardar a cópia fora do repositório (o git não deve conter documento com dados pessoais).
5. Escopo: **mais completo.** Premissa que estou usando, a corrigir se estiver errada: 7 majors mais 3 cruzados líquidos (EURJPY, GBPJPY, EURGBP); 5 famílias de estratégia (rompimento de sessão, fixing, reversão de pico, deriva pós-notícia, pares correlacionados) mais 2 controles negativos; 5 anos de dados de 1 minuto (3 de descoberta, 2 de confirmação, estendendo para mais se a fonte permitir); 1 corretora agora e uma interface neutra para plugar uma segunda depois do canário; plataforma completa (lista, detalhe por bot, gráfico ao vivo, tela do laboratório, alertas).

## A verdade sobre "real hoje"

Um bot operando com dinheiro real hoje não é possível: não existe bot (nada envia ordem a uma corretora de forex), não existe conta (aprovação em dias, remessa em 1 a 3 dias úteis), nenhuma estratégia rápida foi testada e 63% a 80% das contas de varejo perdem dinheiro. Hoje dá para abrir as contas (o que mais atrasa) e rodar os spikes. O primeiro dinheiro real é o canário (step 27): 1 a 2 semanas, dominadas por aprovação de conta e remessa, e não por código.

## O que a pesquisa mostrou

- Estratégia de **segundos** não tem evidência de lucro para varejo: o custo é de 1 a 2,4 pips por trade e um movimento de 1 minuto no EURUSD tem ~1,7 pip. Arbitragem triangular, de latência e market making são inviáveis. A vantagem do bot é cobertura 24/5 e disciplina, não velocidade.
- O que vale testar é movimento de **10 a 30 pips em minutos a horas**, onde o custo é 5 a 15% do alvo. Reversão em M1/M5 e momentum de segundos entram só como controle negativo.
- Os dados atuais (velas de 1 hora) não testam isso: precisamos de M1 (ou tick) com bid e ask.
- **Pesquisas de corretora e método (2026-09-20):** minutos a horas é viável e segundos não (no feed medido, o EURUSD anda ≥ 1 pip em só 2,3% das janelas de 10 s, contra 0,5-1,0 pip de custo); todas as corretoras de CFD são contraparte e as cláusulas de "scalping/algoritmo" (Fusion, Pepperstone, Capital.com) dão poder de anular ordens, e só a Dukascopy diz por escrito que robôs e notícia são permitidos; o cliente Python oficial da cTrader (OpenApiPy) está parado desde 2024 e fixa versões antigas, então o cliente será nosso, em asyncio, sobre o `.proto` oficial; a aprovação do aplicativo no portal da Spotware é manual e sem prazo, então o step 1 é urgente; o risco central do laboratório é mineração de dados (com 1.000 variantes e 3 anos, o melhor Sharpe de estratégias sem vantagem nenhuma é ~1,9), então cada variante testada entra num registro append-only e o critério inclui Sharpe deflacionado e teste de superioridade; vale incluir M15 e H1, onde o custo é 15% a 45% do desvio-padrão do movimento (em M1 é 65% a 200%).
- Só ~14% do código atual serve (laboratório FX, login e TOTP, esqueleto das telas com gráfico TradingView, banco e deploy, a disciplina do gate). O motor atual reage só a vela fechada e faz ~15 consultas ao banco por decisão; o núcleo do bot é novo.

## Decisões técnicas já tomadas

- Python com asyncio no runner: a latência que importa é a da corretora, e estratégia de minutos não precisa de mais. Sem Redis no caminho de ordem (fila em processo). Journal em lote no Postgres, fora do caminho da decisão; dados de mercado em Parquet.
- **Simulador do laboratório compilado (numba), meta ≥ 1 milhão de barras por segundo.** O simulador atual faz ~7 mil por segundo. A conta: 10 pares × 5 anos × ~370 mil barras M1 por par-ano × 7 configurações = ~130 milhões de passos por rodada, e o placebo pede centenas de rodadas. Com a meta acima, uma rodada leva ~2 minutos e o placebo cabe em horas usando os 12 núcleos.
- O runner roda no Railway no início: latência de centenas de ms é irrelevante em minutos a horas. VPS perto da corretora só se algum dia entrar estratégia de segundos.
- Credenciais da corretora ficam só no runner, cifradas.
- Adotar um motor de código aberto (por exemplo NautilusTrader) ou construir o núcleo próprio: decisão por evidência, no step 3.
- Modelo de instrumento próprio (pip, lote, conversão para a moeda da conta), porque os cruzados exigem conversão que o `fx_costs` atual recusa de propósito.

## Plano de implementação

O `main` só recebe merge quando o Igor pedir; todo o trabalho fica na branch `feat/forex-rebuild`, com a suíte verde a cada step. Onde diz "esforço", é dev; passos do Igor estão marcados.

### Fase 0: incerteza primeiro (spikes)

**Step 1: Contas e credenciais [Igor]** (~1-2h, mais dias de espera)
  - O que: abrir a conta demo do cTrader (cTID), registrar um aplicativo no portal da Spotware para obter as credenciais da Open API, iniciar o KYC real e mandar as perguntas por escrito · Arquivos: `docs/forex/step5-contas-checklist.md` (já existe)
  - Depende de: nada · Paralelo com: 2, 3, 5, 7, 28
  - Teste: credenciais de demo funcionando; resposta por escrito da corretora sobre residentes no Brasil guardada

**Step 2: Spike de dados M1** (~4-8h)
  - O que: baixar 1 mês de M1 bid/ask do EURUSD da Dukascopy, validar a qualidade, medir o ritmo real e estimar o tempo para 10 pares × 5 anos; se inviável, a fonte alternativa (HistData/TrueFX) · Arquivos: `backend/scripts/research/fx_dataset_m1.py` (novo) + teste
  - Depende de: nada · Paralelo com: 1, 4, 5, 7, 28
  - Teste: o mês passa no controle de qualidade (densidade 5/7, nenhum candle de sábado, spread plausível, sem cotação cruzada) e o relatório traz arquivos/min e ETA. Pior caso: limite de taxa do servidor (a Dukascopy já devolve 503) e formato diferente do H1 (dobrado por ser terceiro)

**Step 3: Spike do motor: adotar pronto ou próprio, e desempenho** (~5-9h)
  - O que: rodar 1 mês de M1 com bid/ask e o mesmo custo (a) num motor aberto (NautilusTrader ou o que a pesquisa indicar) e (b) num núcleo próprio em numba; comparar com um cálculo independente, medir barras por segundo e checar se há adaptador ao vivo para a corretora · Arquivos: `backend/scripts/research/engine_spike/` (novo), `docs/adr/0001-motor-do-bot.md`
  - Depende de: 2 · Paralelo com: 4, 5, 6, 7
  - Teste: o P&L bate com o cálculo independente (< 0,01 pip por trade), o throughput é medido contra a meta de 1 milhão de barras por segundo e a decisão fica registrada. Pior caso: numba ou motor aberto sem suporte ao Python 3.13. **A estimativa dos steps 8-11 e 21-24 é refeita depois dele.**

**Step 4: Spike da API cTrader na demo** (~6-10h)
  - O que: avaliar **dois caminhos** e ficar com o que funcionar: (a) cliente próprio em asyncio sobre o `.proto` oficial da Open API (o SDK Python oficial está parado desde 2024 e fixa versões antigas), e (b) o **cTrader Console oficial** (imagem Docker `ghcr.io/spotware/ctrader-console`, headless em Linux, comandos de ordem, posição e preço, **sem aplicativo da Open API**, o que evita a aprovação manual da Spotware). Para cada um: conectar, receber cotações, abrir e fechar ordem com SL/TP, derrubar o cliente e conferir que o SL continua no servidor, reconectar · Arquivos: `backend/scripts/research/ctrader_probe.py` (novo)
  - Depende de: 1 · Paralelo com: 2, 3, 5, 6, 7
  - Teste: log mostrando posição e SL vivos depois de matar o processo. Integração externa: pior caso dobra (aprovação do aplicativo no portal)

**Step 5: Spike do calendário econômico histórico** (~4-8h)
  - O que: achar uma fonte com horário, previsão e valor divulgado de indicadores de alto impacto para 5 anos, verificar cobertura e termos · Arquivos: `backend/scripts/research/fx_calendar_probe.py` (novo), `docs/forex/calendar-source.md`
  - Depende de: nada · Paralelo com: 1, 2, 3, 4
  - Teste: 5 anos de eventos de alto impacto (NFP, CPI, decisões de juros) carregados, com horários conferidos contra 3 eventos conhecidos. Pior caso: não existir fonte gratuita boa; então a família de deriva pós-notícia (step 16) sai do escopo e conversamos

### Fase 1: laboratório (descobrir se existe estratégia que paga)

**Step 6: Ticks do HistData para M1 bid/ask, 5 anos de 10 pares** (~6-10h de esforço, 1 a 4 horas de download)
  - O que: baixar os ticks mensais, resolver o fuso por arquivo (regra europeia de verão, testada contra o H1 da Dukascopy que já temos), gerar M1 bid/ask, marcar buracos e tapá-los com a Dukascopy, validar cada par-mês contra o oráculo · Arquivos: `backend/scripts/research/fx_histdata_ticks.py` (novo) + teste
  - Depende de: 2 · Paralelo com: 3, 4, 5, 7
  - Teste: cada par-mês tem ≥ 99% das horas não planas idênticas ao H1 da Dukascopy e nenhuma cotação cruzada; relatório de buracos por par. Pior caso: fuso diferente em algum ano, buracos grandes em 2023 e ticks sem milissegundos em 2026 (dobrado por ser terceiro)

**Step 7: Modelo de instrumento e conversão de moeda** (~5-8h)
  - O que: pip, lote, tamanho de contrato e conversão do P&L para a moeda da conta, inclusive cruzados (EURJPY, GBPJPY, EURGBP) · Arquivos: `backend/app/fx/instruments.py` (novo) + teste
  - Depende de: nada · Paralelo com: 1 a 6
  - Teste: testes de propriedade (conversão de ida e volta, pip value de JPY e de cruzados batendo com a tabela da corretora)

**Step 8: Interface de estratégia em streaming** (~3-5h)
  - O que: contrato `on_bar`/`on_quote` com estado incremental e pedidos de ordem, o mesmo código no simulador e no bot vivo · Arquivos: `backend/app/fx/strategy.py` (novo) + teste
  - Depende de: 3 · Paralelo com: 6, 7, 4
  - Teste: uma estratégia de exemplo gera os mesmos sinais alimentada barra a barra e em lote (prova de que não olha adiante)

**Step 9: Simulador, núcleo em numba** (~5-8h)
  - O que: preencher a mercado no bid/ask, stop e alvo com a semântica do `fx_gap`, comissão do `fx_costs`, posição líquida · Arquivos: `backend/app/fx/sim/core.py` (novo) + teste
  - Depende de: 8, 6, 7 · Paralelo com: 12
  - Teste: reproduz, trade a trade, o simulador atual (`fx_spike`) nos dados H1, que é a regressão

**Step 10: Simulador, custo por hora, rollover, latência e swap** (~4-6h)
  - O que: spread por hora do dia, alargamento no rollover (21-23 UTC), latência de ordem, slippage e swap no rollover das 17h de Nova York · Arquivos: `backend/app/fx/sim/costs.py` + teste
  - Depende de: 9 · Paralelo com: 12, 13
  - Teste: a mesma ordem custa mais no rollover do que às 14h UTC; a latência atrasa o preenchimento e muda o preço

**Step 11: Simulador, regressão e benchmark** (~3-5h)
  - O que: provar que o resultado não mudou e que a meta de velocidade foi atingida · Arquivos: `backend/tests/unit/test_fx_sim_regression.py`, `backend/scripts/research/fx_sim_bench.py`
  - Depende de: 10
  - Teste: os relatórios do G0 e da confirmação de estratégia lenta reproduzem os mesmos números e o benchmark passa de 1 milhão de barras por segundo em 1 núcleo

**Step 12: Pré-registro** (~2-4h)
  - O que: fixar em commit, antes de rodar qualquer estratégia, as famílias, as regras, o custo por hora, a divisão descoberta/confirmação, o critério calibrado por placebo e o que acontece se nada confirmar (parar e conversar) · Arquivos: `docs/forex/fast-preregistration.md` (novo)
  - Depende de: 6, 8 · Paralelo com: 9, 10, 11
  - Teste: commit datado anterior a qualquer resultado; revisado pelo Igor

**Step 13: Família 1, rompimento de abertura de sessão** (~3-5h) · `backend/app/fx/strategies/session_breakout.py` · Depende de: 12 · Paralelo com: 14, 15, 16, 17, 18 · Teste: sinal correto em sequência sintética e teste de não olhar adiante

**Step 14: Família 2, fluxo de fixing** (~3-5h) · `strategies/fixing_flow.py` · Depende de: 12 · Paralelo com: 13, 15, 16, 17, 18 · Teste: entra e sai só nas janelas fixadas, com horário de verão

**Step 15: Família 3, reversão depois de pico de volatilidade** (~3-5h) · `strategies/spike_fade.py` · Depende de: 12 · Paralelo com: 13, 14, 16, 17, 18 · Teste: stop duro sempre anexado; nunca entra em pico sem o spread normalizado

**Step 16: Família 4, deriva pós-notícia** (~4-6h) · `strategies/news_drift.py` · Depende de: 12, 5 · Paralelo com: 13, 14, 15, 17, 18 · Teste: nunca entra no segundo da divulgação; espera o spread normalizar; usa só horário e valor conhecidos no momento

**Step 17: Família 5, pares correlacionados** (~4-6h) · `strategies/pairs_spread.py` · Depende de: 12, 7 · Paralelo com: 13 a 16, 18 · Teste: no AUDNZD sintético as duas pernas entram e saem juntas e o custo das duas pernas é contado; o EURGBP opera como cruzado direto, com stop e alvo fixados no preço da entrada (ver o pré-registro, F5a e F5b)

**Step 18: Controles negativos** (~2-3h) · `strategies/controls.py` (reversão em M1/M5 e momentum de segundos; esperamos falha) · Depende de: 12 · Paralelo com: 13 a 17 · Teste: rodam no simulador; se algum "passar" no gate, o resultado é tratado como suspeita de defeito e não como achado

**Step 19: Laboratório, descoberta em paralelo com placebo** (~5-8h)
  - O que: rodar as famílias na amostra de descoberta, com ≥ 200 embaralhamentos do placebo distribuídos pelos núcleos, poder e sensibilidade a custo · Arquivos: `backend/scripts/research/fx_fast_lab.py` (novo), relatório em `docs/forex/`
  - Depende de: 11, 13, 14, 15, 16, 17, 18 · Paralelo com: 21
  - Teste: relatório com valor-p ajustado pelo número real de configurações, efeito mínimo detectável e custo por hora do dia; o placebo dos controles não aprova ruído acima de 5%

**Step 20: Confirmação em amostra nunca vista, Portão G1** (~3-5h)
  - O que: aplicar, sem mudar nada, as famílias que sobreviverem à descoberta · Arquivos: `docs/forex/fast-confirmation-report.md` (novo)
  - Depende de: 19
  - Teste: resultado conforme o pré-registro. **G1:** só o que confirmar vira estratégia do canário. **Se nada confirmar: paro, trago os números e conversamos.** O canário nesse caso roda só o encanamento

### Fase 2: runner e canário real

**Step 21: Runner, feed e barras ao vivo** (~4-6h) · `backend/app/fx/runner/feed.py` · Depende de: 4, 8 · Paralelo com: 9 a 20 · Teste: monta barras M1 em tempo real na demo e confere com as barras históricas da corretora

**Step 22: Runner, executor** (~5-8h) · `runner/executor.py` · Depende de: 21 · Teste: contra um venue falso e depois contra a demo: ordem com SL/TP anexado no servidor e lote calculado por risco

**Step 23: Runner, risco e kill-switch** (~4-6h) · `runner/risk.py` · Depende de: 22 · Teste: o teto de perda diária dispara e para o bot; o kill-switch fecha tudo; o dead-man switch desliga se o processo travar

**Step 24: Runner, reconciliação no boot** (~4-6h) · `runner/reconcile.py` · Depende de: 23 · Teste: matar o processo com posição aberta e reiniciar reconcilia sem duplicar ordem nem perder o SL

**Step 25: Soak em demo** (~3-5h de esforço, 3 a 5 dias de calendário)
  - O que: rodar uma estratégia trivial de encanamento (entra, sai depois de N minutos, com SL) e comparar custo, latência e slippage medidos contra o simulado · Arquivos: `backend/scripts/research/fx_soak_report.py` (novo)
  - Depende de: 24
  - Teste: relatório com custo medido dentro de 30% do simulado e zero ordem órfã

**Step 26: Remessa e conta real [Igor]** (~3-5h, mais dias) · Depende de: 1 (o parecer já existe, informado pelo Igor) · Teste: conta financiada; IOF e spread cambial anotados

**Step 27: Canário real** (~3-5h de esforço, 1 a 2 semanas de calendário)
  - O que: lote mínimo e o teto de perda da Decisão 1, com a estratégia confirmada no G1 ou o encanamento trivial · Arquivos: `backend/app/fx/runner/config.py` (novo)
  - Depende de: 25, 26
  - Teste: primeira ordem real vista na corretora, relatório diário, nenhum estouro do teto. **Critério: medir custo e slippage reais e provar o encanamento. Não é lucro.**

### Fase 3: plataforma completa

**Step 28: Modelo de dados por bot (migração aditiva)** (~4-6h, rollback escrito)
  - O que: tabelas de bots, execuções, ordens, fills e equity por bot, só colunas novas · Arquivos: `backend/alembic/versions/019_fx_bots.py`, `backend/app/models/fx_bot.py` · Depende de: nada · Paralelo com: 1 a 7
  - Teste: `upgrade` e `downgrade` limpos e dados de cripto intactos. Rollback: `alembic downgrade 018`

**Step 29: Journal em lote e API de leitura** (~5-8h) · `runner/journal.py`, `backend/app/api/v1/fx_bots.py` · Depende de: 28, 22 · Teste: 1 bot em demo gera linhas e o endpoint devolve o retorno por bot

**Step 30: Tela, lista de bots** (~4-6h) · `frontend/src/app/bots/page.tsx` · Depende de: 29 · Teste: Playwright cria a lista e mostra retorno, tempo e status por bot

**Step 31: Tela, detalhe do bot** (~5-8h) · `frontend/src/app/bots/[id]/page.tsx` · Depende de: 30 · Teste: Playwright vê operações, retorno, tempo em posição, drawdown e os pontos bons e ruins do bot

**Step 32: Tela, gráfico ao vivo com marcadores** (~5-8h) · `frontend/src/components/charts/` (lightweight-charts) + push do runner · Depende de: 29, 21 · Teste: Playwright vê o preço se mexendo e as entradas e saídas sobre o gráfico

**Step 33: Tela, laboratório** (~5-8h) · `frontend/src/app/lab/page.tsx` · Depende de: 19, 30 · Teste: Playwright compara famílias com valor-p, custo por hora e amostra, e mostra a de confirmação separada

**Step 34: Ligar e parar pela tela, credenciais seguras** (~4-6h) · Depende de: 29, 23 · Teste: o botão liga o runner na demo; as credenciais nunca aparecem em log nem na API

**Step 35: Kill-switch, TOTP e guard persistente** (~4-6h) · Depende de: 34 · Teste: kill fecha posições; reiniciar o runner mantém o estado do guard; ligar exige TOTP

**Step 36: Alertas por WhatsApp (Evolution API)** (~4-8h) · Depende de: 29 · Teste: desligar o runner dispara a mensagem. Integração externa, pior caso dobra

**Step 37: Vários bots, orçamento de risco por bot** (~3-5h) · Depende de: 23, 28 · Teste: um bot não consegue gastar o orçamento de outro

**Step 38: Vários bots, host de N bots** (~3-5h) · Depende de: 37 · Teste: 5 bots no mesmo par sem conflito e P&L atribuído ao bot certo

### Fase 4: depois do canário

**Step 39: Interface neutra de corretora e segunda corretora (IBKR)** (~8-14h) · `backend/app/fx/venues/` · Depende de: 27 · Teste: o mesmo bot roda na demo das duas corretoras. Integração externa, o pior caso dobra (login com 2FA semanal, gateway)

**Step 40: Congelar e remover o código de cripto** (~4-8h, rollback: `git tag pre-crypto-removal`) · Depende de: 27 · Teste: suíte verde sem os módulos de cripto

## Riscos

- Nenhuma estratégia rápida ser lucrativa, o desfecho mais provável segundo a pesquisa → o laboratório vem antes do dinheiro; se nada confirmar, paro e conversamos (Decisão 3).
- O canário ser lido como "fui operar de verdade" → é teste de encanamento com perda esperada e teto obrigatório.
- Corretora de varejo como contraparte (last look, rejeição por preço velho, conta pode ser encerrada por fluxo de latência) → nada de estratégia de latência; a demo e o canário medem a diferença entre o simulado e o real.
- Dado agregado da Dukascopy não é o preço executável da corretora → o step 25 mede a diferença antes de qualquer estratégia entrar no canário.
- Escopo grande: mais famílias e pares aumentam a chance de falso positivo → o critério é a melhor configuração contra o placebo, com o número real de configurações.
- Simulador lento inviabiliza o placebo → meta de 1 milhão de barras por segundo medida no step 3 e exigida no step 11.
- Download de dados demorado → o servidor limita o ritmo; o step 2 mede e propõe a fonte alternativa antes do step 6.
- Cauda: gap e alargamento de spread (SNB 2015, libra 2016) → stop no servidor da corretora, tamanho mínimo, flat na sexta.
- Estimativa: a base já errou 2 a 3× em planos anteriores → usar o teto +50% para calendário.

## Total

**152-251h de desenvolvimento** (38 steps de código; mais 4-7h do Igor nos steps 1 e 26) e semanas de calendário por causa de aprovação da conta, remessa, download de dados e soak. Estimativa do avaliador de código para um escopo ainda mais amplo, sem ajuda de IA, dev sênior: laboratório rápido 25-39 dias e runner com corretora em demo 39-61 dias.

**Caminho crítico até o primeiro dinheiro real:** 1 → 4 → 21 → 22 → 23 → 24 → 25 → 26 → 27, com o step 3 → 8 em paralelo com o 4 (31 a 50h de código, 1 a 2 semanas de calendário). **Caminho crítico até saber se existe estratégia que paga:** 2 → 6 → 9 → 10 → 11 → 19 → 20 (com 12 a 18 em paralelo).
