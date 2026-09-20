# Plano: plataforma de bots de forex de curto prazo

Versão formal de 2026-09-20. Substitui o rascunho do mesmo dia e o plano `forex-rebuild.md` (encerrado no G0: estratégia lenta). **Aguarda o OK do Igor.** As pesquisas de dados M1/tick, corretora/infraestrutura e método de backtest ainda estavam rodando quando este plano foi escrito; os steps 2 e 3 são exatamente os que elas podem mudar, e a estimativa é reajustada depois do step 3.

## Visão geral

Construir, no mesmo repositório, um núcleo novo e pequeno para bots de forex de minutos a horas (24 horas nos dias de mercado): um laboratório que testa famílias de estratégia no passado sem se enganar, um runner que opera numa corretora com API, e um painel com lista de bots, retorno por bot, gráficos e botões de ligar, parar e kill-switch. O primeiro dinheiro real é um canário de encanamento de risco mínimo; a decisão sobre estratégias vem do laboratório, não do desejo de operar.

## O que o Igor pediu

Só forex; bots rápidos 24/5; testar 5 a 10 estratégias no passado e manter só as que pagam; corretora com API, começando pequeno com capital próprio e escalando; painel com gráficos, retorno por bot e um botão para ligar tudo; **operar com dinheiro real ainda hoje**.

## A verdade sobre "real hoje"

Um bot operando com dinheiro real hoje não é possível: não existe bot (nada envia ordem a uma corretora de forex), não existe conta (aprovação em dias, remessa em 1 a 3 dias úteis), nenhuma estratégia rápida foi testada e 63% a 80% das contas de varejo perdem dinheiro. Hoje dá para abrir as contas (o que mais atrasa) e começar os spikes. O primeiro dinheiro real é o canário do step 22: previsão de 1 a 2 semanas, dominada por aprovação de conta e remessa, e não por código.

## O que a pesquisa mostrou

- Estratégia de **segundos** não tem evidência de lucro para varejo: o custo é de 1 a 2,4 pips por trade e um movimento de 1 minuto no EURUSD tem ~1,7 pip. Arbitragem triangular, de latência e market making são inviáveis. A vantagem do bot é cobertura 24/5 e disciplina, não velocidade.
- O que vale testar é movimento de **10 a 30 pips em minutos a horas**, onde o custo é 5 a 15% do alvo: rompimento de abertura de sessão, fluxo de fixing, reversão depois de pico de volatilidade (e deriva pós-notícia, que só entra se houver fonte de calendário econômico histórico). Reversão em M1/M5 e momentum de segundos entram só como controle negativo.
- Os dados atuais (velas de 1 hora) não testam isso: precisamos de M1 ou tick com bid e ask.
- **Regra de parada:** se as famílias fixadas antes não confirmarem em amostra nunca vista, encerramos em vez de procurar mais estratégias.
- Só ~14% do código atual serve (laboratório FX, login e TOTP, esqueleto das telas com gráfico TradingView, banco e deploy, a disciplina do gate). O motor atual reage só a vela fechada e faz ~15 consultas ao banco por decisão; o núcleo do bot é novo.

## Decisões técnicas já tomadas

- Python com asyncio: a latência que importa é a da corretora, e estratégia de minutos não precisa de mais. Sem Redis no caminho de ordem (fila em processo). Journal em lote no Postgres, fora do caminho da decisão; ticks em Parquet.
- O runner roda no Railway no início: latência de centenas de ms é irrelevante em minutos a horas. VPS perto da corretora só se algum dia entrar estratégia de segundos.
- Credenciais da corretora ficam só no runner, cifradas.
- Se adotamos um motor de código aberto (por exemplo NautilusTrader) ou construímos o núcleo próprio é decisão por evidência, no step 3.

## Decisões necessárias

1. ⚠ **Quanto aceita perder no canário.** Se ninguém responder, assumo conta de US$ 500, risco por trade de 0,25% (~US$ 1,25) e perda diária máxima de 1% (~US$ 5). O canário mede custo, não lucra.
2. ⚠ **Corretora do canário.** Assumo cTrader Open API (Fusion Markets ou equivalente): demo instantânea, Linux, SDK Python. Ressalva: a CVM considera irregular oferta que capte brasileiro (Pix, site em português).
3. ⚠ **Regra de parada.** Assumo que se as famílias do pré-registro não confirmarem, encerramos o forex rápido em vez de buscar outras.
4. ⚠ **Contador e advogado antes da primeira remessa.** Assumo que a remessa só sai depois do parecer. Sem ele eu não avanço para dinheiro real.
5. ⚠ **Escopo inicial enxuto:** 3 pares (EURUSD, GBPUSD, USDJPY), 3 famílias reais mais 2 controles, 1 corretora, 2 anos de M1. Assumo isso; mais pares e famílias só depois do primeiro resultado.

## Plano de implementação

O `main` só recebe merge quando o Igor pedir; todo o trabalho fica na branch `feat/forex-rebuild`, com a suíte verde a cada step.

### Fase 0: incerteza primeiro (spikes)

**Step 1: Contas e credenciais [Igor]** (~1-2h, mais dias de espera)
  - O que: abrir a conta demo do cTrader (cTID), registrar um aplicativo no portal da Spotware para obter as credenciais da Open API, iniciar o KYC real e mandar as perguntas por escrito · Arquivos: `docs/forex/step5-contas-checklist.md` (já existe)
  - Depende de: nada · Paralelo com: 2, 23
  - Teste: credenciais de demo funcionando; resposta por escrito da corretora sobre residentes no Brasil guardada

**Step 2: Spike de dados M1** (~4-8h)
  - O que: baixar 1 mês de M1 bid/ask do EURUSD da Dukascopy, validar a qualidade, medir o ritmo real e estimar o tempo para 2 anos × 3 pares; se inviável, a fonte alternativa (HistData/TrueFX) · Arquivos: `backend/scripts/research/fx_dataset_m1.py` (novo) + teste
  - Depende de: nada · Paralelo com: 1, 4, 23
  - Teste: o mês passa no controle de qualidade (densidade 5/7, nenhum candle de sábado, spread plausível, sem cotação cruzada) e o relatório traz arquivos/min e ETA. Pior caso: limite de taxa do servidor e formato diferente do H1 (dobrado por ser terceiro)

**Step 3: Spike do motor: adotar pronto ou núcleo próprio** (~4-8h)
  - O que: rodar 1 mês de M1 com bid/ask e o mesmo custo num motor aberto (NautilusTrader ou o que a pesquisa indicar), comparar com um cálculo independente, medir eventos por segundo e checar se há adaptador ao vivo para a corretora · Arquivos: `backend/scripts/research/engine_spike/` (novo), `docs/adr/0001-motor-do-bot.md`
  - Depende de: 2 · Paralelo com: 4, 5
  - Teste: o P&L bate com o cálculo independente (< 0,01 pip por trade) e o throughput é medido; a decisão fica registrada. Pior caso: instalação no Python 3.13. **A estimativa dos steps 6-8 e 16-19 é refeita depois dele.**

**Step 4: Spike da API cTrader na demo** (~6-10h)
  - O que: conectar, receber cotações em streaming, abrir e fechar ordem com SL/TP, derrubar o cliente e conferir que o SL continua no servidor, reconectar · Arquivos: `backend/scripts/research/ctrader_probe.py` (novo)
  - Depende de: 1 · Paralelo com: 2, 3, 5
  - Teste: log mostrando posição e SL vivos depois de matar o processo. Integração externa: pior caso dobra (aprovação do aplicativo no portal)

### Fase 1: laboratório (descobrir se existe estratégia que paga)

**Step 5: Baixar M1 de 2 anos de 3 pares** (~2-3h de esforço, horas de calendário em segundo plano)
  - O que: rodar o downloader do step 2 para EURUSD, GBPUSD e USDJPY com supervisor e retomada · Arquivos: `backend/scripts/research/fx_dataset_m1.py`
  - Depende de: 2 · Paralelo com: 3, 4
  - Teste: os 3 parquets passam o controle de qualidade; relatório por par

**Step 6: Interface de estratégia em streaming** (~3-5h)
  - O que: contrato `on_bar`/`on_quote` com estado incremental e pedidos de ordem, o mesmo código no simulador e no bot vivo · Arquivos: `backend/app/fx/strategy.py` (novo) + teste
  - Depende de: 3 · Paralelo com: 5, 4
  - Teste: uma estratégia de exemplo gera os mesmos sinais alimentada barra a barra e em lote (prova de que não olha adiante)

**Step 7: Simulador event-driven, núcleo** (~4-6h)
  - O que: preencher a mercado no bid/ask, stop e alvo com `fx_gap`, comissão e swap com `fx_costs` · Arquivos: `backend/app/fx/sim/` (novo) + teste
  - Depende de: 6, 5 · Paralelo com: 4
  - Teste: reproduz o P&L do `fx_spike` atual nos mesmos dados H1 (regressão) e roda ≥ 50 mil barras por segundo

**Step 8: Simulador: custo por hora, rollover e latência** (~3-5h)
  - O que: spread por hora do dia, alargamento no rollover (21-23 UTC), latência de ordem e slippage · Arquivos: `backend/app/fx/sim/costs.py` + teste
  - Depende de: 7 · Paralelo com: 9
  - Teste: a mesma ordem custa mais no rollover do que às 14h UTC; a latência atrasa o preenchimento e muda o preço

**Step 9: Pré-registro** (~2-3h)
  - O que: fixar em commit, antes de rodar qualquer estratégia, as famílias, as regras, o custo por hora, a divisão descoberta e confirmação e o critério calibrado por placebo · Arquivos: `docs/forex/fast-preregistration.md` (novo)
  - Depende de: 5, 6 · Paralelo com: 7, 8
  - Teste: commit datado anterior a qualquer resultado; revisado pelo Igor

**Step 10: Família 1, rompimento de abertura de sessão** (~3-5h) · `backend/app/fx/strategies/session_breakout.py` · Depende de: 9 · Paralelo com: 11, 12, 13 · Teste: sinal correto em sequência sintética e teste de não olhar adiante

**Step 11: Família 2, fluxo de fixing** (~3-5h) · `strategies/fixing_flow.py` · Depende de: 9 · Paralelo com: 10, 12, 13 · Teste: entra e sai só nas janelas fixadas, em fusos com horário de verão

**Step 12: Família 3, reversão depois de pico de volatilidade** (~3-5h) · `strategies/spike_fade.py` · Depende de: 9 · Paralelo com: 10, 11, 13 · Teste: stop duro sempre anexado; nunca entra em pico sem spread normalizado

**Step 13: Controles negativos** (~2-3h) · `strategies/controls.py` (reversão em M1/M5 e momentum de segundos, esperamos falha) · Depende de: 9 · Paralelo com: 10, 11, 12 · Teste: rodam no simulador; se algum "passar" no gate, o resultado é tratado como suspeita de defeito

**Step 14: Rodar a descoberta com placebo** (~4-6h)
  - O que: rodar as famílias na amostra de descoberta, com placebo de ≥ 200 embaralhamentos, poder e sensibilidade a custo · Arquivos: `backend/scripts/research/fx_fast_lab.py` (novo), relatório em `docs/forex/`
  - Depende de: 8, 10, 11, 12, 13 · Paralelo com: 16
  - Teste: relatório com valor-p ajustado, efeito mínimo detectável e custo por hora

**Step 15: Confirmação em amostra nunca vista, Portão G1** (~3-5h)
  - O que: aplicar sem mudar nada as famílias que sobreviverem à descoberta · Arquivos: `docs/forex/fast-confirmation-report.md` (novo)
  - Depende de: 14
  - Teste: resultado conforme o pré-registro. **G1:** só o que confirmar vira estratégia do canário; se nada confirmar, vale a regra de parada e o canário roda só o encanamento

### Fase 2: runner e canário real

**Step 16: Runner, feed e barras ao vivo** (~4-6h) · `backend/app/fx/runner/feed.py` · Depende de: 4, 6 · Paralelo com: 7 a 15 · Teste: monta barras M1 em tempo real na demo e conferem com as barras do histórico da corretora

**Step 17: Runner, executor** (~5-8h) · `runner/executor.py` · Depende de: 16 · Paralelo com: 14 · Teste: contra o venue falso e depois contra a demo: ordem com SL/TP anexado no servidor; lote calculado por risco

**Step 18: Runner, risco e kill-switch** (~4-6h) · `runner/risk.py` · Depende de: 17 · Teste: teto de perda diária dispara e para o bot; kill-switch fecha tudo; dead-man switch desliga se o processo travar

**Step 19: Reconciliação no boot** (~4-6h) · `runner/reconcile.py` · Depende de: 18 · Teste: matar o processo com posição aberta e reiniciar reconcilia sem duplicar ordem nem perder o SL

**Step 20: Soak em demo** (~3-5h de esforço, 3 a 5 dias de calendário)
  - O que: rodar uma estratégia trivial de encanamento (entra, sai depois de N minutos, com SL) e comparar custo, latência e slippage medidos contra o simulado · Arquivos: `backend/scripts/research/fx_soak_report.py` (novo)
  - Depende de: 19
  - Teste: relatório com custo medido dentro de 30% do simulado e zero ordem órfã

**Step 21: Remessa e conta real [Igor]** (~3-5h, mais dias) · Depende de: 1 e do parecer do contador e do advogado (Decisão 4) · Teste: conta financiada; IOF e spread cambial anotados

**Step 22: Canário real** (~3-5h de esforço, 1 a 2 semanas de calendário)
  - O que: lote mínimo e o teto de perda da Decisão 1, com a estratégia confirmada no G1 ou o encanamento trivial · Arquivos: `backend/app/fx/runner/config.py` (novo)
  - Depende de: 20, 21
  - Teste: primeira ordem real vista na corretora, relatório diário, nenhum estouro do teto. **Critério: medir custo e slippage reais e provar o encanamento. Não é lucro.**

### Fase 3: plataforma (painel e vários bots)

**Step 23: Modelo de dados por bot (migração aditiva)** (~4-6h, rollback escrito)
  - O que: tabelas de bots, execuções, ordens, fills e equity por bot, só colunas novas · Arquivos: `backend/alembic/versions/019_fx_bots.py`, `backend/app/models/fx_bot.py` · Depende de: nada · Paralelo com: 1, 2
  - Teste: `upgrade` e `downgrade` limpos e dados de cripto intactos. Rollback: `alembic downgrade 018`

**Step 24: Journal em lote e API de leitura** (~5-8h) · `runner/journal.py`, `backend/app/api/v1/fx_bots.py` · Depende de: 23, 17 · Teste: 1 bot em demo gera linhas e o endpoint devolve o retorno por bot

**Step 25: Tela, lista de bots e retorno por bot** (~4-6h) · `frontend/src/app/bots/page.tsx` · Depende de: 24 · Teste: Playwright cria a lista e mostra retorno, tempo e pontos bons e ruins por bot

**Step 26: Tela, gráfico de equity e operações com marcadores** (~4-6h) · `frontend/src/components/charts/` (lightweight-charts) · Depende de: 25 · Teste: Playwright vê as entradas e saídas sobre o gráfico

**Step 27: Ligar e parar pela tela, credenciais seguras** (~4-6h) · Depende de: 24, 18 · Teste: o botão liga o runner na demo; as credenciais nunca aparecem em log nem na API

**Step 28: Kill-switch, TOTP e guard persistente** (~4-6h) · Depende de: 27 · Teste: kill fecha posições; reiniciar o runner mantém o estado do guard; ligar exige TOTP

**Step 29: Alertas por WhatsApp (Evolution API)** (~4-8h) · Depende de: 24 · Teste: desligar o runner dispara a mensagem. Integração externa, pior caso dobra

**Step 30: Vários bots, orçamento de risco por bot** (~3-5h) · Depende de: 18, 23 · Teste: um bot não consegue gastar o orçamento de outro

**Step 31: Vários bots, host de N bots** (~3-5h) · Depende de: 30 · Teste: 5 bots no mesmo par sem conflito; P&L atribuído ao bot certo

**Step 32: Congelar e remover o código de cripto** (~4-8h, rollback: `git tag pre-crypto-removal`) · Depende de: 22 · Teste: suíte verde sem os módulos de cripto

## Riscos

- Nenhuma estratégia rápida ser lucrativa, o desfecho mais provável segundo a pesquisa → o laboratório vem antes do dinheiro e a regra de parada evita gastar meses procurando.
- O canário ser lido como "fui operar de verdade" → é teste de encanamento com perda esperada e teto obrigatório.
- Corretora de varejo como contraparte (last look, rejeição por preço velho, conta pode ser encerrada por fluxo de latência) → nada de estratégia de latência; a demo e o canário medem a diferença entre o simulado e o real.
- Dado agregado da Dukascopy não é o preço executável de cada corretora → step 20 mede a diferença antes de qualquer estratégia entrar no canário.
- Cauda: gap e alargamento de spread (SNB 2015, libra 2016) → stop no servidor da corretora, tamanho mínimo, flat na sexta.
- Jurídico e tributário: remessa só por banco autorizado, declaração da conta no exterior, 15% ao ano sobre o ganho (Lei 14.754) → contador e advogado antes do dinheiro.
- Estimativa: a base já errou 2 a 3× em planos anteriores → usar o teto +50% para calendário.

## Total

**108-177h de desenvolvimento** (30 steps de código; mais 4-7h do Igor nos steps 1 e 21) e semanas de calendário por causa de KYC, remessa e soak. Estimativa do avaliador de código para o escopo completo e amplo (sem ajuda de IA, dev sênior): laboratório rápido 25-39 dias e runner com corretora em demo 39-61 dias; este plano é o escopo enxuto.

**Caminho crítico até o primeiro dinheiro real:** 1 → 4 → 16 → 17 → 18 → 19 → 20 → 21 → 22 (29 a 46h de código, 1 a 2 semanas de calendário). **Caminho crítico até saber se existe estratégia que paga:** 2 → 5 → 6 → 7 → 8 → 9 → 10 a 13 → 14 → 15.
