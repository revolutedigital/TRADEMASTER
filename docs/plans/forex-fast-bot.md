# Plano: plataforma de bots de forex de curto prazo (rascunho)

Escrito em 2026-09-20 a partir do pedido do Igor. **Rascunho:** as pesquisas de dados M1/tick, corretora/infraestrutura e método de backtest ainda estão rodando; este documento é fechado quando elas terminarem. Substitui o plano `forex-rebuild.md`, encerrado no G0 (estratégia lenta).

## O que o Igor pediu

1. Só forex, nada de cripto.
2. Bots que entram e saem rápido, 24 horas nos dias de mercado, aproveitando momentos que um humano não consegue.
3. Uma plataforma para vários bots. Testar 5 a 10 estratégias (ou mais) no passado de forma massiva e manter só as que dão dinheiro.
4. Corretora com API, rodar primeiro pequeno com capital próprio e escalar se provar.
5. Painel com gráficos, retorno por bot, tempo, pontos bons e ruins, e um botão para ligar tudo com as credenciais da corretora, deixando rodar por horas.
6. **Operar com dinheiro real ainda hoje (2026-09-20).**

## A verdade sobre "real hoje"

Não dá para um **bot** operar com dinheiro real hoje. O que bloqueia:

| Bloqueio | Situação |
|---|---|
| Não existe bot | Nada no repositório envia ordem a uma corretora de forex. O motor atual é de cripto Spot, só compra, só reage a vela fechada de 15 minutos ou mais |
| Não existe conta | Abertura e aprovação (KYC) levam dias; a remessa do dinheiro, de 1 a 3 dias úteis |
| Nenhuma estratégia com vantagem provada | As lentas foram testadas e reprovadas. As rápidas nunca foram testadas |
| Base de comparação | 63% a 80% das contas de varejo perdem dinheiro (ESMA, ASIC, CFTC, corretoras) |
| Jurídico e tributário | Pendente (contador e advogado) |

**O que dá para fazer hoje:** abrir as contas (real e demo) e começar a aprovação, que é o que atrasa tudo; a demo do cTrader abre na hora.

**O caminho mais rápido honesto até dinheiro real** é um **canário de encanamento**: um bot mínimo, 1 estratégia, 1 par, lote mínimo, teto de perda diária, com o objetivo de medir o custo e o slippage **reais** (contra o simulado) e provar que ordem, stop e reconexão funcionam. **Não é para ganhar dinheiro; a expectativa é perder o custo.** Previsão realista: 1 a 2 semanas, limitada por aprovação da conta, remessa e uma passagem em demo.

## O que a pesquisa já mostrou

- **"Rápido" precisa ser redefinido.** Estratégia de segundos (scalping, momentum de segundos, arbitragem, market making) não tem evidência de lucro para varejo: o custo por trade é de 1 a 2,4 pips, e um movimento de M1 no EURUSD tem ~1,7 pip. Arbitragem triangular, de latência e market making são **inviáveis** (dependem de latência de microssegundos e a corretora é a contraparte).
- **O que vale testar:** movimentos de **10 a 30 pips em minutos a horas**, onde o custo é 5 a 15% do alvo: (1) rompimento de abertura de sessão, (2) deriva depois de notícia, (3) fluxo de fixing, (4) reversão depois de pico de volatilidade. Reversão em M1/M5 e momentum de segundos entram só como controle negativo (esperamos falha).
- **A vantagem do bot é cobertura 24/5 e disciplina, não velocidade.** Por isso a latência de um servidor comum basta no início.
- **Os dados atuais (velas de 1 hora) não testam nada disso.** Precisamos de M1 ou tick com bid e ask.
- **Regra de parada:** se as famílias fixadas antes não passarem na confirmação, encerramos em vez de procurar mais estratégias (o risco de achar falso positivo cresce a cada tentativa).

## Decisão sobre o código: construir o núcleo novo no mesmo repositório

Guardar ~14% (laboratório FX, login e TOTP, esqueleto das telas com gráfico TradingView, infraestrutura de banco e deploy, a disciplina do gate). **Substituir** o motor, o gerenciador de ordens, os reconciliadores, a coleta de dados e o hub de WebSocket. **Descartar** o ML, a camada multi-exchange e ~11,6 mil linhas sem uso.

Arquitetura (3 processos, 1 banco, tudo em Python): (1) **lab** offline (simulador event-driven sobre M1/tick, relatório e registro de aprovação); (2) **runner** por conta de corretora (cotações, N bots com orçamento de risco por bot, gate de risco em memória, ordens com SL/TP no servidor da corretora, journal em lote, kill-switch e dead-man switch, reconcilia antes de operar); (3) **api + ui** (FastAPI enxuto e Next.js: lista de bots, retorno por bot, ligar/parar tudo, kill-switch, gráficos com marcadores).

## Decisões necessárias

1. ⚠ **Capital do canário e teto de perda.** Se ninguém responder, assumo conta de US$ 500, risco por trade de 0,25% (~US$ 1,25) e perda diária máxima de 1% (~US$ 5). Você define o valor que aceita perder; o objetivo dele é medir custo, não lucrar.
2. ⚠ **Corretora do canário.** Assumo cTrader Open API (Fusion Markets ou equivalente): demo instantânea, Linux, SDK Python. Ressalva jurídica: a CVM considera oferta irregular quem capta brasileiro (Pix, site em português).
3. ⚠ **Onde roda no início.** Assumo o Railway (a estratégia é de minutos a horas, a latência de algumas centenas de ms não importa); VPS perto da corretora só se algum dia entrar estratégia de segundos.
4. ⚠ **Regra de parada.** Assumo que, se as famílias 1 a 4 não passarem na confirmação em amostra nunca vista, encerramos o forex rápido em vez de buscar outras estratégias.
5. ⚠ **Contador e advogado antes da primeira remessa.** Assumo que a remessa só sai depois do parecer. Sem ele eu não avanço para dinheiro real.

## Trilha A: dinheiro real o mais cedo possível (canário de encanamento)

**A1: Contas e demo** (Igor, ~1-2h hoje, mais dias de espera)
- Abrir a conta demo do cTrader (cTID) e registrar um aplicativo no portal da Spotware para obter as credenciais da Open API; iniciar o KYC na corretora escolhida. Usar `docs/forex/step5-contas-checklist.md`.
- Depende de: nada · Paralelo com: A2 depois que as credenciais existirem.
- Teste: credenciais de demo funcionando; resposta por escrito da corretora sobre residentes no Brasil guardada.

**A2: Prova de API na demo** (~3-6h)
- Conectar, receber cotações em streaming, abrir e fechar ordem com SL/TP, derrubar o cliente e conferir que o SL continua no servidor. Arquivos: `backend/scripts/research/ctrader_probe.py`.
- Depende de: A1 · Paralelo com: B1.
- Teste: log mostrando a posição e o SL vivos depois de matar o processo. Integração externa: pior caso dobra (o portal exige aprovação do aplicativo).

**A3: Runner mínimo** (~8-14h)
- 1 bot, 1 par, estratégia simples da família 1, kill-switch, perda diária máxima, journal em arquivo, sem interface. Arquivos: `backend/app/fx/runner/` (novo).
- Depende de: A2.
- Teste: 3 dias em demo sem intervenção, com reinício forçado no meio e reconciliação correta depois.

**A4: Soak em demo** (calendário 3 a 5 dias, ~1h de esforço) — comparar fills e custo reais da demo com o simulado.

**A5: Remessa e conta real** (Igor) — depende do parecer do contador/advogado (Decisão 5) e da aprovação da conta.

**A6: Canário real** (calendário 1 a 2 semanas) — lote mínimo, teto de perda da Decisão 1. **Critério de sucesso: medir o custo e o slippage reais e provar o encanamento. Não é lucro.**

**Earliest:** cerca de 1 a 2 semanas até o primeiro dinheiro real, dominado por aprovação de conta e remessa, não por código.

## Trilha B: descobrir se existe estratégia que paga (em paralelo, decide o que vai para a Trilha C)

**B1: Dados M1/tick com bid e ask** — EURUSD, GBPUSD e USDJPY, ≥ 2 anos. A fonte depende da pesquisa em andamento (a Dukascopy está limitando o ritmo).
**B2: Pré-registro** — fixar em commit, antes de olhar qualquer resultado, as famílias 1 a 4 (e 6 e 7 como controle negativo), as regras, o custo por hora do dia, o critério e a amostra de confirmação.
**B3: Simulador event-driven M1/tick** — custo por hora, spread de rollover (21-23 UTC), latência e slippage, reaproveitando `fx_costs`, `fx_gap` e a estatística do `fx_spike`.
**B4: Rodar e confirmar** — placebo, holdout que ninguém viu, e só então a decisão.
**Portão:** só passa para a Trilha C o que confirmar em amostra nunca vista. Se nada confirmar, aplicar a regra de parada.

## Trilha C: a plataforma completa

Depois do canário e de pelo menos uma estratégia confirmada: runner multi-bot com orçamento de risco por bot, adapter completo da corretora, modelo de dados por bot, telas (lista de bots, retorno por bot, gráficos com marcadores, ligar/parar tudo, kill-switch), alertas por WhatsApp (Evolution API) e guard persistente com TOTP na abertura. **Estimativa do avaliador de código (dev sênior, sem contar ajuda de IA, ±40%):** laboratório rápido 25 a 39 dias; runner e corretora em demo 39 a 61 dias. Esta trilha é quebrada em steps de 4 a 8h quando a pesquisa fechar e a Trilha B disser quais bots existem.

## Riscos

- O canário pode ser lido como "fui operar de verdade". É teste de encanamento com perda esperada; o teto de perda é obrigatório.
- Nenhuma estratégia rápida ser lucrativa. É o desfecho mais provável segundo a pesquisa; a regra de parada evita gastar meses procurando.
- Backtest com dado agregado da Dukascopy não é o preço executável de cada corretora; a demo e o canário existem para medir essa diferença.
- Jurídico e tributário: remessa por banco autorizado, declaração da conta no exterior e imposto de 15% ao ano sobre ganho (Lei 14.754). Contador e advogado antes do dinheiro.
- Cauda: gap e alargamento de spread (SNB 2015, libra 2016). Stop no servidor da corretora e tamanho mínimo.
