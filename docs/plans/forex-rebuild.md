# Plano: refazer o TradeMaster em forex

Aprovado por Igor em 2026-09-19 ("siga até o final, em todas as decisões faça o que recomendou").
Branch de trabalho: `feat/forex-rebuild` (commits locais, sem push).
Pesquisa que sustenta o plano: [docs/forex/venue-research-2026-09-19.md](../forex/venue-research-2026-09-19.md).

## Visão geral

O TradeMaster vira plataforma de forex long/short, mantendo o gate de evidência e o live guard. Troca cripto Spot por instrumentos FX com custo em pips, margem e calendário de sessão, e uma corretora atrás de uma interface neutra. A Fase 0 prova a vantagem antes de reescrever; a Fase A entrega tudo em PAPER; a Fase B liga a corretora em demo e depois em real.

## Decisões (todas aprovadas na recomendação)

1. **Rota de corretora:** IBKR primeiro (forex spot real, Brasil listado, paper na mesma API), com interface neutra `ForexVenue` para plugar cTrader depois. Muda steps 6, 7, 25, 26.
2. **Capital e risco:** conta em USD, risco de 0,25% por trade e perda diária máxima de 1% da equity (mesmo canário de hoje).
3. **Timeframe e universo:** H4/D1 (sem 15m), 7 majors com gate agrupado; USDBRL como caso à parte (carry pesa).
4. **Netting** (uma posição líquida por par), como na IBKR.
5. **Código cripto:** congelar, sem apagar, até a Fase B passar; apagar é o step 32, com rollback.
6. **Parecer de advogado e contador antes do step 31** (primeiro dinheiro real). Sem ele não se avança.

## Portões (o plano para aqui se falhar)

- **G0 (após step 8):** só segue para a Fase A se alguma configuração tiver expectativa positiva depois do custo, com IC 95% excluindo zero, em pelo menos 2 pares. Senão: parar e reportar (carry/D1, rota WDO da B3, ou não fazer).
- **G1 (step 5):** se nenhuma corretora aceitar o KYC nem a remessa for viável, o plano B é WDO na B3 (só USD/BRL).
- **Portão A (após step 23):** 4 semanas de paper soak, comparando fills simulados com spread real.
- **Step 31:** exige o parecer jurídico/contábil (Decisão 6).

## Quadro de progresso

| Step | Título | Estimativa | Status |
|---|---|---|---|
| 1 | Alinhar spec do gate ao código | 2-3h | concluído (spec = código; teste de contrato falha se divergir) |
| 2 | Dados 3 anos H1, 7 majors, bid/ask | 4-8h | código e testes prontos; download real em andamento (servidor lento, ~5 arquivos/min); validação de qualidade sobre dado real pendente |
| 3 | Custo em pips (spread + comissão) | 4-6h | concluído (`fx_costs.py`, 22 testes) |
| 4 | Swap/rollover e fill de gap | 4-6h | concluído (swap em `fx_costs.py`, gap em `fx_gap.py`, 20 testes) |
| 5 | Contas e caminho do dinheiro (humano) | 3-5h + espera | pendente |
| 6 | Prova de API cTrader demo | 6-10h | pendente |
| 7 | Prova de API IBKR paper | 8-14h | pendente |
| 8 | Walk-forward FX com IC (G0) | 6-10h | pendente |
| 9 | Spec v2 do perfil FX | 4-6h | pendente |
| 10 | Migração aditiva 019 | 4-6h | pendente |
| 11 | Instrumentos, pip/lote e conversão | 6-8h | pendente |
| 12 | Long/short nos schemas e sinais | 5-8h | pendente |
| 13 | Long/short no gate e estudo de ativo | 6-10h | pendente |
| 14 | Custo FX e gate com IC no backtest | 6-10h | pendente |
| 15 | Calendário de sessão | 6-10h | pendente |
| 16 | Dados FX em produção | 6-10h | pendente |
| 17 | Estratégias sem volume | 6-10h | pendente |
| 18 | Sizing em lotes e equity real | 6-10h | pendente |
| 19 | Exposição líquida por moeda | 5-8h | pendente |
| 20a | Short em paper e SL/TP atômico | 6-10h | pendente |
| 20b | Regras de sessão no engine | 6-10h | pendente |
| 21 | Simulador paper bid/ask | 6-8h | pendente |
| 22 | Tela: escolher par e estudar | 4-8h | pendente |
| 23 | Tela: gate e ativar em PAPER | 4-8h | pendente |
| 24 | Interface ForexVenue + venue falso | 5-8h | pendente |
| 25 | Adapter da corretora #1: dados | 8-12h | pendente |
| 26 | Adapter: ordens, posições, conta | 8-14h | pendente |
| 27 | Reconciliador por posição | 8-12h | pendente |
| 28 | Verificador de release em demo | 6-10h | pendente |
| 29 | Live guard FX | 6-10h | pendente |
| 30 | Canário em demo | 3-5h + 2-4 semanas | pendente |
| 31 | Primeiro dinheiro real (mínimo) | 4-6h + gate humano | pendente |
| 32 | Aposentar o código cripto | 4-8h | pendente |

**Total:** Fase 0 = 37-62h, Fase A = 86-140h, Fase B = 52-85h → **175-287h** (22 a 36 dias de dev). Um agente que leu o código inteiro estimou 65 a 95 dias; a diferença é retrabalho de teste e integração que só aparece no meio. Para prazo com terceiros, usar o teto +50% e reestimar no G0.

**Caminho crítico:** 2 → 3 → 4 → 8 → G0 → 9 → 12 → 13 → 14 → 15 → 20b (via 20a) → 24 → 26 → 27 → 29 → 30 → 31.

## Fase 0: provar vantagem depois do custo (nada toca a produção)

**Step 1: Alinhar spec do gate ao código** (~2-3h)
- O que: spec dizer 50 trades / PF 1,15 / 65% (o código aplica isso; o spec dizia 20 / 1,05 / 60%). Arquivos: `docs/openapi/strategy-deployment.yaml`, `backend/tests/unit/test_strategy_deployments.py`.
- Depende de: nada · Paralelo com: 2, 5, 6.
- Teste: um teste compara as constantes do módulo com o spec parseado e falha se divergirem.

**Step 2: Dados de 3 anos em H1, 7 majors, com bid/ask** (~4-8h)
- O que: baixar histórico público e gravar em parquet fora do banco. Arquivos: `backend/scripts/research/fx_dataset.py`.
- Depende de: nada · Paralelo com: 1, 5, 6.
- Teste: densidade ≈ 5/7 dos candles corridos, nenhum candle no sábado, spread mediano plausível por par. Pior caso: formato binário, limite de taxa ou termos de uso. Fallback: histórico da corretora demo (step 6).

**Step 3: Custo em pips (spread real + comissão por lote)** (~4-6h)
- O que: módulo novo de custo, sem tocar no `engine.py`. Arquivos: `backend/app/services/backtest/fx_costs.py` + teste.
- Depende de: 2 · Paralelo com: 5, 6, 7.
- Teste: EURUSD com spread de 1 pip e comissão de US$ 2,25 por lote por lado custa o valor esperado em USD por lote.

**Step 4: Swap/rollover e fill de gap** (~4-6h)
- O que: swap diário com triplo na quarta, sinal por direção, stop preenchido no preço do gap de domingo. Arquivos: `fx_costs.py`, `fx_gap.py`.
- Depende de: 3 · Paralelo com: 6, 7.
- Teste: posição carregada da quarta pra quinta paga 3 dias; stop atravessado por gap sai no preço do gap.

**Step 5: Contas e caminho do dinheiro** (~3-5h do Igor, mais dias de espera)
- O que: abrir conta IBKR; pedir por escrito a Fusion, Dukascopy e Capital.com se aceitam residente no Brasil; cotar remessa em 2-3 bancos; agendar contador e advogado. Arquivos: nenhum (checklist em `docs/forex/step5-contas-checklist.md`).
- Depende de: nada · Paralelo com: 1 a 4, 6.
- Teste: ≥2 respostas por escrito guardadas e ≥2 cotações de remessa com IOF explícito. **G1.**

**Step 6: Prova de API cTrader demo** (~6-10h)
- O que: OAuth demo, abrir posição com SL/TP, derrubar o cliente, conferir que o SL continua no servidor. Arquivos: `backend/scripts/research/ctrader_probe.py`.
- Depende de: nada (credenciais do Igor) · Paralelo com: 1 a 5.
- Teste: log mostrando a posição e o SL vivos depois de matar o processo.

**Step 7: Prova de API IBKR paper** (~8-14h)
- O que: IB Gateway em container Linux, `ib_async`, bracket em paper, reiniciar o container e conferir SL e re-login. Arquivos: `backend/scripts/research/ibkr_probe.py`, compose de pesquisa.
- Depende de: 5 (conta aprovada) · Paralelo com: 3, 4, 8.
- Teste: bracket aberto, container reiniciado, posição e SL reconciliados.

**Step 8: Walk-forward FX com intervalo de confiança (G0)** (~6-10h)
- O que: rodar as estratégias técnicas atuais long/short, sem volume, com custo e calendário, e reportar PF e expectativa depois do custo com IC bootstrap. Arquivos: `backend/scripts/research/fx_spike.py`.
- Depende de: 3, 4 · Paralelo com: 6, 7.
- Teste: relatório em markdown por par e estratégia.

## Fase A: núcleo FX em PAPER (sem corretora, sem dinheiro)

**Step 9: Spec v2 do perfil FX** (~4-6h) — perfil `fx_margin_long_short`, instrumento, lote, margem, textos do gate/guard; o v1 Spot fica intacto. Arquivos: `docs/openapi/*.yaml`. Depende de: G0. Teste: Prism sobe o mock e o spec valida (3.1).

**Step 10: Migração aditiva 019** (~4-6h, rollback escrito) — tabela `fx_instruments`; colunas novas *nullable* em `positions`. Arquivos: `backend/alembic/versions/019_fx_core.py`, `models/portfolio.py`, `models/market.py`. Depende de: 9. Teste: `upgrade` e `downgrade` limpos, dados Spot intactos. Rollback: `alembic downgrade 018`.

**Step 11: Instrumentos, pip/lote e conversão de moeda** (~6-8h) — `backend/app/services/fx/instruments.py`, `conversion.py`. Depende de: 10. Teste: propriedade (conversão de ida e volta, pip value de JPY e BRL).

**Step 12: Long/short nos schemas e sinais** (~5-8h) — `schemas/trading.py`, `backtest/technical_strategy.py`. Depende de: 9. Teste: testes Spot existentes passam; sinal −1 no perfil FX vira short.

**Step 13: Long/short no gate e no estudo de ativo** (~6-10h) — trocar `Literal['spot_long_only']` (~12 pontos) por união discriminada. Arquivos: `strategy_deployments.py`, `asset_intelligence.py`. Depende de: 12. Teste: `test_strategy_deployments` e `test_asset_intelligence` verdes nos dois perfis.

**Step 14: Custo FX e gate com IC no backtest de produção** (~6-10h) — promover `fx_costs` ao `BacktestEngine`. Arquivos: `engine.py`, `walk_forward.py`, `strategy_deployments.py`. Depende de: 8, 13. Teste: backtest de referência com custo conhecido reproduz o P&L; `test_backtest` e `test_walk_forward` verdes.

**Step 15: Calendário de sessão** (~6-10h) — `market/freshness.py`, `walk_forward.py`, `trading_engine.py`. Depende de: 14. Teste: janela cruzando o fim de semana passa como contínua; engine não fica bloqueado na segunda.

**Step 16: Dados FX em produção** (~6-10h) — tabela `fx_candles` com bid/ask e spread, importador, validador. Arquivos: `market/fx_data.py`, `data_validator.py`. Depende de: 10. Teste: importar 1 mês de EURUSD e conferir contagem, spread e gaps.

**Step 17: Estratégias sem volume** (~6-10h) — tirar `volume_confirmation` dos candidatos FX; tendência H4/D1, rompimento de sessão, filtro de carry. Arquivos: `asset_intelligence.py`, `pattern_intelligence.py`, `technical_strategy.py`. Depende de: 13, 16. Teste: estudo de EURUSD H4 devolve candidatos.

**Step 18: Sizing em lotes e equity real** (~6-10h) — lote = risco em moeda da conta ÷ (stop em pips × pip value); equity = saldo + flutuante. Arquivos: `risk/position_sizer.py`, `risk/manager.py`. Depende de: 11. Teste: propriedade "perda no stop = risco definido" em EURUSD, USDJPY e USDBRL.

**Step 19: Exposição líquida por moeda e alavancagem efetiva** (~5-8h) — `risk/correlation.py`, `risk/manager.py`. Depende de: 18. Teste: par de posições que dobra a exposição ao USD é recusado.

**Step 20a: Short em paper e SL/TP atômico** (~6-10h) — `trading_engine.py`, `portfolio/tracker.py`. Depende de: 12, 18. Teste: testes de caracterização do engine antes de mexer; ciclo abre short, stop e fecha em paper.

**Step 20b: Regras de sessão no engine** (~6-10h) — blackout de rollover/notícia, flat na sexta, kill-switch de perda diária em equity. Depende de: 15, 20a. Teste: entrada bloqueada no rollover; perda diária > 1% trava novas entradas.

**Step 21: Simulador paper bid/ask** (~6-8h) — fill em bid/ask, comissão por lote, swap, recusa com mercado fechado. Arquivo: `exchange/order_manager.py` (caminho paper). Depende de: 11, 14. Teste: ordem com mercado fechado é recusada; P&L de ida e volta bate com o custo.

**Step 22: Tela: escolher par e estudar** (~4-8h) — `frontend/src/app/operar/page.tsx`. Depende de: 13, 16. Teste: teste de tela + Playwright.

**Step 23: Tela: gate e ativar em PAPER** (~4-8h) — Depende de: 22, 20a. Teste: Playwright ativa em PAPER e vê a posição short simulada.

## Fase B: corretora real (demo primeiro)

**Step 24: Interface `ForexVenue` + venue falso** (~5-8h) — `backend/app/services/fx/venue.py`. Depende de: 11, 20a. Teste: suíte do engine roda contra o venue falso.

**Step 25: Adapter da corretora #1, dados e candles** (~8-12h) — `backend/app/services/fx/venues/ibkr.py`. Depende de: 24, 7. Teste: candles bid/ask no paper batem com o importador do step 16.

**Step 26: Adapter: ordens com SL/TP, posições e conta** (~8-14h) — Depende de: 25. Teste: abrir e fechar posição mínima em paper com SL, conferir por leitura.

**Step 27: Reconciliador por posição** (~8-12h) — substitui `spot_protection_reconciler`, `spot_account_inventory_reconciler`, `spot_position_closer`. Arquivo: `fx/reconciler.py`. Depende de: 26. Teste: divergência de SL provocada é detectada e marca a posição.

**Step 28: Verificador de release em demo** (~6-10h) — equivalente ao `testnet_protection_verifier`. Arquivo: `fx/release_verifier.py`. Depende de: 26. Teste: relatório PASSED em conta demo, sem posição sobrando.

**Step 29: Live guard FX** (~6-10h) — `exchange/live_trading_guard.py`, `config.py`, specs. Depende de: 27, 28. Teste: `test_live_trading_guard` adaptado; arm sem verificação de demo recente é recusado.

**Step 30: Canário em demo** (~3-5h de esforço, 2-4 semanas de calendário) — Depende de: 29, Portão A. Teste: relatório semanal de derrapagem dentro do limite.

**Step 31: Primeiro dinheiro real em tamanho mínimo** (~4-6h + gate humano) — Depende de: 30, Decisão 6. Teste: primeira ordem real vista na corretora, reconciliação READY, SL confirmado por leitura.

**Step 32: Aposentar o código cripto** (~4-8h, rollback escrito) — Depende de: 31 + 4 semanas estáveis. Teste: suíte verde sem os módulos. Rollback: `git tag pre-crypto-removal` antes e `git revert` do commit único.

## Riscos

- Nenhuma estratégia sobrevive ao custo (G0) → spike primeiro, barato; se falhar, parar e reportar.
- KYC recusado ou banco trava a remessa → step 5 cedo; WDO como plano B.
- IBKR com 2FA semanal, headless não suportado e container comunitário (IBC aposentado em 01/09/2026) → step 7 testa reinício e re-login; operação já é supervisionada por TOTP.
- `trading_engine.py` (1.178 linhas, cobertura parcial) → testes de caracterização antes do refactor (step 20a).
- Suíte de ~584 testes assume Spot/USDT → cada step conserta o que quebra.
- Stop simulado e gap → flat na sexta, teto de alavancagem efetiva, SL testado com o cliente derrubado (steps 6, 7, 28).
- Jurídico e tributário → gate humano antes do step 31.
