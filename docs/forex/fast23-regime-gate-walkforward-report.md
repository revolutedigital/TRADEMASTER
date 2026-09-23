# Rodada 23 — gate de regime, correção de vazamento e candidata robusta

Data: 2026-09-23  
Mercado: BTCUSDT futures, WALs prospectivos Fast20/Fast21/Fast22 de 30 minutos  
Resultado: ainda não há estratégia aprovada, mas há uma candidata robusta para futuro holdout

## Pergunta

Fast22 mostrou que a regra isolada `spot_perp_basis` não é estável: funcionou em Fast20/Fast22 e
falhou forte em Fast21. A pergunta desta rodada foi:

> existe um gate causal que bloqueia os regimes ruins e mantém a regra `spot_perp high` SELL viva?

Base congelada:

| Campo | Valor |
|---|---:|
| Regra base | `directed_spot_perp_basis_bps` high |
| Lado selecionado | SELL |
| Threshold base | fitado só em Fast16 |
| Fração base | 20% |
| Horizonte | 120s |
| Alvo bruto | 5 bps |

## Correção metodológica

O primeiro scout de gate tinha um vazamento leve: o ranking era feito por desenvolvimento, mas o
holdout também filtrava candidatos com pouca amostra. Isso não usava o payoff do holdout para escolher,
mas ainda deixava a janela de confirmação influenciar a lista elegível.

Correção aplicada:

- gate é escolhido e ranqueado só no gate-development;
- confirmação só mede o gate já escolhido;
- contagem mínima de confirmação é checada depois da seleção;
- o relatório agora registra `confirmation_minimum_count_checked_after_selection=true`.

Isso invalidou a leitura “conservadora” anterior. Sem esse vazamento, o min30 não sobrevive.

## Top-1 por fold não é estável

Rodei três folds com a base em Fast16 e alternando Fast20/Fast21/Fast22 como desenvolvimento/holdout.

| Gate-development | Holdout | Top gate escolhido | Status | N holdout | Gross holdout | Net 1 bps |
|---|---|---|---|---:|---:|---:|
| Fast20+Fast21 | Fast22 | `book_ask_liquidity_removed_qty_30s` high 20% | `READY_FOR_REGIME_GATE_REVIEW` | 23 | +1,544 bps | +0,544 bps |
| Fast20+Fast22 | Fast21 | `funding_rate` high 10% | `CONFIRMATION_INSUFFICIENT_SELECTIONS` | 0 | 0,000 bps | -1,000 bps |
| Fast21+Fast22 | Fast20 | `funding_rate` high 10% | `CONFIRMATION_INSUFFICIENT_SELECTIONS` | 0 | 0,000 bps | -1,000 bps |

Conclusão: o “melhor top-1” continua overfitado. Ele parece ótimo na janela onde foi escolhido, mas
não forma estratégia estável.

## O candidato que apareceu nos três folds

Ao cruzar todos os candidatos exportados, não só o top-1, apareceu um gate recorrente:

| Campo | Valor |
|---|---:|
| Gate | `spot_update_age_ms` low |
| Fração | 33% |
| Interpretação | só operar quando o dado spot está fresco |
| Folds com N mínimo | 3/3 |
| Folds positivos após 1 bps | 3/3 |
| N mínimo em holdout | 12 |
| Gross médio em holdout | +1,917 bps |
| Net médio em 1 bps | +0,917 bps |
| Pior net em 1 bps | +0,090 bps |

Isso muda a hipótese: o edge não parece ser “mais um indicador mágico”. Parece ser:

1. vender perp quando o basis spot/perp está no extremo;
2. mas só quando a referência spot está atualizada o suficiente para o sinal não virar ruído;
3. se o spot está atrasado/stale, o sinal de basis pode estar mentindo.

## Confirmação da regra fixa

Depois do resumo walk-forward, rodei um script novo que não procura feature nenhuma. Ele recebe
explicitamente `spot_update_age_ms low 33%`, fita o threshold no gate-development, congela, e mede no
holdout.

| Gate-development | Holdout | Threshold `spot_update_age_ms` | N holdout | Gross holdout | Net 0,5 bps | Net 1 bps | Net 2 bps |
|---|---|---:|---:|---:|---:|---:|---:|
| Fast20+Fast21 | Fast22 | 143,00 ms | 45 | +1,892 bps | +1,392 bps | +0,892 bps | -0,108 bps |
| Fast20+Fast22 | Fast21 | 133,04 ms | 12 | +1,090 bps | +0,590 bps | +0,090 bps | -0,910 bps |
| Fast21+Fast22 | Fast20 | 142,76 ms | 20 | +2,768 bps | +2,268 bps | +1,768 bps | +0,768 bps |

O ponto fraco é claro: o fold que segura Fast21 como holdout passou por pouco, com só 12 trades e
+0,090 bps líquido em 1 bps. Portanto ainda não é estratégia.

## Decisão

Ainda não temos estratégia aprovada.

Mas agora temos uma candidata real para futuro holdout:

```text
Base: SELL quando directed_spot_perp_basis_bps está no top 20% fitado em Fast16.
Gate: permitir só quando spot_update_age_ms está no bottom 33% fitado no conjunto de desenvolvimento.
Horizon: 120s.
Target bruto: 5 bps.
```

O que melhorou:

- deixou de ser um top-1 instável;
- confirmou em 3/3 folds walk-forward internos;
- ficou positivo depois de 1 bps em 3/3 folds;
- tem interpretação operacional: qualidade/frescor do dado spot;
- bloqueia parte da sujeira que destruía o `spot_perp_basis` puro.

O que ainda bloqueia operação:

- foi escolhido olhando Fast20/Fast21/Fast22; precisa de uma janela futura Fast24;
- o pior fold tem só 12 trades;
- o pior net em 1 bps é só +0,090 bps;
- a 2 bps de custo, 2 de 3 folds ficam negativos;
- isso ainda não inclui simulação real maker/taker, fila, latência, slippage ou rejeição de ordem;
- não passou por gate estatístico final;
- `approved_strategy=false`, `order_submission_allowed=false`, `execution_authorization=none`.

Próximo passo correto:

1. capturar Fast24 sem mudar regra;
2. fitar o threshold do gate em Fast20+Fast21+Fast22;
3. aplicar em Fast24 sem reotimizar;
4. exigir N mínimo e net positivo com custo realista;
5. se sobreviver, promover para Testnet shadow/replay; não para LIVE.

Nenhuma ordem foi enviada. Demo/Testnet/LIVE seguem bloqueados.

## Artefatos

- Script scout de gate:
  `backend/scripts/research/run_book_regime_gate_scout.py`
- Script walk-forward matrix:
  `backend/scripts/research/summarize_book_regime_gate_walkforward.py`
- Script confirmação de gate fixo:
  `backend/scripts/research/run_book_regime_gate_confirmation.py`
- Scout Fast20+Fast21 → Fast22:
  `backend/data/microstructure_v1/reports/book-regime-gate-scout-fast23-spot-f20-f21-to-f22/report.json`
  (`d248e29a43476a143af74ccfb517b22eb1f5302b938cf1d22fdc3204690aa738`)
- Scout Fast20+Fast22 → Fast21:
  `backend/data/microstructure_v1/reports/book-regime-gate-scout-fast23-spot-f20-f22-to-f21/report.json`
  (`733b82b3e633eb4564622ab639ae61fd99f917597e47ecab577c5e6572a8ac78`)
- Scout Fast21+Fast22 → Fast20:
  `backend/data/microstructure_v1/reports/book-regime-gate-scout-fast23-spot-f21-f22-to-f20/report.json`
  (`a1a3face70664a27f99456adb1b63cd998436ce18dcf3ab0facc54868d1c0a85`)
- Scout min30 corrigido Fast20+Fast21 → Fast22:
  `backend/data/microstructure_v1/reports/book-regime-gate-scout-fast23-spot-f20-f21-to-f22-min30/report.json`
  (`cc86f9cd85471c33d2466d4c157c1555a17efd27ef0157ab0e1745b13dee4848`)
- Walk-forward matrix:
  `backend/data/microstructure_v1/reports/book-regime-gate-walkforward-fast23-spot/report.json`
  (`28cb1eb3d106532dcc43ba30a8b6c464ed92163f60f85372c6016f6e3100ea3b`)
- Confirmação fixa Fast20+Fast21 → Fast22:
  `backend/data/microstructure_v1/reports/book-regime-gate-confirmation-fast23-spot-age-f20-f21-to-f22/report.json`
  (`06e03602203783c928a50662accf5919b6545feaedbaa2b67eadd0be049ea85b`)
- Confirmação fixa Fast20+Fast22 → Fast21:
  `backend/data/microstructure_v1/reports/book-regime-gate-confirmation-fast23-spot-age-f20-f22-to-f21/report.json`
  (`9e85f7555a05980019f0a1e080c77a3d9117e15472a48f818b3ac858ef5fb4d7`)
- Confirmação fixa Fast21+Fast22 → Fast20:
  `backend/data/microstructure_v1/reports/book-regime-gate-confirmation-fast23-spot-age-f21-f22-to-f20/report.json`
  (`6b080f7ccb1ef8cd4061d67e51c8e408e7ce35238c523847c5672a8c9ab13a41`)
