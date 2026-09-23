# Rodada 15 — ranking por payoff executável direto

Data: 2026-09-23  
Mercado: BTCUSDT futures, microestrutura local `research-v1`, 7 dias  
Resultado: ranking por payoff terminal rejeitado no scout

## Pergunta

A rodada 14 mostrou que o problema não era apenas o label ignorar 100 ms de latência. Ainda restava
uma hipótese importante: talvez a arquitetura estivesse otimizando o alvo errado. Em vez de prever
“tocou alvo antes do stop”, esta rodada tentou prever diretamente o retorno líquido terminal depois
da latência.

O objetivo era testar se ranking por payoff executável separa melhor as entradas antes de passar pelo
mesmo replay com uma posição por vez, breakeven e trailing.

## O que foi rodado

Comando:

`PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_direct_payoff_scout.py --target stress --latency-ms 100`

O script:

- lê `backend/data/microstructure_v1/research-v1`;
- recalcula o caminho executável a partir de `decision_time + 100 ms`;
- cria `latency_terminal_stress_net_bps`;
- treina regressão temporal walk-forward com `flow`, `flow_price` e `flow_price_session`;
- escolhe o candidato pelo top 1% de payoff previsto;
- passa o candidato selecionado pelo replay de portfólio com políticas de trailing v1.

## Resultado do ranking

O melhor candidato foi:

| Campo | Valor |
|---|---:|
| Feature set | `flow_price` |
| Horizonte | 120s |
| Correlação média previsão→payoff | 0,0145 |
| Top-p payoff monotônico | sim |
| Eventos selecionados no top 1% | 1.935 |
| Média realizada no top 1% | -23,499 bps |
| Taxa positiva no top 1% | 3,62% |

Ou seja: o modelo não conseguiu prever payoff líquido de forma útil. A correlação ficou praticamente
zero e o melhor top 1% ainda teve média stress negativa.

## Resultado do replay

A melhor política no replay foi `fixed_24`, top 1%, também negativa:

| Métrica | Valor |
|---|---:|
| Trades | 431 |
| Dias distintos | 3 |
| Média expected | -11,501 bps |
| Média stress | -23,501 bps |
| Win rate expected | 10,67% |
| Drawdown máximo | 4.957 bps |

Todas as 20 combinações de política/trailing ficaram negativas em expected e stress.

## Decisão

Esta hipótese também não encontrou estratégia. No dataset local de 7 dias, as features agregadas de
fluxo/preço não predizem payoff líquido terminal com força suficiente para virar trade. Isso reforça
que o gargalo não é só o target; é a informação disponível.

Próximo caminho válido: não insistir em `aggTrades`/preço-volume agregado. A próxima linha precisa
de dados prospectivos de book/depth, microprice, spread real, mark/spot-perp e, quando houver, liquidação.
Sem isso, ficamos reciclando variações que já estão estatisticamente sem sinal.

Nenhuma ordem foi enviada e este resultado não autoriza Demo, Testnet ou LIVE.

## Artefatos

- Relatório JSON ignorado pelo Git: `backend/data/microstructure_v1/reports/direct-payoff-scout-stress/report.json`
  (`616c4b3598ddd36f1c482f4ff40211b3b10361e1335b09e55623dd5a626a519a`)
- Script: `backend/scripts/research/run_direct_payoff_scout.py`
  (`da8c045c98bceab35c35b7386a23980707475951302946c3dd3126d2b85519b5`)
- Teste: `backend/tests/unit/test_direct_payoff_scout.py`
  (`e7016448ef7a6d1541cb80a9a9da9568e7522bb5cab623662bdd6669ee7bfc46`)
