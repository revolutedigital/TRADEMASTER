# Rodada 17 — sensibilidade de alvo/custo no book prospectivo

Data: 2026-09-23  
Mercado: BTCUSDT futures, WAL prospectivo Fast16 de 18 minutos  
Resultado: alvos curtos baixos são tocáveis; ainda não há estratégia

## Pergunta

A rodada 16 mostrou que o alvo de 12/24 bps em 30s não foi tocado nenhuma vez. A dúvida seguinte era
se o problema estava no tamanho do alvo: talvez o caminho curto exista, mas em alvos bem menores.

Esta rodada mede reachability: para cada alvo bruto em bps, quantas vezes o preço toca esse alvo a
favor antes de tocar um stop inicial de 20 bps, usando as mesmas decisões causais com book fresco.
Isso não é edge líquido, não inclui modelo de entrada e não aprova operação.

## O que foi implementado

Script novo:

`PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_book_threshold_sensitivity.py --wal-root <wal> --output-root <out> --horizons-seconds 30 --thresholds-bps 0.5,1,2,3,5,8,12,24 --decision-stride-seconds 5 --minimum-decisions-for-modeling 200`

O script:

- reutiliza `trade + depth` do WAL prospectivo;
- mantém book obrigatório e fresco;
- recalcula labels exatos para vários alvos;
- separa melhor taxa de toque, melhor caminho bruto com toque e melhor caminho bruto incondicional;
- grava `report.json`;
- mantém flags explícitas de pesquisa: `research_only=true`, `order_submission_allowed=false`,
  `execution_authorization=none`.

## Resultado

Base:

| Campo | Valor |
|---|---:|
| Decisões | 204 |
| Horizonte | 30s |
| Stop inicial | 20 bps |
| Status | `READY_FOR_SENSITIVITY_REVIEW` |

Taxa de toque antes do stop:

| Alvo bruto | BUY toca | SELL toca | Leitura |
|---:|---:|---:|---|
| 0,5 bps | 39,22% | 53,43% | tocável, mas payoff bruto médio ainda negativo no melhor lado |
| 1 bps | 33,82% | 40,20% | tocável, mas fraco |
| 2 bps | 18,14% | 17,16% | já cai forte |
| 3 bps | 8,33% | 4,41% | raro |
| 5 bps | 1,47% | 0,00% | praticamente inexistente |
| 8 bps | 0,00% | 0,00% | inexistente no recorte |
| 12 bps | 0,00% | 0,00% | inexistente no recorte |
| 24 bps | 0,00% | 0,00% | inexistente no recorte |

Melhor taxa de toque:

| Campo | Valor |
|---|---:|
| Lado | SELL |
| Alvo | 0,5 bps |
| Toque antes do stop | 53,43% |
| Stop antes do alvo | 0,00% |
| Payoff bruto médio do caminho | -0,257 bps |

Melhor caminho bruto com algum toque:

| Campo | Valor |
|---|---:|
| Lado | SELL |
| Alvo | 3 bps |
| Toque antes do stop | 4,41% |
| Timeout | 95,59% |
| Payoff bruto médio do caminho | +0,051 bps |

Essa média positiva não é uma estratégia: ela vem quase toda de timeouts e é menor que qualquer custo
operacional realista de entrada/saída. O script marca isso explicitamente como diagnóstico bruto de
reachability, não edge líquido.

## Decisão

Ainda não temos estratégia.

O que a rodada 17 matou foi a esperança de que “é só baixar um pouco o alvo” resolveria para day trade
de 30s. Nesse recorte:

- 0,5/1 bps aparece, mas é estreito demais para custo/slippage comuns;
- 3 bps quase não aparece;
- 5 bps ou mais praticamente não existe;
- 12/24 bps, usados nas rodadas anteriores, são incompatíveis com esse regime curto.

Se houver edge explorável, ele provavelmente precisa de uma destas condições:

1. custo efetivo extremamente baixo, possivelmente maker/rebate/fee tier favorável;
2. horizonte maior que 30s;
3. filtro de regime/evento que só opere quando a volatilidade instantânea abre;
4. entrada baseada em book/imbalance que antecipe movimento antes do restante do mercado;
5. rejeição completa desse mercado/timeframe se o custo real não couber.

Nenhuma ordem foi enviada e este resultado não autoriza Demo, Testnet ou LIVE.

## Artefatos

- Script: `backend/scripts/research/run_book_threshold_sensitivity.py`
- Teste: `backend/tests/unit/test_book_threshold_sensitivity.py`
- Relatório Fast17 ignorado pelo Git:
  `backend/data/microstructure_v1/reports/book-threshold-sensitivity-fast17-18m/report.json`
  (`cbc6966b119fc386d9ed8f331144230aeaa959adfc9c855f1aad7613f5eadf68`)
