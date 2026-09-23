# Fast38 — diagnóstico de drift por regime

Data: 2026-09-23  
Mercado: BTCUSDT futures  
Janelas: Fast24, Fast28, Fast29, Fast32, Fast35, Fast36  
Resultado: a falha Fast37 parece ligada a instabilidade de regime, não só ao modelo de entrada.

## Contexto

Fast37 corrigiu o alvo do modelo para payoff executável, mas ainda falhou
replicação por janela. O melhor candidato limpo ficou positivo no agregado,
mas quebrou em Fast35 e não replicou one-position em todas as janelas.

Este diagnóstico comparou médias de features causais por janela, usando apenas
dados conhecidos no momento da decisão, para mapear drift de regime.

## Drift observado

### Fast35 — janela adversa para o melhor candidato Fast37

Fast35 foi a pior janela do melhor candidato Fast37:

- independente: net -5,4088 bps em 1 bps;
- one-position: net -2,7393 bps em 1 bps.

Top drifts causais observados:

| Feature | z-score | Média Fast35 | Média geral |
| --- | ---: | ---: | ---: |
| `spot_perp_return_gap_1s_bps` | +1,908 | +0,019763 | -0,008986 |
| `book_depth_imbalance_change_1s` | +1,601 | +0,019393 | -0,000300 |
| `book_microprice_displacement_change_bps_1s` | +1,596 | +0,000334 | -0,000030 |
| `flow_imbalance_30s` | +1,482 | +0,136583 | -0,017550 |
| `flow_imbalance_5s` | +1,242 | +0,133215 | +0,019955 |
| `book_pressure_imbalance_30s` | +1,240 | +0,007984 | -0,013446 |
| `spot_mean_interarrival_ms_5s` | -1,092 | 54,758 | 108,087 |
| `return_30s_bps` | +1,042 | +0,536528 | -0,590563 |
| `book_pressure_imbalance_1s` | -1,036 | -0,027930 | +0,015293 |

Leitura: Fast35 não foi simplesmente "mercado parado". Foi uma janela com
fluxo direcional/spot-perp e microestrutura divergentes. O modelo de payoff
superestimou o edge nesse regime.

### Fast36 — regime de volume/volatilidade extrema

Fast36 rejeitou a tese one-position anterior, mas no payoff scout limpo foi uma
janela positiva para o melhor candidato Fast37.

Top drifts causais observados:

| Feature | z-score | Média Fast36 | Média geral |
| --- | ---: | ---: | ---: |
| `book_spread_recovery_bps_5s` | +2,041 | +0,008482 | +0,001426 |
| `book_spread_recovery_bps_30s` | +2,041 | +0,007406 | +0,001247 |
| `book_microprice_displacement_change_bps_5s` | +2,035 | +0,003125 | +0,000453 |
| `realized_vol_5s_bps` | +2,029 | +2,400064 | +0,633648 |
| `realized_vol_30s_bps` | +2,026 | +8,747552 | +2,238410 |
| `book_microprice_displacement_change_bps_30s` | +2,025 | +0,002620 | +0,000383 |
| `spot_quote_volume_5s` | +2,006 | 521.829 | 168.320 |
| `spot_quote_volume_30s` | +2,005 | 3.090.980 | 997.550 |
| `quote_volume_30s` | +2,004 | 35.846.800 | 10.666.800 |
| `trade_count_5s` | +1,984 | 1.070,62 | 380,85 |

Leitura: Fast36 parece outro regime: volume e volatilidade muito acima da
média. Isso reforça que a entrada precisa saber quando não operar ou trocar de
submodelo.

## Decisão

Não temos estratégia.

Mas a próxima frente ficou mais específica:

> construir primeiro um gate de regime/no-trade, e só depois ranquear entradas.

O erro das rodadas Fast34-Fast37 foi tentar resolver entrada e saída antes de
resolver "em qual regime este edge existe?".

## Próximo experimento sugerido

Fast39 deve testar uma dessas duas abordagens:

1. `regime_gate_then_payoff_model`: classificar regime por features agregadas
   causais e só permitir entrada em regimes historicamente estáveis;
2. `window_abstention_model`: modelo prediz se a janela atual deve ser operada
   ou bloqueada antes de qualquer top-p.

Critério mínimo:

- gate precisa ser decidido sem olhar o payoff da janela de holdout;
- não pode usar a identidade da janela como feature;
- qualquer candidato precisa replicar one-position em todas as janelas
  restantes a 1 bps antes de virar candidato de shadow.

Sem Testnet/LIVE.
