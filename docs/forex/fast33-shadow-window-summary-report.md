# Rodada 33 — agregação Fast24 + Fast28 + Fast29 + Fast32

Data: 2026-09-23  
Mercado: BTCUSDT futures  
Janelas: Fast24, Fast28, Fast29, Fast32  
Latência agregada: 100ms  
Resultado: a regra fixa segue reprovada como estratégia standalone.

## Fonte

Summary report:

`backend/data/microstructure_v1/reports/book-fixed-gate-shadow-window-summary-fast24-fast28-fast29-fast32-100ms/report.json`

Hash:

`da9a81100ef32e7cea5dda1f4869779b8fc40ba4300c062702aae01d07e8229a`

Status:

`SHADOW_WINDOW_SUMMARY_FAILED_PROFIT_COST_REPLICATION`

Motivos:

- `independent_has_non_positive_window_at_1bps`;
- `independent_has_non_positive_window_at_2bps`;
- `one_position_has_non_positive_window_at_1bps`;
- `one_position_has_non_positive_window_at_2bps`.

## Resultado agregado

| Modo | Janelas positivas em 1 bps | Trades exec. | Net médio 1 bps | Net médio 2 bps | Pior janela em 1 bps | Saídas |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| independente | 3/4 | 140 | +0,1727 | -0,8273 | -0,0705 | 51 TARGET / 89 TIMEOUT |
| uma posição | 3/4 | 42 | +0,3626 | -0,6374 | -1,6421 | 16 TARGET / 26 TIMEOUT |

## Decisão

Depois de quatro janelas prospectivas, a conclusão endureceu:

- média agregada em 1 bps ainda fica levemente positiva;
- mas consistência por janela falha;
- 2 bps continua completamente inviável;
- o filtro adicional escolhido em Fast31 falhou no holdout Fast32.

Não temos estratégia operável.

O caminho agora precisa mudar de “ajustar essa regra” para uma dessas duas frentes:

1. procurar uma família diferente de sinal;
2. transformar esta família em feature auxiliar dentro de modelo/ensemble, não regra standalone.

Sem Testnet/LIVE.
