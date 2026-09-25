# Swap real da Fusion (conta demo cTrader 10139135), lido em 2026-09-25 01:34 UTC

Lido por um cBot só de leitura (`backend/scripts/research/ctrader_cbots/SwapProbe.cs`, `Symbol.SwapLong/SwapShort`), rodado na demo pelo cTrader CLI com a senha por arquivo. Nenhuma ordem enviada.

| Símbolo | Comprado | Vendido | Unidade |
|---|---:|---:|---|
| US500, NAS100, US30, US2000 | −6,14 | +1,76 | % ao ano |
| GER40, FRA40, EUSTX50 | −4,66 | +0,42 | % ao ano |
| UK100 | −6,02 | +1,64 | % ao ano |
| JPN225 | −3,13 | −1,07 | % ao ano |
| AUS200 | −9,05 | +0,30 | % ao ano |
| HK50 | −5,09 | +0,72 | % ao ano |
| XAUUSD | −6,915 | +3,203 | pips por dia (pip 0,1) |
| XAGUSD | −1,11 | +0,01 | pips por dia (pip 0,01) |
| XTIUSD | +11,455 | −18,678 | pips por dia (pip 0,01) |
| XBRUSD | +3,133 | −6,761 | pips por dia (pip 0,01) |
| EURUSD | −0,634 | +0,252 | pips por dia |
| USDJPY | +0,635 | −1,625 | pips por dia |

Valores positivos são crédito; negativos, débito. SPA35 não existe nesta conta.

## Efeito em C5 (informativo, não muda o critério nem a decisão)

Refazendo C5 com o swap medido (comprado paga ~6% ao ano, vendido recebe ~1% ao ano) em vez do simétrico do pré-registro: TSMOM +0,90 ATR por operação nos índices já vistos (2000-2011, t ingênuo 2,0) e **+0,28 ATR nos índices novos (2000-2026, t ingênuo 1,1)**; RSI2_D1 +0,12 e +0,02. Continua positiva e continua sem significância nos mercados novos: não é aprovação.
