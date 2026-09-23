# Rodada 13 — eventos macro oficiais em M1 FX

Data: 2026-09-22  
Mercado: FX M1 bid/ask, 10 pares líquidos, calendário oficial 2019-01-01 a 2026-08-31  
Resultado: família calendário M1 rejeitada no P0

## O teste

Esta rodada testou se notícias macro oficiais, com horário verificável, geram drift ou reversão
curta o bastante para virar estratégia depois de spread, slippage e comissão. Foram usados apenas
eventos com fonte oficial e horário confiável: NFP, CPI US, FOMC, ECB e BoE. BoJ ficou fora porque
a fonte oficial não publica horário da decisão.

O sinal não interpreta o conteúdo da notícia. Ele mede o primeiro impulso no próprio par e testa
duas leituras mecânicas: continuar o movimento (`drift`) ou operar contra ele (`reversal`). A entrada
ocorre depois do caos inicial, com delays de 5, 10 ou 15 minutos. As saídas testam alvo fixo e
duas versões de trailing.

## Controle de qualidade

Foram processados 187.278 trades simulados em 108 hipóteses congeladas. Os candles impossíveis com
`ask < bid` foram removidos antes da preparação de eventos. Isso afetou só USDCAD: 23 candles
removidos em 2.796.052 candles brutos. Nenhum outro par teve quote cruzado.

| Par | Eventos preparados | Trades | Candles removidos |
|---|---:|---:|---:|
| AUDUSD | 234 | 21.852 | 0 |
| EURGBP | 121 | 10.680 | 0 |
| EURJPY | 57 | 4.878 | 0 |
| EURUSD | 295 | 27.114 | 0 |
| GBPJPY | 61 | 5.760 | 0 |
| GBPUSD | 295 | 27.456 | 0 |
| NZDUSD | 235 | 22.434 | 0 |
| USDCAD | 234 | 22.104 | 23 |
| USDCHF | 235 | 22.266 | 0 |
| USDJPY | 237 | 22.734 | 0 |

## Resultado

Todas as 108 hipóteses tiveram amostra suficiente na seleção. As 25 melhores foram auditadas no
período 2024-01-01 a 2026-08-31. Nenhuma passou.

O ponto mais importante: a melhor hipótese da seleção já era negativa no cenário stress. Ou seja, a
família nem chegou perto de pagar custo severo antes da auditoria.

| Etapa | Melhor regra | Trades | Base | Stress |
|---|---|---:|---:|---:|
| Seleção | `drift_thr2_d10_h120_fixed_1_5r` | 1.055 | +0,485 pips | -0,678 pips |
| Auditoria | mesma regra | 632 | -1,686 pips | -2,778 pips |

Na auditoria, o limite inferior corrigido por Bonferroni da melhor regra ficou negativo nos dois
cenários: -5,350 pips em base e -6,412 pips em stress. Não houve caso limítrofe: as 25 finalistas
ficaram com média stress negativa.

## O que isso demonstra

Essa linha não virou estratégia. Eventos macro realmente criam movimento e volatilidade, mas a regra
mecânica “espera o primeiro impulso e entra com drift/reversal + trailing” não separou trades bons
o bastante para pagar execução. O trailing também não salvou aqui: as melhores regras foram as de
alvo fixo, e mesmo elas quebraram na auditoria.

O aprendizado é parecido com BTC, mas por outro caminho: gestão de saída não cria edge quando a
entrada não tem vantagem líquida. O dado novo de calendário ajudou a testar uma tese plausível, mas
não produziu estratégia aprovada.

Nenhuma ordem foi enviada e este resultado não autoriza Demo, Testnet ou LIVE.

## Artefatos

- Relatório JSON: `backend/data/lab_fast13/calendar-m1-report.json`
  (`47f137466b51ca7102d03925a068ff570948b6fb92978287c8b5d193348296fe`)
- Trades: `backend/data/lab_fast13/calendar-m1-trades.parquet`
  (`81f8ac1fedfe24b7404c967ce019ec3feaea32ea8616ab45be8e7af85f8dce29`)
- Seleção: `backend/data/lab_fast13/selection-metrics.csv`
  (`d7088da9165465581cf9f676ecb1a5666c8589e92792608f800f9ccd3c1816e9`)
- Auditoria: `backend/data/lab_fast13/audited-finalists.csv`
  (`2e4198e7a653444bc934caf029f4c64e6613e734630ca259c02604c5d975feeb`)
- Qualidade por par: `backend/data/lab_fast13/pair-quality.json`
  (`061d1035455fe03a0092fbb636b19c512a38fa63867473298897c71f17ad400f`)
