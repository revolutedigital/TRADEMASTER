# G0, confirmação em amostra nunca vista: sma_rsi e bollinger_reversion no diário

Gerado em 2026-09-20 01:01 UTC. Pares: AUDUSD, EURUSD, GBPUSD, NZDUSD, USDCAD, USDCHF, USDJPY. Dados: 2016-09-01 a 2023-08-31.
Configurações testadas: 2 (estratégias x timeframes), parâmetros de fábrica, sem otimização: toda a amostra é fora da amostra.
Custo base = interbank spread + IBKR-style commission (0.2 bp, $2 minimum) + 0.2 pip slippage. Carry/swap real não modelado no caso base.

## Critério de aprovação (calibrado por placebo)

O critério usado compara a MELHOR configuração real com a melhor configuração em dados embaralhados (sem vantagem por construção): só passa quem tiver valor-p ajustado <= 0.05, média R positiva, ao menos 4 de 7 pares positivos e 30+ trades.

## Veredito

**G0 NÃO PASSOU.** Nenhuma configuração se separa do que a melhor de 2 configurações faria em dados sem vantagem (limite t de 95% no placebo: 1.94, 300 embaralhamentos).

A mais próxima foi sma_rsi/1D: t = 0.41, valor-p ajustado = 0.402, média R = 0.019, 3/7 pares positivos.

Calibração (300 embaralhamentos): a regra escrita no plano (IC 95% nominal excluindo zero em 2+ pares) aprovou dado sem vantagem em 3 (1%) das rodadas; o limite por bootstrap com Bonferroni aprovou 4 (1.3%), conservador demais. Por isso o critério é a distribuição da melhor configuração: mediana t = 0.14, p90 = 1.56, p95 = 1.94, p99 = 2.70.

## Configurações, caso base

| Estratégia | TF | Trades | Média R | t | Valor-p ajustado | PF | Custo por trade (pips: spread + slippage + comissão) | Pares > 0 | Efeito mínimo detectável (R) | Passa |
|---|---|---|---|---|---|---|---|---|---|---|
| bollinger_reversion | 1D | 540 | 0.021 | 0.39 | 0.402 | 1.04 | 3.79 | 4/7 | 0.166 | não |
| sma_rsi | 1D | 628 | 0.019 | 0.41 | 0.402 | 1.04 | 3.67 | 3/7 | 0.140 | não |

## Maior média R: bollinger_reversion / 1D

| Par | Trades | Média R | Média pips líquidos | PF | Win rate |
|---|---|---|---|---|---|
| AUDUSD | 83 | 0.170 | 22.9 | 1.33 | 0.48 |
| EURUSD | 82 | -0.067 | -12.8 | 0.89 | 0.38 |
| GBPUSD | 77 | 0.023 | 16.4 | 1.04 | 0.43 |
| NZDUSD | 75 | 0.100 | 14.3 | 1.18 | 0.44 |
| USDCAD | 74 | -0.053 | -4.6 | 0.91 | 0.42 |
| USDCHF | 71 | -0.068 | -7.6 | 0.88 | 0.44 |
| USDJPY | 78 | 0.028 | 8.2 | 1.05 | 0.41 |

Por ano civil (estabilidade):

| Ano | Trades | Média R |
|---|---|---|
| 2017 | 70 | -0.071 |
| 2018 | 85 | -0.112 |
| 2019 | 76 | 0.047 |
| 2020 | 82 | 0.153 |
| 2021 | 87 | -0.081 |
| 2022 | 82 | 0.063 |
| 2023 | 58 | 0.201 |

Para confirmar um efeito de 0.021 R com 80% de poder e significância ajustada seriam necessários cerca de 33,372 trades, contra 540 nesta amostra (≈ 433 anos nesta cadência com estes 7 pares).

## Sensibilidade a custo, horário de entrada e carry (média R agregada)

| Estratégia | TF | base | ecn_raw | liquid_entry | stress | adverse_swap |
|---|---|---|---|---|---|---|
| sma_rsi | 1D | 0.019 | 0.019 | 0.027 | 0.002 | -0.021 |
| bollinger_reversion | 1D | 0.021 | 0.021 | 0.034 | 0.001 | -0.022 |

## Como ler

- Média R é o resultado médio por trade em múltiplos do risco (distância do stop, 2 ATR).
- t é a média dividida pelo erro padrão. O valor-p ajustado é a chance de a MELHOR das 2 configurações, em dados embaralhados, atingir esse t.
- Efeito mínimo detectável é o menor R médio que esta amostra separaria de zero (80% de poder). Se for maior que qualquer efeito plausível, o resultado é INCONCLUSIVO por falta de amostra, não prova de ausência de vantagem.
- O caso base não inclui carry: em pares como USDJPY o diferencial de juros pode somar ou subtrair pips por dia. Ver adverse_swap. Custo pesa pouco em estratégias lentas (poucos pips contra uma distância de stop de 50 a 150 pips): o que decide é a vantagem, não o custo.
- Spreads vêm do feed interbancário da Dukascopy. liquid_entry mede o efeito de executar em horário líquido; stress dobra o spread (conta de varejo padrão).
