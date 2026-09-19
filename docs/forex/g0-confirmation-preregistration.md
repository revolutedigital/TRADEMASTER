# Registro prévio: confirmação do G0 em amostra nunca vista

Registrado em 2026-09-19, **antes** de baixar ou olhar qualquer dado de 2016-09 a 2023-08. Este documento é commitado antes de a análise existir; qualquer mudança depois disso só vale se for correção de defeito comprovado, descrita no relatório, e nunca uma mudança de critério.

## Por que existe

O G0 na amostra de descoberta (2023-09 a 2026-08) não passou, mas deixou dois candidatos diários com média positiva e estável entre os anos:

| Configuração (diário, caso base) | Trades | Média R | t | Valor-p ajustado (12 configurações) |
|---|---|---|---|---|
| sma_rsi | 210 | +0,196 | 2,29 | 0,175 |
| bollinger_reversion | 227 | +0,129 | 1,58 | 0,450 |

Escolher os dois melhores de 12 depois de ver o resultado é a receita clássica de achar vantagem que não existe. O único jeito honesto de saber é testá-los, sem mudar nada, em dados que o teste nunca viu. Só 2 hipóteses entram, então o peso de múltiplos testes cai.

## Hipóteses (fixadas)

- **H1:** `sma_rsi`, timeframe diário (barras de 17h Nova York), parâmetros de fábrica, tem expectativa positiva depois do custo.
- **H2:** `bollinger_reversion`, timeframe diário, parâmetros de fábrica, tem expectativa positiva depois do custo.

Definições exatas no código: `STRATEGIES` em `backend/scripts/research/fx_spike.py`, desenho `CONFIRMATION`. **Nenhum parâmetro muda.** Mesmos 7 pares, mesmo simulador (stop de 2 ATR, alvo de 2R, entrada na abertura da barra seguinte), mesmo modelo de custo do caso base (spread real do feed, comissão 0,2 bp com mínimo de US$ 2, slippage de 0,2 pip).

## Amostra

- 2016-09-01 a 2023-08-31, 7 majors (EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCAD, USDCHF, USDJPY), fonte Dukascopy.
- Não se sobrepõe à amostra de descoberta, que começa em 2023-09-01.
- As primeiras 120 barras servem de aquecimento dos indicadores e não geram trades.

## Critério de aprovação (fixado)

Uma hipótese é **confirmada** se, no caso base:

1. o valor-p ajustado for **≤ 0,05**, comparado com a distribuição da melhor das **2** configurações em ≥ 200 embaralhamentos da própria amostra de confirmação (mesmo procedimento do G0 original);
2. a média R agregada for **positiva**;
3. pelo menos **4 de 7 pares** tiverem média R positiva;
4. houver **30 trades ou mais**.

Se H1 e H2 falharem, o G0 fica reprovado e a rota de análise técnica em forex é encerrada. Se ao menos uma for confirmada, o G0 é considerado aprovado para essa configuração e o plano segue do step 9, **usando somente a configuração confirmada**.

## O que se espera, antes de ver o dado

- Vantagem escolhida por ter sido a melhor de 12 costuma encolher ("maldição do vencedor"). Um efeito verdadeiro de metade do observado (~0,10 R) daria t próximo de 1,8 em ~7 anos: **provavelmente não confirmaria**. Um efeito de ~0,20 R daria t próximo de 3,5.
- Portanto uma reprovação é o desfecho mais provável e não deve ser lida como falha do teste.
- A informação de swap/carry não entra no critério. Ela aparece só como sensibilidade.

## O que NÃO será feito

- Ajustar parâmetros, timeframes, pares, custos ou o critério depois de ver o resultado.
- Somar as duas amostras para "resgatar" uma confirmação que falhou. Eventual estimativa combinada seria apenas informativa.
- Testar outras estratégias nesta amostra. Se surgir outra ideia, ela precisa de amostra nova.
