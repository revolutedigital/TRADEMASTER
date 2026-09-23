# Pré-registro, rodada 13: eventos macro oficiais em M1 FX

Registrado em 2026-09-22 antes de calcular qualquer PnL desta família. Pesquisa offline; nenhuma
chave, conta, banco de produção ou ordem é usado.

## Pergunta

As rodadas de preço/volume agregado com trailing não encontraram separação suficiente para pagar
custo. Esta rodada testa uma informação diferente: horários oficiais de eventos macro com choque
observável no M1. A hipótese é que, depois do primeiro impulso da notícia, alguns pares carregam
drift ou reversão por tempo suficiente para pagar spread, slippage e comissão.

## Universo fechado

- calendário: `backend/data/raw/calendar/events.parquet`;
- eventos permitidos: `NFP`, `CPI US`, `FOMC rate decision`, `ECB rate decision`,
  `BoE rate decision`;
- `BoJ rate decision` fica excluído porque a fonte oficial não publica horário confiável;
- candles: `backend/data/raw/fx_m1_hd/*_M1_*.parquet`, bid/ask M1 da Dukascopy;
- pares por moeda do evento:
  - USD: `EURUSD`, `GBPUSD`, `AUDUSD`, `NZDUSD`, `USDCAD`, `USDCHF`, `USDJPY`;
  - EUR: `EURUSD`, `EURGBP`, `EURJPY`;
  - GBP: `GBPUSD`, `EURGBP`, `GBPJPY`.

O lado da operação é definido pelo próprio impulso observado no par, não por interpretação humana
do conteúdo da notícia. Se o preço do par sobe no impulso, `drift` compra e `reversal` vende; se
cai, os lados são invertidos.

Candles impossíveis com `ask < bid` em qualquer OHLC são removidos antes da preparação de eventos,
com contagem reportada por par. Eles não podem ser usados para entrada, saída, impulso ou baseline.

## Features causais

Para cada evento/par:

1. medir o impulso dos primeiros 5 candles a partir do minuto oficial do evento;
2. calcular a faixa mediana dos 240 minutos anteriores, ignorando os 10 minutos imediatamente
   anteriores ao evento;
3. só permitir trade se `abs(impulso) >= threshold * faixa_mediana`;
4. entrar apenas depois do caos inicial, no open de `evento + delay`.

Nenhuma feature pode usar candles depois do horário de entrada.

## Grade congelada

Foram congeladas 108 hipóteses:

- direção: `drift`, `reversal`;
- threshold do impulso: `1.5`, `2.0`;
- delay de entrada: `5`, `10`, `15` minutos;
- horizonte máximo: `30`, `60`, `120` minutos;
- gestão:
  - `fixed_1_5r`: stop inicial e alvo de 1,5R;
  - `trail_1r_0_75r`: stop inicial, ativa no 1R e traila a 0,75R;
  - `trail_0_75r_0_5r`: stop inicial, ativa no 0,75R e traila a 0,5R.

O risco inicial `R` é `max(1.0 * faixa_mediana, 0.5 * abs(impulso))`, com piso de 0,5 pip.

## Execução simulada e custos

Entradas e saídas usam o lado executável:

- long entra no `ask_open` e sai no `bid`;
- short entra no `bid_open` e sai no `ask`;
- se stop e alvo/trailing cabem no mesmo candle, a ordem intrabar é pessimista e o stop vence;
- se o open atravessa o stop, sai no open piorado, não no preço ideal do stop;
- sem stop/alvo, sai no close do horizonte máximo.

São reportados dois cenários:

- base: spread real + slippage de 0,1 pip por lado + comissão Fusion de US$ 2,25 por lote por lado;
- stress: spread dobrado simetricamente + slippage de 0,3 pip por lado + mesma comissão.

## Seleção e auditoria

- seleção: eventos até 2023-12-31;
- auditoria única: eventos de 2024-01-01 até 2026-08-31;
- cada hipótese exige pelo menos 120 trades na seleção;
- as 25 melhores por média stress, desempate por média base e tamanho de amostra, são auditadas;
- auditoria exige pelo menos 40 trades e 25 eventos oficiais distintos;
- o limite inferior unilateral usa clusters por timestamp oficial do evento;
- Bonferroni corrige as 25 finalistas;
- para passar, média e limite inferior devem ser positivos em base e stress.

Se nenhuma hipótese passar, esta família é rejeitada no P0. Se alguma passar, ela autoriza somente
expansão offline e estudo de portabilidade; não autoriza Demo, Testnet ou LIVE.
