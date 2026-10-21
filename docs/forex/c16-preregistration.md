# C16 — book do cTrader como confirmação de sinais C13

Pré-registro: 2026-09-29, antes de iniciar a captura desta rodada. **C15 já está ocupado** pelo estudo FOMC/VIX de 2026-09-27; seus arquivos e resultados permanecem intactos. Este experimento usa C16.

## Pergunta

O Nível II e as cotações do feed cTrader acrescentam informação útil para selecionar os sinais que o C13 já gera em FX? C16 é uma camada de observação e pesquisa; não altera o modelo, as entradas, a gestão ou as ordens do C13.

## Fonte e limitações dos dados

- Feed: `ProtoOASpotEvent` e `ProtoOADepthEvent` do cTrader Open API, por conta e corretora escolhidas.
- A comparação com o journal `data/c13/demo_journal.jsonl` usa somente captura `demo`; captura `live` fica fora desta H1 e exige pré-registro separado para não misturar ambientes.
- Universo inicial: EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD e NZDUSD; um símbolo ausente na conta é erro explícito, não é substituído silenciosamente.
- O depth é **o book fornecido por essa corretora**, não um book consolidado do mercado OTC de FX nem fita de negócios executados.
- A API documenta deltas `newQuotes`/`deletedQuotes`, mas não documenta histórico de depth nem garante no evento uma fotografia inicial completa. A coleta é prospectiva. Até comparar a primeira fotografia de **cada geração de conexão** com a interface cTrader da mesma conta/símbolo, os snapshots dessa geração ficam `provisional` e não são elegíveis para sinal. A geração 1 pode ser liberada com `--initial-snapshot-verified`; outras exigem `--verified-generations` com os números conferidos.
- O cliente identifica cada conexão; o replay descarta o book anterior ao reconectar. O protocolo não fornece sequência para detectar toda perda silenciosa upstream. O WAL local é durável e para em overflow, mas isso não prova completude do feed do broker.

## Dados e causalidade

- Cada evento guarda horário de recebimento local, monotônico, geração de conexão, sequência local, símbolo e payload de mercado em unidades brutas do protocolo. Credenciais e identificadores pessoais não são gravados.
- O replay publica no máximo um snapshot por segundo por par. Features: spread e mid do spot; spread de book; profundidade bid/ask e desequilíbrio nos 1/5/10 níveis; deslocamento do microprice; mudanças de desequilíbrio e contagens de cotações adicionadas/removidas em janelas de 1/5/30 segundos.
- “Adicionada/removida” descreve atualização de cotações, **não** agressões, ordens executadas ou fluxo consolidado de negócios.
- O join escolhe o último snapshot recebido **até** o instante do sinal C13. Book ausente, provisório, cruzado ou com idade acima de 1.000 ms não passa pelo gate.
- A rotulagem sombra entra 100 ms após o sinal; se não houver cotação bid/ask fresca em até 2.000 ms, o caso é inválido. O retorno usa preço executável: compra no ask e saída no bid; venda no bid e saída no ask. O arquivo guarda retorno, MFE e MAE em 30, 120 e 300 segundos.
- Esses retornos já pagam o spread observado, mas **não** incluem comissão nem slippage. O cenário de estresse subtrai mais um spread de entrada e de saída (spread total duplicado); ainda assim, não substitui medir comissão/slippage nas execuções demo. Nenhum resultado deste rótulo, sozinho, aprova operação.

## Hipótese e julgamento

- **H1 primária:** entre sinais C13 `take=true` com book fresco e conferido, confirmação direcional simultânea (`directed_depth_imbalance_5 > 0` e `directed_microprice_displacement_pips > 0`) melhora o retorno líquido de spread em 120 segundos contra os mesmos sinais C13 `take=true` no subconjunto de cobertura elegível, sem filtro.
- Requisitos mínimos antes de julgar: 300 sinais elegíveis, 30 dias distintos, pelo menos 90 dias corridos de coleta e pelo menos 90% de labels completos no subconjunto analisado. O relatório deve incluir comissão/slippage demo quando disponíveis e o estresse de spread duplicado.
- H1 só sobrevive se o subconjunto filtrado tiver média positiva após spread duplicado, superar o baseline C13 em retorno médio de 120 s e a diferença pareada por dia tiver p unilateral ≤ 0,05. Menos que os mínimos é **inconclusivo**, não aprovado.
- Horizontes de 30/300 s, níveis alternativos, limiares diferentes, stops/alvos/trailing e qualquer modelo calibrado são exploratórios até terem novo pré-registro e holdout cronológico intocado. Não escolher o melhor corte olhando o mesmo holdout.
- Uma eventual sobrevivência gera apenas uma segunda rodada de validação prospectiva, em demo/sombra. C16 nunca habilita ordens, e não autoriza alterar o runner C13 nem operar LIVE.

## Sequência operacional

1. De `backend/`, rodar `python -m scripts.research.fx_c16_microstructure record --environment demo --symbols EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD`. Para um preflight curto, acrescentar `--duration-seconds 600`; sem duração, encerra com Ctrl+C. O script não envia ordens.
2. Guardar o `run_id` impresso. Comparar a primeira fotografia de cada geração capturada com o DOM da mesma conta/símbolo. O resumo da materialização informa as gerações vistas. Sem correspondência observável, deixar aquela geração provisória.
3. Materializar com `materialize --run-id RUN_ID --initial-snapshot-verified` somente depois de conferir a geração 1; para outras gerações, usar `--verified-generations 1,2` apenas se ambas foram conferidas. Fazer `join-c13`, `label-c13` e `evaluate` com arquivos de saída novos; os comandos recusam sobrescrever artefatos existentes.
4. Só depois do mínimo amostral, calcular o julgamento definido acima. Até lá: nenhum edge alegado.
