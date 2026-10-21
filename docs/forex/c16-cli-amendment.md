# C16 — adendo de captura pelo cTrader CLI

Data: 2026-09-29. Este adendo preserva `c16-preregistration.md` sem alterações. Foi documentado após os smoke tests técnicos e antes de qualquer rotulagem/avaliação da hipótese. Corrige somente o transporte de dados: a hipótese H1, os pares, o join causal, os rótulos, os custos, as janelas e os gates estatísticos originais continuam congelados.

## Correção do preflight

O preflight anterior verificou apenas as variáveis OAuth do coletor Open API e descreveu incorretamente que o cTrader não estava acessível. O projeto já tinha um fluxo cTrader CLI autenticado e quatro runners de demo ativos. Uma consulta `price EURUSD` pelo script já existente respondeu sem enviar ordem. A captura C16 foi então migrada para um cBot read-only executado por esse CLI; não exige credenciais de aplicativo Open API.

## Fonte e semântica

- O cBot assina `Symbol.Tick` para obter bid/ask e `MarketDepth.Updated` para ler snapshots completos de `MarketDepth.BidEntries` e `AskEntries`; o cTrader CLI transmite a saída do cBot ao processo de captura. Referências oficiais: [Symbol.Tick](https://help.ctrader.com/ctrader-algo/references/MarketData/Symbols/Symbol/), [MarketDepth](https://help.ctrader.com/ctrader-algo/references/MarketData/MarketDepth/MarketDepth/) e [execução de cBots pelo CLI](https://help.ctrader.com/ctrader-cli/cbots/).
- A captura limita a 20 níveis por lado; as features pré-registradas usam no máximo 10. Os callbacks de DOM fornecem snapshots, não IDs de cotação/deltas de adição e remoção. Por isso as métricas de quote-add/cancel ficam `null` nesta fonte — nunca zero inventado — e não entram na H1.
- Ticks e snapshots guardam horário de recebimento e relógio monotônico no host, além do timestamp de origem do cBot. O book só é elegível quando há níveis dos dois lados, não está cruzado e passa pelos gates de idade e verificação.
- O cBot tem `AccessRights.None` e não chama API de ordem. Ainda assim, a própria imagem CLI alerta que Linux não fornece sandbox de Algo; portanto só deve ser executado este cBot de código revisado, que não chama ordens nem usa acesso a arquivos ou rede. A captura fixa a conta demo já usada pelo CLI, monta o arquivo privado de senha somente para leitura e recusa colisão de nome de container. Desconexão do servidor encerra o cBot e falha a captura; um reinício cria outro `run_id`.
- O primeiro estado pode vir vazio. Compare o primeiro snapshot **populado** de cada par com o DOM da interface cTrader da mesma conta antes de liberar a geração 1 em `materialize`. Até isso acontecer, ela permanece provisória e não produz `book_eligible`.
- Spreads bid=ask observados são mantidos literalmente; spread zero não é classificado como mercado cruzado. A captura não tem autoridade para corrigir o feed. Confirme esses casos na interface antes de usar qualquer sinal. Comissão e slippage continuam fora dos labels conforme o pré-registro.

## Smoke test registrado

Em demo, `record-cli --duration-seconds 10` concluiu com `run_id=797891a78cef45e998aa1d289b31ffa3`, **113 eventos**: 39 ticks e 74 snapshots de DOM nos sete pares; 66 snapshots tinham níveis dos dois lados. Nenhuma ordem foi enviada. A materialização criou 30 amostras de replay, todas provisórias e portanto **0 elegíveis**. Os 817 sinais históricos do journal C13 ficaram fora da janela curta e não tiveram book associado. Em 58 dos 66 snapshots populosos, o bid e ask observados eram iguais; esse achado é diagnóstico do feed, não evidência de edge, e exige conferência visual.

Este smoke test valida o caminho técnico de captura, não o sinal, a completude de 90 dias, nem a estratégia. A hipótese não foi avaliada e nenhum threshold foi ajustado a partir desta amostra.

## Operação

De `backend/`:

```bash
python -m scripts.research.fx_c16_microstructure record-cli --duration-seconds 600
```

Sem `--duration-seconds`, a captura permanece ativa até Ctrl+C. Depois, aplicar os mesmos passos `materialize`, `join-c13`, `label-c13` e `evaluate` do pré-registro, sempre com um novo arquivo de saída. Não usar `--initial-snapshot-verified` até comparar o primeiro DOM populado de cada par com a interface.
