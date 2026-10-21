# C16 — status da captura pelo CLI

Data: 2026-09-29. O pré-registro original permanece congelado em `c16-preregistration.md`; a troca do transporte está documentada em `c16-cli-amendment.md` e registrada no `c16-registry.jsonl`.

**Correção importante:** a conclusão anterior de que o cTrader não estava acessível estava errada. O teste verificou apenas credenciais OAuth específicas da Open API e ignorou o cTrader CLI já autenticado no projeto. Uma leitura `price EURUSD` funcionou pela demo. O C16 agora captura preço e DOM via cBot read-only no CLI, sem depender dessas credenciais OAuth.

## Evidência técnica

| Evidência | Resultado |
|---|---|
| Build do cBot C16 | Compilou pela imagem oficial fixada do cTrader CLI |
| Demo CLI / cBot | Conectou, recebeu `Symbol.Tick` e níveis `MarketDepth` nos sete pares |
| Captura smoke (`run_id=797891a78cef45e998aa1d289b31ffa3`) | 113 eventos em 10 s após iniciar: 39 ticks e 74 snapshots DOM; 7/7 símbolos |
| Book populado | 66/74 snapshots tinham níveis dos dois lados |
| Replay | 30 amostras em 7 pares; 0 `book_eligible` porque a geração ainda não foi conferida na interface |
| Journal C13 | 817 sinais históricos; nenhum coincidiu com a janela deste smoke test |
| Bid/ask iguais | 58/66 snapshots populados; conferir na UI da mesma conta antes de liberar sinal |
| Ordens enviadas | 0 |

O smoke test prova que o caminho de captura existe e persiste dados. Não testa a hipótese H1, não mede edge e não é amostra de estratégia. Não houve ajuste de thresholds. A coleta também não é HFT: é um feed cTrader de corretora, com callbacks e latências de aplicação; não é livro global/consolidado nem execução de microssegundos.

## O que ainda não está aprovado

- Nenhum snapshot desta geração foi comparado com o DOM da interface. O replay deixa tudo provisório e exclui os sinais do gate.
- O spread bid=ask predominou no smoke. O valor bruto foi preservado; não presumimos que seja bug nem que represente custo zero correto. É preciso validar visualmente.
- O smoke cobriu apenas 10 segundos por rodada, suficiente para validar o transporte, não a H1. A H1 exige os mínimos originais de 300 sinais, 30 dias distintos, 90 dias corridos e 90% de labels completos. Comissão e slippage também não entram automaticamente nos labels existentes.
- Não há estratégia aprovada, autorização para alterar C13, autorização LIVE ou ordem enviada.

Próxima etapa de pesquisa: conferir os primeiros snapshots populados de cada par na UI cTrader demo e só então materializar a geração verificada; depois deixar `record-cli` rodar prospectivamente, sem reiniciar/mesclar sessões silenciosamente, até haver cobertura amostral suficiente.
