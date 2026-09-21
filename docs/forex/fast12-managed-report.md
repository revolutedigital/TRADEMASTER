# Rodada 12 — sinal + breakeven + trailing

Data: 2026-09-21  
Mercado: BTCUSDT spot, barras de 1 segundo  
Resultado: arquitetura gerenciada rejeitada no P0

## O teste que faltava

Esta rodada testou diretamente a tese de entrada probabilística com proteção de caminho. Foram
reconstruídas as 75 entradas congeladas da rodada 11 e combinadas com sete gestões de stop inicial,
breakeven líquido e trailing. O horizonte máximo foi ligado ao horizonte do sinal: 30, 120 ou 300
segundos.

Foram avaliadas 525 combinações. A gestão foi selecionada em 18–19/09 e as 25 melhores foram
congeladas antes da auditoria de 20/09. O arquivo novo de 20/09 passou pelo checksum oficial
`9f2e4e6f9258d6eaccc8e261392f2d1a8456de4fc046447b9ccfaa9efb31b8ec`.

## Resultado

Nem a melhor das 525 combinações ficou positiva na seleção sob 20 bps de custo. Na auditoria nova,
as 25 finalistas ficaram negativas nos dois cenários de custo.

| Etapa | Melhor regra observada | Eventos | 5 bps | 20 bps |
|---|---|---:|---:|---:|
| Seleção | Extra Trees 1%, stop 5 / ativa 3 / trail 2 | 389 | -4,149 bps | -17,825 bps |
| Auditoria | Extra Trees 2%, stop 5 / ativa 3 / trail 2 | 320 | -3,994 bps | -19,301 bps |

O melhor limite inferior corrigido no cenário severo foi -20,502 bps. Não houve caso limítrofe:
zero das 25 regras auditadas teve média positiva com 5 bps e zero teve média positiva com 20 bps.

## O que isso demonstra

O trailing funcionou mecanicamente: limitou perdas, levou o piso líquido a zero quando houve
excursão suficiente e carregou movimentos favoráveis. O problema foi anterior a ele. Os sinais
congelados não separaram caminhos com excursão direcional suficiente para pagar a entrada e saída.
O melhor desempenho bruto ficou na ordem de aproximadamente 1 bps por evento; o custo otimista já
era 5 bps.

Portanto, aumentar a exposição ou ajustar mais o trailing sobre o mesmo sinal não cria edge. Isso
só muda a distribuição de uma expectativa que continua negativa. O próximo avanço válido precisa
de informação ausente nas barras de 1 segundo: sequência negócio a negócio, spread bid/ask real,
profundidade e desequilíbrio do livro, liquidações ou relação cross-venue. Repetir combinações de
preço, volume agregado e trailing está encerrado no P0.

Nenhuma ordem foi enviada e este resultado não autoriza Demo ou LIVE.

Artefato auditado: `2d2aa2a4ad6c195a2150ffc4c0ae09e8b1c6746440bb2c6b8ea9fb4480905ff0`.
