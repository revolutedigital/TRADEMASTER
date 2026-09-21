# Pré-registro, rodada 11: fluxo executado em mercado centralizado

Registrado em 2026-09-21 antes de baixar ou calcular os sete dias declarados. Pesquisa offline;
nenhuma chave, conta ou ordem é usada.

## Por que mudar a informação

O torneio FX esgotou preço, spread e top-of-book sem produzir expectativa positiva. O feed FX não
contém volume negociado nem lado agressor. Esta rodada testa uma informação genuinamente nova em um
mercado centralizado, sem expandir volume de dados antes de provar aprendibilidade.

## Fonte e amostra fixa

- arquivo público oficial Binance Vision, `BTCUSDT` spot, klines de 1 segundo;
- dias UTC 2026-09-13 a 2026-09-19;
- treino: dias 13–16;
- seleção: dia 17;
- auditoria final: dias 18–19;
- cada ZIP e seu `CHECKSUM` oficial devem ser verificados.

Os arquivos guardam volume, quote volume, quantidade de negócios e volume taker-buy. Isso permite
medir fluxo agressor aproximado. Não há bid/ask histórico nem livro de ofertas, portanto esta rodada
é um filtro de aprendibilidade e hurdle de custo, não backtest executável.

## Eventos, targets e custos-hurdle

Decisão a cada cinco segundos no fechamento de uma barra completa; entrada teórica no open do
segundo seguinte. Horizontes fixos `{5; 30; 120}` segundos e dois lados. Eventos com segundos
ausentes são inválidos.

Retorno bruto usa o preço futuro observado. São reportados retornos líquidos sob custos roundtrip
fixos `{5; 10; 20}` bps. Uma hipótese só pode passar se permanecer positiva no hurdle de 20 bps;
passar não afirma que esse seja o custo real da venue.

## Features causais

Para janelas `{5; 15; 60; 300}` segundos:

- retorno, range e volatilidade realizada;
- volume base, quote volume e número de negócios;
- fração taker-buy e desequilíbrio agressor em quote volume;
- aceleração de volume, negócios e fluxo curto/longo;
- posição do fechamento no range e distância do VWAP.

## Torneio

Para cada horizonte, entram as mesmas famílias auditáveis da rodada 10: caudas univariadas,
interseções das 30 melhores, árvores rasas, K-Means e rankings Ridge, Extra Trees,
HistGradientBoosting e XGBoost regressivo/classificadores. No máximo 25 regras por horizonte chegam
à auditoria.

A correção de Bonferroni considera todas as finalistas dos três horizontes em conjunto. Um candidato
exige ao menos 100 eventos na auditoria, presença nos dois dias, médias líquidas positivas com 5 e
20 bps e limites inferiores unilaterais corrigidos positivos nos dois custos.

Zero aprovados encerra o piloto. Um aprovado autoriza apenas repetir a regra congelada em mais dois
ativos e capturar/obter top-of-book para custo executável; nunca autoriza Demo ou LIVE.
