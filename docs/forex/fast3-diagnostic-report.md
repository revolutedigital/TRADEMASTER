# Rodada 3 — relatório diagnóstico da máquina de trades curtos

Data: 2026-09-21. Estado: **desenvolvimento; nenhum modelo travado; D3 e S2 fechados**.

## O que foi construído

- Painel causal de 29.596.992 barras M1 e 5.950.410 decisões M5, em 10 pares, de 2014-11 a
  2022-12, respeitando os meses excluídos dos manifestos já commitados.
- 48 features disponíveis no fechamento da decisão: preço/volatilidade/compressão/spread, pulso M1,
  contexto M15/H1, sessões e fatores cross-pair.
- 112 outcomes por decisão: quatro horizontes, dois lados, custo-base e estresse, MFE/MAE e três
  saídas stop/alvo. Fill sempre no próximo open executável; gaps invalidam o evento.
- Guardas append-only impedem abrir D3 (2023-01 a 2024-08) antes de `model_locked` e S2
  (2024-09 a 2026-08) antes de D3 aprovado. Nenhum dos dois foi aberto.
- Diagnóstico XGBoost pooled entre pares/lados, treinado até 2020, calibrado em 2021-H1 e selecionado
  em 2021-H2. O treino usa aproximadamente 1 milhão de linhas determinísticas por candidato; a
  avaliação usa o semestre completo sem amostragem.

## Controle incondicional

Operar cada barra sem condicionamento perde cerca de -0,25R no custo-base e de -0,47R a -0,48R no
estresse. Isso é coerente com spread + comissão + slippage e confirma que o pipeline não fabrica
lucro gratuito.

| horizonte | long base | short base | long estresse | short estresse |
|---:|---:|---:|---:|---:|
| 15m | -0,252 | -0,253 | -0,468 | -0,469 |
| 60m | -0,255 | -0,256 | -0,474 | -0,475 |
| 180m | -0,259 | -0,256 | -0,480 | -0,477 |
| 360m | -0,263 | -0,251 | -0,485 | -0,473 |

## Diagnóstico 1: saída terminal

O modelo estima separadamente EV-base e EV-estresse; o score é o menor dos dois. Nenhum horizonte
passou os gates. H15, H60 e H180 foram claramente negativos. H360 chegou a parecer quase viável
quando treinado só no custo-base (134 trades, +0,205R base, 8 pares), mas ficou -0,010R no estresse e
os meses alternaram ganhos e perdas grandes. Quando o próprio treino passou a exigir EV-estresse,
o quase-sinal desapareceu: 91 trades no corte mínimo, -0,035R base e -0,233R estresse.

A região H360 era dominada por interações de hora da semana, tendência H1, retorno de 2h, fator
comum/USD, posição versus EMA curta e sessão de Londres. Ela não era estável: setembro/dezembro
ganharam muito; julho/agosto/novembro perderam; alguns pares ganharam enquanto JPY destruiu o
resultado. Isso é regime, não uma regra transportável.

## Diagnóstico 2: stop 1R + alvo + timeout

Foram declarados antes do resultado 12 templates: horizontes {15, 60, 180, 360} minutos × alvos
{0,5R; 1R; 2R}. Stop vence empate intrabar, paga slippage e comissão; alvo-limite paga comissão;
timeout sai no preço executável. Cada candidato treinou EV-base e EV-estresse.

Resultado: **0/12 produziu uma única entrada com score conservador ≥ 0,03R em 2021-H2**. O maior
EV-estresse previsto de cada família ainda foi negativo:

| horizonte | alvo 0,5R | alvo 1R | alvo 2R |
|---:|---:|---:|---:|
| 15m | -0,287 | -0,176 | -0,053 |
| 60m | -0,210 | -0,221 | -0,207 |
| 180m | -0,133 | -0,139 | -0,157 |
| 360m | -0,124 | -0,124 | -0,130 |

Como nenhum candidato passou seleção, 2022 não foi aberto nesta família.

## Defeitos encontrados e fechados

1. Features pandas foram inicialmente alinhadas por RangeIndex contra DatetimeIndex e viraram NaN.
   O piloto real detectou; o teste passou a exigir valores finitos e a fábrica usa arrays posicionais.
2. O primeiro diagnóstico exigia P(R>0) > 52%, confundindo taxa de acerto com valor esperado. A
   emenda removeu o veto; probabilidade continua calibrada e auditada.
3. Uma versão do diagnóstico abria o split seguinte apenas por atingir 300 trades, mesmo com EV
   negativo. Um teste de regressão passou a exigir simultaneamente base, estresse, pares e meses.
   Isso abriu 2022 uma vez para H180 dentro do próprio desenvolvimento; o trecho é desenvolvimento
   contaminado e nunca será apresentado como confirmação. D3 e S2 não foram tocados.

## Conclusão e próximo experimento válido

Esta rodada não sustenta “não existe padrão”. Ela sustenta algo mais específico: **OHLC bid/ask +
features técnicas multi-timeframe não venceram o custo de varejo com saída terminal nem com os 12
stops/alvos fixos testados**. Continuar ajustando threshold nessa mesma representação seria
overfitting.

O próximo salto precisa acrescentar informação, não parâmetros: eventos de microestrutura
(agressão, volume, desequilíbrio, absorção, spread/DOM e reação a eventos) e amostragem por mudança
de estado, com histórico que permita replay. A arquitetura causal, labels, custos, guardas e
execução desta rodada são reutilizáveis. Trailing só será treinado depois que uma entrada tiver EV
positivo fora da amostra.
