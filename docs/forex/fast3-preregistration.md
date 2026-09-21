# Pré-registro, rodada 3: máquina de oportunidades curtas condicionais

Registrado em 2026-09-21, antes de calcular features, labels ou resultados desta rodada. O objetivo
é descobrir **interseções reproduzíveis de sinais** em vez de testar mais uma lista de estratégias
artesanais. Qualquer mudança posterior exige uma emenda prévia em
`docs/forex/fast3-registry.jsonl`, com motivo e novos hashes.

## Pergunta

Em dez pares líquidos de FX, existe informação disponível no fechamento de uma barra M5 que
identifique entradas compradas ou vendidas com duração esperada entre 15 minutos e 6 horas e valor
esperado positivo **depois** de spread real, comissão e slippage, também sob custo de estresse?

O produto procurado não é um oráculo que precise operar sempre. A terceira ação, **não operar**, é
parte central da política. O sistema deverá dizer qual combinação de condições produziu o sinal,
qual o valor esperado estimado, quão calibrada é a estimativa e sob quais regimes ela já falhou.

## Fronteira desta rodada

- Pesquisa offline somente. Não envia ordem e não altera o runner de Demo ou LIVE.
- Entrada e seleção são estudadas agora. Gestão dinâmica, trailing e extensão de vencedores serão
  uma rodada posterior, treinada apenas sobre trades gerados por uma entrada aprovada. Assim, uma
  busca não mascara a outra.
- Decisão no fechamento M5; execução simulada na abertura M5 seguinte. Nenhuma decisão subminuto.
- Contexto M1, M5, M15 e H1, sempre composto só por barras já fechadas.
- Horizontes fixos: H15, H60, H180 e H360 minutos.
- Pares: EURUSD, GBPUSD, AUDUSD, NZDUSD, USDCAD, USDCHF, USDJPY, EURJPY, GBPJPY e EURGBP.

## Dados e blindagem temporal

O dado é o M1 bid/ask já validado nas rodadas anteriores. Entram somente pares-mês marcados como
`included=True` nos manifestos commitados `fast2-data-manifest.csv` (2014-11 a 2018-11) e
`fast-data-manifest.csv` (2019-01 em diante). Um exemplo é descartado se qualquer barra necessária
ao lookback, entrada ou horizonte atravessar um buraco superior a cinco minutos.

- **DESENVOLVIMENTO:** 2014-11 a 2022-12. Esta amostra não é virgem; as rodadas 1 e 2 já produziram
  conhecimento agregado sobre ela. Serve para feature engineering, treino, calibração e seleção.
- **D3, holdout de desenvolvimento:** 2023-01 a 2024-08. Os meses 2023-03 a 2023-07 continuam
  excluídos pelo manifesto. Nenhum resultado de estratégia/modelo pode ser visto aqui antes de um
  artefato `model_locked` no registro.
- **S2, confirmação final congelada:** 2024-09 a 2026-08. O loader da rodada 3 deve recusar essa
  faixa em qualquer comando de desenvolvimento. Ela só pode ser aberta uma vez, por uma política
  que tenha passado D3 e esteja registrada em `development_report`.

Dentro do desenvolvimento, a validação é walk-forward expansiva: treino mínimo de 24 meses, teste
trimestral, purge de 6 horas antes de cada teste e embargo de 6 horas depois. Transformações,
imputação, calibração e escolha de hiperparâmetro são refeitas dentro de cada fold; nunca se ajusta
um scaler ou modelo no futuro do fold.

## Unidade de decisão e labels

Para cada fechamento M5 elegível são criadas duas linhas, uma por lado. A entrada é o próximo open
executável (ask para compra, bid para venda). O risco `1R` é o ATR verdadeiro médio das 24 barras
M5 fechadas, calculado no mid, limitado inferiormente a quatro vezes o custo-base estimado da
entrada e saída. Linhas sem 24 barras contínuas ou com R não positivo são descartadas.

Para cada horizonte e lado, o label guarda:

1. resultado terminal líquido em R, saindo no bid (compra) ou ask (venda);
2. MFE e MAE executáveis em R durante o horizonte;
3. se +0,5R, +1R ou +2R ocorreu antes de -1R; em empate intrabar, o stop vence;
4. resultado terminal sob `FUSION_ZERO` e sob `STRESS`.

Spread vem do bid/ask observado. Slippage usa exatamente `app.fx.sim.costs.prepare_run`. Comissão é
a mesma conta da rodada 2. Financiamento é zero porque H360 não atravessa mais de seis horas; eventos
que atravessariam o fechamento semanal são eliminados pela regra de continuidade.

## Features declaradas

Todas são adimensionais, calculadas no fechamento da decisão e sem backfill:

- retornos direcionais de 1, 3, 6, 12, 24 e 72 barras M5;
- inclinação preço/EMA e distância de médias exponenciais 6, 12, 24 e 72;
- range verdadeiro/ATR, corpo, pavios e posição do fechamento na barra;
- volatilidade realizada e razão de volatilidade nas janelas 6, 24 e 72;
- compressão: range das últimas 6, 12 e 24 barras dividido pelo ATR;
- distância dos máximos e mínimos móveis de 12, 24 e 72 barras;
- spread atual, percentil do spread e mudança do spread nas últimas 24 barras;
- pulso M1 das últimas 5 barras: retorno, range, desequilíbrio de corpos e spread;
- tendência e volatilidade dos contextos M15 e H1;
- minuto da semana em seno/cosseno e sessões Londres/Nova York;
- momentum contemporâneo do fator USD e dos pares correlatos, calculado somente com timestamps
  comuns e informação já fechada;
- identidade do par, lado e interações direcionais obtidas multiplicando features de preço pelo
  lado (+1/-1).

Não entram notícia, calendário econômico, volume sintético, dado fundamental ou feature escolhida
depois de olhar D3. O schema exato, nomes e fórmulas será versionado e hasheado antes de D3.

## Modelos e controles

- **B0:** não operar (retorno zero).
- **B1:** retorno incondicional por par, lado, horizonte e bloco horário; só opera se a média de
  treino for positiva. Mede se o ML apenas aprendeu horário.
- **B2:** regressão logística L2 para `P(R líquido > 0)` e regressão ridge para E[R].
- **M1:** XGBoost raso para E[R] e classificadores XGBoost para as barreiras. Profundidade máxima
  entre 2 e 5, learning rate entre 0,02 e 0,08, subsample e colsample entre 0,6 e 1,0, de 200 a
  1.200 árvores com early stopping. A busca tem no máximo 40 trials por horizonte.

São quatro candidatos B2 e quatro M1, um por horizonte. Um dataset pooled entre pares e lados dá ao
modelo suporte para aprender interações compartilhadas. Probabilidades são calibradas por isotonic
regression quando houver ao menos 1.000 exemplos por classe; abaixo disso, Platt scaling.

## Política de entrada

Em cada instante, cada modelo produz E[R], P(R>0), probabilidades de barreira e uma faixa de
incerteza. Os thresholds permitidos de E[R] são {0,03; 0,05; 0,08; 0,12; 0,18}; e de quantil mínimo
de confiança, {0,00; 0,02; 0,05}. A combinação é escolhida só nos folds internos. P(R>0) é exibida,
calibrada e auditada, mas não é um veto: uma distribuição com menos de 50% de acerto pode ter EV
positivo quando o ganho condicional é maior que a perda. Se compra e venda passarem, vence o maior
E[R] conservador.

Há no máximo uma posição por par. Ela fica aberta até o horizonte fixo; novos sinais do par são
ignorados. Essa política evita contar eventos sobrepostos como trades independentes. Os controles
recebem a mesma restrição.

## Critério de seleção antes de D3

Uma política só pode ser travada para D3 se, nas previsões walk-forward fora da amostra:

- média líquida > 0 em custo-base e estresse;
- limite inferior unilateral de 95% do bootstrap estacionário por dia FX > 0 no custo-base;
- ao menos 1.500 trades, 7 de 10 pares com média positiva e 60% dos trimestres positivos;
- erro de calibração absoluto médio <= 0,05 para P(R>0);
- supera B1 no mesmo conjunto de eventos;
- nenhum único par responde por mais de 35% do lucro líquido;
- Probabilidade de Backtest Overfitting (CSCV) <= 0,20 e Deflated Sharpe Ratio >= 0,95, contando
  todos os 8 candidatos, todos os thresholds e todos os 40 trials como tentativas.

Se mais de uma política passar, escolhe-se a maior média de estresse; desempates, nesta ordem:
menor drawdown, maior largura entre custo-base e zero, modelo B2 em vez de M1, horizonte menor.
Somente uma política pode ser registrada em `model_locked`.

## Critério de D3 e confirmação S2

D3 é executado uma vez. Aprovação exige: média > 0 em custo-base e estresse; limite inferior
unilateral de 95% > 0 no base; pelo menos 300 trades; pelo menos 6 pares positivos; 55% dos meses
positivos; nenhuma perda individual maior que 1,25R; e calibração <= 0,07. Falhou, não abre S2.

S2 repete exatamente a política travada, uma vez, sem retreino posterior a 2024-08. Exige os mesmos
critérios de D3, salvo mínimo de 250 trades. Passar S2 autoriza apenas uma proposta de canário Demo;
LIVE continua exigindo autorização explícita separada do Igor.

## Testes e auditoria obrigatórios

- teste de causalidade: alterar preços futuros não pode mudar nenhuma feature/decisão passada;
- teste de fill: entrada e saída usam o lado correto do bid/ask e o próximo open;
- teste adversarial de empate intrabar: stop vence;
- teste de gaps, purge, embargo e fronteiras D3/S2;
- teste de paridade entre cálculo batch e um `FeatureState` streaming;
- relatório por par, lado, trimestre, sessão, volatilidade, spread, custo e horizonte;
- SHAP agregado e regras de interação extraídas apenas como explicação, nunca como nova busca;
- cada execução real registra commit, hashes de dados/features/modelo e resultado no JSONL.

## Ordem

1. Commit deste pré-registro, registro e guardas.
2. Implementar fábrica de eventos/features/labels e validá-la apenas em dados sintéticos.
3. Materializar desenvolvimento; rodar walk-forward, baselines e busca; registrar cada tentativa.
4. Travar uma política ou encerrar a rodada sem candidato.
5. Abrir D3 uma vez. Só se aprovada, registrar o relatório.
6. Abrir S2 uma vez. Só depois discutir canário Demo. Nunca ativar LIVE nesta rodada.

## Emendas

- **2026-09-21, durante desenvolvimento, antes de travar modelo e sem abrir D3 ou S2:** removido o
  threshold absoluto de P(R>0) da política de entrada. Os quatro diagnósticos preliminares usaram
  os cortes originais {0,52; 0,55; 0,58; 0,62}; nenhum atingiu 300 trades no semestre de seleção e,
  por isso, 2022 não foi aberto em nenhum deles. A distribuição prevista revelou o erro conceitual:
  P(R>0) mede frequência, não valor esperado, e elimina legitimamente trades de payoff assimétrico.
  E[R] líquido continua com a grade pré-fixada; probabilidade continua calibrada, reportada e sujeita
  ao limite de erro, mas não veta uma entrada cujo E[R] conservador seja positivo. A emenda altera a
  metodologia a partir deste ponto e todo resultado anterior permanece identificado como diagnóstico.
