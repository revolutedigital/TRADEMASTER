# Pré-registro, rodada 4: microestrutura de cotações subminuto

Registrado em 2026-09-21, antes de calcular qualquer feature, label ou resultado desta rodada.
Qualquer mudança posterior exige uma emenda prévia em `docs/forex/fast4-registry.jsonl`, com motivo
e novo hash deste documento. Esta rodada é pesquisa offline: não envia ordens e não altera Demo ou
LIVE.

## Pergunta

Em dez pares líquidos de FX, a sequência causal de atualizações bid/ask contém interseções de sinais
capazes de selecionar operações de 30 segundos a 10 minutos com valor esperado positivo depois de
spread observado, comissão e slippage, inclusive sob custo de estresse?

A hipótese não é que "mais resolução" crie edge por si só. O teste procura informação perdida nos
candles: intensidade e espaçamento das atualizações, qual lado da cotação muda, persistência,
abertura/fechamento do spread e aceleração da volatilidade. O sistema pode escolher não operar.

## Decisão sobre o feed da corretora

O desenvolvimento usa primeiro o histórico local já adquirido. Ticks da Fusion/cTrader não serão
baixados para escolher features, modelos ou thresholds. Somente depois de uma política passar D4 e
S3 será baixada **uma semana** disponível na cTrader, com a política completamente congelada. Essa
semana é um teste cego de portabilidade do feed, não uma nova amostra de ajuste. Se falhar, a política
é rejeitada ou uma nova rodada é pré-registrada; nenhum parâmetro pode ser corrigido olhando a semana.

O teste da corretora não autoriza Demo nem LIVE. LIVE continua exigindo autorização explícita e
separada do Igor.

## Dados e blindagem temporal

Fonte primária: arquivos mensais de ticks HistData, com timestamp, bid e ask, já presentes em
`backend/data/raw/histdata/zips`. Somente a fonte espelhada consistente a partir de 2019 entra nesta
rodada; o feed legado de 2014-2018 fica fora. O campo volume é zero e não será usado nem descrito como
volume negociado.

Entram apenas pares-mês com `included=True` no manifesto commitado
`docs/forex/fast-data-manifest.csv`. Cotações cruzadas (`ask < bid`) invalidam o mês. Atualizações
consecutivas com o mesmo bid e ask são colapsadas de forma estável. Um exemplo é inválido se o
lookback, a entrada ou a saída atravessar intervalo sem cotação superior a 120 segundos.

- **DESENVOLVIMENTO:** 2019-01 a 2022-12. Treino, engenharia declarada, calibração e seleção.
- **D4, holdout de desenvolvimento:** 2023-01 a 2024-08. Protegido até `model_locked`; meses
  excluídos pelo manifesto continuam excluídos.
- **S3, confirmação final:** 2024-09 a 2026-05. Protegido até D4 aprovado. Junho a agosto de 2026
  ficam fora porque o feed perdeu milissegundos durante junho, o que degrada features de intervalo.
- **SEMANA DA CORRETORA:** exatamente uma semana histórica da Fusion/cTrader, escolhida pela primeira
  semana completa que a API disponibilizar depois de S3 aprovado. Só pode ser aberta uma vez e não
  pode alterar a política.

Dentro do desenvolvimento, treino vai até 2020-12, calibração usa 2021-H1, seleção usa 2021-H2. O ano
de 2022 só pode ser aberto se a política satisfizer simultaneamente os gates de base e estresse na
seleção. Purge e embargo são de 30 minutos em todas as fronteiras.

## Unidade de decisão

Após remover duplicatas exatas, o primeiro evento elegível acontece depois de 512 atualizações. Um
novo evento é criado a cada 128 atualizações dentro do mesmo bloco contínuo. O evento pertence ao
último tick do bloco; uma entrada hipotética usa a primeira cotação **posterior** ao evento.

São criadas duas linhas direcionais por evento, comprada e vendida. A política final aceita no máximo
uma direção e mantém no máximo uma posição por par; eventos enquanto a posição estiver aberta são
ignorados na avaliação.

Horizontes terminais fixos: **30, 120 e 600 segundos**. A saída é a primeira cotação no ou depois do
prazo, desde que chegue em até 10 segundos; do contrário o label é inválido. Compra entra no ask e sai
no bid; venda entra no bid e sai no ask.

## Custos e unidade de risco

- **Base:** bid/ask observado, comissão `FUSION_ZERO` já usada nas rodadas anteriores e slippage por
  lado de `0,1 pip + 0,05 × range mid em pips das 256 atualizações anteriores`.
- **Estresse:** spread dobrado simetricamente, mesma comissão e slippage por lado de
  `0,3 pip + 0,10 × o mesmo range causal`.
- O risco `1R` é o maior entre o range mid das 256 atualizações anteriores e quatro vezes o custo
  completo base estimado (spread de entrada, dois slippages e comissão).
- Resultado terminal, MFE e MAE são calculados nos preços executáveis e armazenados em pips e R.

Swap é zero porque o horizonte máximo é dez minutos. Nenhum fill usa preço médio, candle ou dado
posterior disfarçado de feature.

## Features declaradas

Todas existem no instante do evento e são calculadas nos ticks anteriores ou no tick da decisão:

- spread atual em pips; média, desvio, mínimo, máximo e z-score em 64 e 256 atualizações;
- mudança do spread em 16, 64 e 256 atualizações;
- duração em segundos e intensidade de updates nas janelas 16, 64 e 256;
- aceleração da intensidade, `intensidade_16 / intensidade_256`;
- retorno do mid em pips nas janelas 16, 64 e 256;
- volatilidade realizada do mid nas mesmas janelas;
- range do mid em pips nas janelas 64 e 256;
- desequilíbrio de direção do bid, do ask e do mid nas janelas 16, 64 e 256;
- desequilíbrio de atividade bid versus ask e fração de atualizações conjuntas;
- comprimento e direção da sequência corrente do mid, limitados a 64 atualizações;
- segundo da semana em seno/cosseno e indicadores das sessões Londres e Nova York;
- identidade do par, lado e orientação pelo lado dos sinais direcionais.

Não entram volume, agressão compradora/vendedora, absorção, negócios executados, DOM histórico,
notícias ou feature criada após ver D4/S3. O nome "microestrutura" nesta rodada significa somente
microestrutura de **cotações top-of-book**.

## Modelos e tentativas

- **B0:** não operar.
- **B1:** média condicional por par, lado, horizonte, sessão e quintil de intensidade.
- **B2:** regressão ridge para E[R] base e estresse, com imputação e escala ajustadas apenas no treino.
- **M1:** XGBoost raso para E[R] base e estresse, profundidade {2, 3}, learning rate {0,03; 0,06},
  árvores máximas {300, 700}, subsample e colsample 0,8, com early stopping.

São três horizontes por B2 e três por M1. O score é o menor entre EV base e EV estresse. Thresholds
permitidos: {0,03R; 0,05R; 0,08R; 0,12R}. Toda combinação executada conta no PBO/DSR. Não haverá
busca de saída dinâmica nesta rodada; trailing só será estudado sobre entradas aprovadas.

## Gates antes de D4

Uma única política pode ser travada se, em 2021-H2 e depois no 2022 ainda fechado:

- média líquida positiva em base e estresse;
- limite inferior unilateral de 95% do bootstrap estacionário por dia FX acima de zero no base;
- pelo menos 1.500 trades, 7 de 10 pares positivos e 60% dos meses positivos;
- supera B1 nos mesmos eventos e custos;
- nenhum par responde por mais de 35% do lucro;
- PBO <= 0,20 e DSR >= 0,95 contando todas as tentativas.

Somente cumprir os gates em 2021-H2 autoriza abrir 2022. Cumprir ambos autoriza `model_locked`.

## D4, S3 e semana da corretora

D4 roda uma vez e exige: base e estresse positivos, limite inferior de 95% positivo no base, ao menos
300 trades, 6 pares positivos e 55% dos meses positivos. Se falhar, S3 continua fechado.

S3 repete a política congelada uma vez, sem retreino, e usa os mesmos gates com mínimo de 250 trades.
Se passar, a semana da Fusion/cTrader pode ser baixada e avaliada uma vez. Ela exige base positivo,
resultado realizado compatível com o intervalo previsto e nenhuma divergência estrutural de spread,
frequência de ticks ou cobertura que invalide as features. Falhar não permite ajuste retroativo.

## Testes obrigatórios

- causalidade: mudar qualquer tick futuro não altera feature passada;
- paridade entre cálculo batch e estado streaming;
- fill bid/ask no primeiro tick posterior e saída no prazo correto;
- colapso estável de duplicatas, rejeição de crossed quote e invalidação por gap;
- custos base/estresse, comissão e risk floor;
- fronteiras temporais e recusas de D4, S3 e semana da corretora;
- relatório por par, lado, mês, sessão, spread, intensidade, custo e horizonte;
- hashes de código, dataset, features e artefato registrados no JSONL.

## Ordem

1. Commit deste pré-registro, registro e guardas.
2. Implementar e testar fábrica causal somente com dados sintéticos.
3. Materializar desenvolvimento e registrar o run.
4. Rodar B0/B1/B2/M1; abrir 2022 apenas se 2021-H2 passar.
5. Travar uma política ou encerrar a rodada sem candidato.
6. Abrir D4 e, somente se aprovada, S3.
7. Somente depois de S3 aprovado, baixar e avaliar uma semana da Fusion/cTrader.
8. Propor trailing e canário Demo em rodada separada. LIVE permanece bloqueado.

