# ADR 0001: motor do bot de forex, núcleo próprio compilado

Data: 2026-09-20. Status: aceita (step 3 do plano `docs/plans/forex-fast-bot.md`).

## Contexto

Precisamos de (a) um laboratório que teste várias famílias de estratégia em anos de dados de 1 minuto de vários pares e ainda rode centenas de embaralhamentos de placebo, e (b) um runner que opere numa corretora com API. A decisão era adotar um motor aberto pronto (NautilusTrader) ou construir um núcleo próprio pequeno.

## Decisão

**Núcleo próprio.** O laboratório roda sobre um simulador compilado com numba; o runner é um processo asyncio em Python. O NautilusTrader fica como candidato só para uma futura rota Interactive Brokers, onde ele tem adaptador oficial.

## Evidência (medida, 2026-09-20, 12 núcleos)

| | Núcleo numba (`engine_spike/numba_core.py`) | NautilusTrader 1.231 | Referência Python | Simulador atual (pandas) |
|---|---|---|---|---|
| 1 configuração, 1 núcleo | **246 milhões de barras/s** (3,67 M de barras em 0,015 s) | 0,11 milhão de cotações/s (32.658 em 0,29 s) | 0,31 milhão/s | ~7 mil/s |
| 480 configurações em 12 núcleos | **1,15 bilhão de barras/s** (1,5 s) | não medido | , | , |

- **Correção:** o núcleo compilado concorda com uma referência independente em Python puro, trade a trade (mesmas entradas, saídas, preços e motivos de saída), em 20 combinações de séries aleatórias e parâmetros (26 testes em `backend/tests/unit/test_engine_spike_numba.py`). Além disso, valem invariantes verificados sem a referência: um stop perde exatamente 1 R mais o slippage, um alvo ganha exatamente 2 R e um stop atravessado por gap perde mais que 1 R.
- **Escala:** placebo de 200 embaralhamentos × 7 famílias × 26 milhões de barras (10 pares × 5 anos de M1) = ~36 bilhões de passos. No núcleo próprio isso leva minutos; no Nautilus, ~90 horas de núcleo.
- **Adaptador:** o NautilusTrader 1.231 tem adaptadores oficiais de Interactive Brokers, Binance, Bybit, Kraken, OKX, Databento e outros, **mas nenhum de cTrader**. Existe um adaptador comunitário de cTrader em pré-alfa (só a camada de transporte; faltam provedor de instrumentos e clientes de dados e de execução; sem release). Adotar o Nautilus para a cTrader significa escrever a parte mais difícil.
- **Instalação:** o Nautilus e o numba instalam e rodam no Python 3.13 do projeto (o Nautilus em ambiente separado, fora dos requisitos).
- **Confirmações da pesquisa independente (verificada por um cético):** na v2 (Rust, ainda release candidate) **não existe caminho suportado para um adaptador externo**: a subclasse Python de cliente de dados e de execução da v1 não tem equivalente, e só entram adaptadores compilados dentro do próprio projeto; o selo "stable" do adaptador de Interactive Brokers é do README, e em 2026-09-20 havia 11 issues abertos com esse título (vários em 2.0.0rc4); e o IB Gateway exige autenticação manual (2FA) uma vez por semana, o que quebra o "liga e deixa rodando". Esses três pontos tornam a rota Nautilus + corretora do Igor mais cara do que o núcleo próprio.
- **Latência não decide:** em passos de minutos, 190 ms de latência do Brasil movem o preço ~0,07 pip (σ do EURUSD ≈ 1,26 pip por minuto), contra 1,1 a 3,8 pips de custo por ida e volta. Um VPS perto da corretora só importa se algum dia entrar estratégia de segundos.

## Consequências

1. **As estratégias precisam rodar o mesmo código no laboratório e ao vivo.** Numba não chama objetos Python arbitrários em velocidade, então cada estratégia é uma função compilada `step(estado, barra) -> intenção de ordem` sobre um estado plano. No runner, o mesmo `step` é chamado em Python barra a barra. Isso evita duas implementações que divergem. É o que o step 6 (interface de estratégia) deve provar: o mesmo sinal alimentado em lote e barra a barra.
2. **Limites do numba:** sem strings nem objetos dentro do laço, estado em arrays, e erros de compilação difíceis de ler. Mitigação: cada estratégia tem teste comparando com uma implementação de referência em Python puro, como neste spike.
3. A capacidade dos vetores de trades foi fixada em 2× o número de barras (uma barra pode conter a saída por sinal e o stop da posição que a substituiu); no núcleo definitivo isso vira contagem prévia ou vetor crescente, para não alocar centenas de MB por configuração.
4. Os 246 milhões de barras/s são o **teto de um laço simples**. Estratégias reais com sessão, calendário e mais estado serão mais lentas; a meta do plano (≥ 1 milhão de barras/s por núcleo) tem folga de dezenas de vezes.
5. **O simulador é uma promessa de fidelidade, não de lucro:** ele reproduz o preenchimento, não o preço executável de cada corretora. O soak em demo (step 25) mede essa diferença.

## Alternativas consideradas

- **NautilusTrader completo (backtest e ao vivo):** descartado pelo adaptador de cTrader e pela velocidade do backtest (~2.000× mais lento). Mantido para a rota IBKR.
- **backtrader, vectorbt, QuantConnect Lean, hftbacktest:** não medidos neste spike; a pesquisa de método em andamento pode reabrir a decisão se algum deles trouxer adaptador de cTrader e velocidade comparável.

## Quando revisar

Se a segunda corretora for a Interactive Brokers, reavaliar o Nautilus como runner dessa rota, olhando o estado dos issues do adaptador e a estabilidade da v2. Se o numba se mostrar um freio de desenvolvimento nas famílias mais complexas (deriva pós-notícia, pares correlacionados), reavaliar a divisão entre laboratório compilado e estratégia em Python.
