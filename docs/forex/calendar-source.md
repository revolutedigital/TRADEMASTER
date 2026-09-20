# Fonte do calendário econômico histórico (step 5)

Data da medição: 2026-09-20. Janela: 2019-01-01 a 2026-08-31. Script: `backend/scripts/research/fx_calendar_probe.py` (47 testes em `backend/tests/unit/test_fx_calendar_probe.py`, sem rede).

## Veredito

**F4 sai do escopo, conforme o critério pré-registrado.** Existe fonte oficial, gratuita e com termos compatíveis para 5 das 6 categorias (NFP, CPI dos EUA, Fed, BCE, BoE), com cobertura de 99,5% e horário conferido contra o mercado. A sexta, a decisão do BoJ, tem a data oficial mas **nenhuma fonte publica o horário**: cobertura com horário 0% nessa categoria e 85,3% no agregado (365 de 428), abaixo dos 90% exigidos pelo pré-registro (seção F4) nas duas leituras possíveis do critério (por categoria e agregado).

Se o Igor quiser F4 mesmo assim, o caminho é uma emenda registrada antes de qualquer resultado (ver "Decisões que dependem do Igor"). Não a apliquei: mexer no critério depois de ver o número é decisão dele, não da pesquisa.

## Fontes escolhidas e como obter

Agregadores (Forex Factory, Investing.com) proíbem copiar o histórico; usei os próprios emissores, todos gratuitos e sem conta. Cada linha do calendário guarda a URL de origem (`source`).

| Categoria | Data | Horário | Página |
|---|---|---|---|
| NFP, CPI (EUA) | tabela anual do BLS | coluna "Release Time" de cada linha (08:30 AM) | `bls.gov/schedule/{ano}/home.htm` |
| Decisão do Fed | páginas de reuniões (2021 em diante no calendário, 2019-2020 nas páginas históricas) | "For release at ..." + EST/EDT em cada comunicado | `federalreserve.gov/monetarypolicy/fomccalendars.htm` e `newsevents/pressreleases/monetaryAAAAMMDDa.htm` |
| Decisão do BCE | lista "Monetary policy decisions" por ano | "press conference starting at 14:30 (ou 14:45) CET today" em cada comunicado, mapeado para o horário da decisão pelo aviso do BCE de 2022-06-27 | `ecb.europa.eu/press/govcdec/mopo/{ano}/html/index_include.en.html` |
| Decisão do BoE | planilha "MPC voting history", aba "Bank Rate Decisions" (datas de anúncio) | 12:00 de Londres, frase da página de política monetária do BoE | `bankofengland.co.uk/-/media/boe/files/monetary-policy-summary-and-minutes/mpcvoting.xlsx` |
| Decisão do BoJ | tabela "Date of MPM" (último dia da reunião) | **não publicado** | `boj.or.jp/en/mopo/mpmsche_minu/index.htm` e `past.htm` |

Reprodução (cerca de 150 requisições, pausa de 1,5 s, robots.txt respeitado, cache em `backend/data/raw/calendar/raw`; a segunda rodada é offline):

```
cd backend && .venv/bin/python -m scripts.research.fx_calendar_probe --contact "<e-mail ou URL seu>"
```

O BLS bloqueia robô sem contato (403 "Access Denied") e diz isso nos termos; por isso `--contact` vira parte do User-Agent. `--refresh` baixa de novo, `--skip-m1` pula a checagem com mercado. Saídas em `backend/data/raw/calendar/` (ignorada pelo git): `events.parquet` e `events.csv` (368 linhas: `timestamp_utc`, `currency`, `event`, `impact`, `source`), `events_without_time.csv` (as 62 datas do BoJ), `coverage_by_year.csv`, `coverage_summary.csv`, `m1_reaction.csv`.

`impact` é "high" por construção: as fontes oficiais não têm rótulo de impacto, e só as 6 categorias foram coletadas.

## Cobertura medida (encontrado com horário / esperado)

Esperado = frequência nominal (NFP e CPI 12 por ano; Fed, BCE, BoE e BoJ 8 por ano), proporcional aos meses de 2026 dentro da janela (Jan-Ago: 8 e 5).

| Categoria | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | Total | Cobertura |
|---|---|---|---|---|---|---|---|---|---|---|
| NFP | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 11/12 | 8/8 | 91/92 | 98,9% |
| CPI EUA | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 11/12 | 8/8 | 91/92 | 98,9% |
| Fed | 8/8 | 9/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 5/5 | 62/61 | 100% |
| BCE | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 5/5 | 61/61 | 100% |
| BoE | 8/8 | 10/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 5/5 | 63/61 | 100% |
| BoJ (data sim, horário não) | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/8 | 0/5 | 0/61 | 0% |
| Sem BoJ | | | | | | | | | 365/367 | 99,5% |
| Com BoJ | | | | | | | | | 365/428 | 85,3% |

"Cobertura" conta no máximo o esperado por ano (evento extra não tapa buraco). Diferenças em relação ao nominal, todas explicáveis:

- **NFP e CPI de 2025 (11 em vez de 12):** o BLS não divulgou o relatório de outubro de 2025 (paralisação do governo). A tabela do BLS mostra o NFP de setembro em 20/11 e o de novembro em 16/12; o CPI de setembro em 24/10 e o de novembro em 18/12. É ausência real, não lacuna da fonte: a cobertura efetiva é 100%.
- **Fed 2020 (9):** 7 reuniões regulares mais 2 de emergência com decisão de juros, 2020-03-03 (10:00 EST) e 2020-03-15 (domingo, 17:00 EDT), no lugar da reunião de 17-18/03 cancelada. Ficaram de fora, por regra, as votações por notação (2020-03-19/23/31 e 08-27), a reunião não programada de 2019-10-04 (sem decisão de juros) e a declaração de metas de longo prazo de 2025-08-22.
- **BoE 2020 (10):** o arquivo do BoE lista duas reuniões fora do calendário em março de 2020 (11 e 19/03; a de 19/03 é "special MPC meeting" no rodapé da planilha). Não separei qual é qual; o horário dessas duas é suposição (12:00), sem confirmação por evento.
- **BCE 2026 (5 até agosto):** 05/02, 19/03, 30/04, 11/06, 23/07. O anúncio noturno do PEPP em 2020-03-18 não é decisão de juros e não entra.
- **BoJ:** 62 datas oficiais (2020 tem 9: reuniões extraordinárias de 16/03 e 22/05), todas sem horário.

## Fuso e horário de verão

Toda hora local vira UTC com o banco IANA (`zoneinfo`), nunca com deslocamento fixo. EUA e Europa trocam em datas diferentes (EUA: 2º domingo de março e 1º de novembro; Europa e Reino Unido: último domingo de março e de outubro), então há semanas com 1 hora de diferença entre as duas regiões. Hora que não existe (salto de primavera) ou que se repete (outono) é erro explícito, não palpite.

| Categoria | Fuso | Horário local publicado |
|---|---|---|
| NFP, CPI | America/New_York | 08:30 (91 de 91 e 91 de 91) |
| Fed | America/New_York | 14:00 (60 de 62); 10:00 em 2020-03-03; 17:00 em 2020-03-15 |
| BCE | Europe/Berlin | 13:45 até 2022-06 (28 decisões); 14:15 de 2022-07-21 em diante (33) |
| BoE | Europe/London | 12:00 (63) |

Pontos que enganam:

- **O BCE mudou o horário em 2022-07-21.** Decisão de 13:45 para 14:15 e coletiva de 14:30 para 14:45 (aviso de 2022-06-27, `ecb.europa.eu/press/pr/date/2022/html/ecb.pr220627~73acedf868.en.html`). Usar 14:15 fixo erraria 28 das 61 decisões em 30 minutos. O script lê o horário da coletiva em cada comunicado e recusa o que contradiz a era (nenhum caso).
- **O BCE escreve "CET" o ano todo** e quer dizer hora local de Frankfurt. Confirmado pelo mercado: em setembro de 2024 o pico está em 12:15 UTC, que é 14:15 CEST.
- O Fed informa EST ou EDT em cada comunicado; o script confere com o calendário de Nova York e falha se divergirem (62 de 62 batem).
- O HistData, base dos M1, usa a regra europeia de verão (`fx_histdata_ticks.py`), o que não interfere aqui: comparo tudo em UTC.

## Verificação dos horários

**O que confere o quê.** O script lê o horário nas páginas oficiais: BLS "08:30 AM", Fed "For release at 2:00 p.m. EDT" (comunicado de 2024-09-18), BCE "press conference starting at 14:45 CET today" (comunicado de 2024-09-12) mais o aviso de 2022 que põe a decisão 30 minutos antes. Isso é a fonte, não a verificação. Verificação independente: (a) o relatório do BLS de julho de 2024 traz o embargo "8:30 a.m. (ET) Friday, August 2, 2024" (`bls.gov/news.release/archives/empsit_08022024.htm`; vi pelo resultado de busca, não baixei a página); (b) o próprio mercado, que não passa por nenhuma das páginas: o minuto de maior amplitude do M1 (EURUSD para USD e EUR, GBPUSD para GBP) num intervalo de 10 minutos em torno do horário esperado, comparado com a mediana da hora anterior.

Eventos conferidos, todos com pico exatamente no minuto esperado em UTC:

| Evento | Horário local (regime) | UTC esperado | Pico no M1 | Amplitude vs. mediana |
|---|---|---|---|---|
| NFP 2024-08-02 | 08:30 (EDT) | 12:30 | 12:30 | 31x |
| Fed 2024-09-18 | 14:00 (EDT) | 18:00 | 18:00 | 47x |
| BCE 2024-09-12 | 14:15 (CEST) | 12:15 | 12:15 | 15x |
| NFP 2024-03-08, antes da troca dos EUA | 08:30 (EST) | 13:30 | 13:30 | 40x |
| NFP 2024-04-05, depois da troca dos EUA | 08:30 (EDT) | 12:30 | 12:30 | 50x |
| BCE 2024-03-07, antes da troca da Europa | 14:15 (CET) | 13:15 | 13:15 | 23x |
| BCE 2024-04-11, depois da troca da Europa | 14:15 (CEST) | 12:15 | 12:15 | 15x |
| BoE 2024-03-21 (GMT) e 2024-05-09 (BST) | 12:00 | 12:00 e 11:00 | 12:00 e 11:00 | 17x e 26x |
| CPI 2024-03-12, EUA em EDT e Europa ainda em CET | 08:30 | 12:30 | 12:30 | 40x |
| Fed 2019-03-20, mesma semana de descompasso | 14:00 (EDT) | 18:00 | 18:00 | 31x |
| NFP 2024-11-01 e Fed 2019-10-30, EUA em EDT e Europa já em CET | 08:30 e 14:00 | 12:30 e 18:00 | 12:30 e 18:00 | 32x e 22x |
| BCE 2022-06-09 e 2022-07-21, os dois lados da mudança de horário do BCE | 13:45 e 14:15 (CEST) | 11:45 e 12:15 | 11:45 e 12:15 | 29x e 19x |

Não houve divergência entre o esperado e o achado nesses eventos.

**Varredura de todos os eventos** (`m1_reaction.csv`, "clara" = amplitude do pico de pelo menos 5x a mediana da hora anterior):

| Categoria | Eventos | Com M1 | Pico claro | No minuto esperado | Até 1 min | Pico 1 h fora mais forte | Amplitude mediana |
|---|---|---|---|---|---|---|---|
| NFP | 91 | 88 | 84 | 76 | 80 | 0 | 20x |
| CPI EUA | 91 | 90 | 73 | 71 | 71 | 2 | 17x |
| Fed | 62 | 61 | 59 | 52 | 58 | 1 | 23x |
| BCE | 61 | 58 | 45 | 35 | 41 | 5 | 10x |
| BoE | 63 | 63 | 54 | 47 | 51 | 1 | 14x |
| Total | 368 | 360 | 315 | 281 | 301 | 9 | |

Dos 315 eventos com pico claro, 89% caem no minuto esperado e 96% a até 1 minuto; os demais reagem 2 a 10 minutos depois (a reação se desenrola) e um adianta 1 minuto (BoE 2020-01-30). Um erro de fuso de 1 hora apareceria como pico forte uma hora fora do esperado, e é isso que a coluna "1 h fora mais forte" mede: 9 casos em 360, nenhum de NFP. Cinco são o BCE antes de 2022-07 e um é o Fed de 2023-03-22; a explicação provável é a coletiva de imprensa (45 minutos depois da decisão no BCE daquela época, 30 no Fed), que às vezes mexe mais que o comunicado, mas não isolei o motivo evento a evento. Dois são CPI de 2020 com reação fraca (2020-04-10 foi Sexta-Feira Santa) e um é a reunião especial do BoE de 2020-03-19, cujo horário já é suposição. Os 8 eventos sem M1 são 6 do buraco de 2023-03 a 07, o domingo de 2020-03-15 e o CPI de 2026-05-12. Os 45 eventos sem pico claro (CPI e BCE sobretudo, com dado em linha) não provam nem refutam o horário.

**BoJ, só para caracterizar (análise pontual fora do script, não é fonte).** Em USDJPY, 43 das 62 decisões têm pico claro (razão de pelo menos 8x na janela 09:30 a 15:00 JST); 37 delas caem entre 11:00 e 12:59 JST, mas os picos vão de 09:55 a 14:49 JST. O horário varia de reunião para reunião, o BoJ não o divulga e só dá para datá-lo depois de ver a reação, o que vaza o futuro numa regra de entrada "10 minutos depois da divulgação".

## Termos de uso

Todos permitem uso de pesquisa pessoal, com citação da fonte. Trechos literais:

- **BLS** (`bls.gov/opub/copyright-information.htm`): "everything that we publish, both in hard copy and electronically, is in the public domain, except for previously copyrighted photographs and illustrations. You are free to use our public domain material without specific permission, although we do ask that you cite the Bureau of Labor Statistics as the source." Robôs (`bls.gov/bls/blsterms.htm`): "BLS also reserves the right to block robots that do not contain information that can be used to contact the owner." Sem contato no User-Agent, o acesso deu 403.
- **Fed** (`federalreserve.gov/disclaimer.htm`): "Unless otherwise indicated, information on Board's website is in the public domain and may be copied and distributed without permission. Please cite to the Board as the source of the information."
- **BCE** (rodapé do comunicado de 2022-06-27): "Reproduction is permitted provided that the source is acknowledged." O `robots.txt` do BCE não bloqueia as páginas usadas.
- **BoE** (`bankofengland.co.uk/legal`): "You may ... download, display or print the Resources for personal use or internal use within an individual organisation for non-commercial purposes." A planilha de votos está em `/-/media/...`, coberta por esse trecho. Uso comercial pede autorização do Bank.
- **BoJ** (`boj.or.jp/en/about/copyright.htm`): "The information included in the site may be copied or reproduced with the exceptions below. When copying or reproducing, the source, the Bank of Japan, should be explicitly credited." Exceção: "The copying or reproduction of the content for commercial purposes."

Se o calendário um dia for redistribuído ou entrar em produto vendido, BoE e BoJ pedem autorização. Para pesquisa própria e para operar o capital próprio não há impedimento nos textos acima.

## Fontes descartadas

- **Forex Factory**: `forexfactory.com/notices`: "The copying, republication, or redistribution of FEI's copyrighted content - including but not limited to, forum posts, calendar schedules and specs ... - is explicitly prohibited without prior written consent." e "FEED includes all data displayed on FEI's Calendar product, including events listed, event names, impact ratings ... historic data ... the copying, republication or redistribution of FEED, in part or in whole, is explicitly prohibited." Não usei e não raspei.
- **Investing.com**: `investing.com/about-us/terms-and-conditions`: "It is prohibited to use, store, reproduce, display, modify, transmit or distribute the data contained in this website without the explicit prior written permission of Fusion Media and/or the data provider." Não usei.
- **FRED/ALFRED (datas de divulgação do BLS)**: a API exige chave gratuita e a URL HTML de datas que tentei devolveu 404. Não criei conta. Serviria só como segunda fonte de NFP e CPI, já cobertos pelo BLS.
- **Datasets abertos no GitHub**: os que apareceram na busca são raspagens do Forex Factory e herdam a proibição acima. Não avaliei a licença de nenhum um a um, porque as fontes oficiais já cobrem 5 das 6 categorias.
- **Trading Economics, FXStreet e similares**: não li os termos; são agregadores pagos ou com histórico restrito e a pesquisa não precisou deles.

## Limitações

1. **BoJ sem horário** (acima). É o motivo do veredito.
2. **Não há rótulo de impacto na fonte.** O pré-registro de F4 fala em "evento de alto impacto (rótulo da fonte)". Com fontes oficiais, "alto impacto" vira lista fixa de 5 eventos programados; ficam de fora PIB, vendas no varejo, ISM e as decisões de BoC, RBA, RBNZ e SNB. Pares como AUDUSD e USDCAD só seriam tocados pelos eventos em USD. É outra hipótese, mais estreita, não a mesma F4.
3. **Sem previsão nem valor divulgado.** O plano (step 5) citava os dois; as páginas oficiais têm só data e hora. A regra de F4 usa o movimento do preço, então não precisa deles.
4. **Horário do BoE:** a única fonte por escrito é a frase atual da página de política monetária; não há confirmação por evento. O mercado confirma 12:00 em 2019-2026 (tabela acima), inclusive nos dois lados das trocas de horário.
5. **Reuniões extraordinárias:** só entram as que têm decisão de juros e comunicado (Fed 2020-03-03 e 03-15). BoE 2020-03-11 e 03-19 têm horário suposto. Não catalogo anúncios extraordinários de outra natureza.
6. **Páginas móveis:** a lista do BCE, o calendário do Fed, as tabelas do BoJ e a página de datas do BoE mudam com o tempo. O cache congela o que foi lido; rodar de novo exige `--refresh`, e uma página que mudou de formato reduz a contagem (não inventa evento).
7. **Base de M1:** a checagem com mercado depende da base HistData já validada contra a Dukascopy; o buraco de 2023-03 a 07 deixa 8 eventos sem checagem.
8. **Requisições:** as 8 páginas anuais do BLS vieram de rodada exploratória com contato no User-Agent e foram colocadas no cache; o resto (cerca de 140 páginas, mais os `robots.txt`) foi baixado pelo script.

## Decisões que dependem do Igor

1. **F4 sai (recomendado) ou entra restrita.** Sai por padrão: o critério pré-registrado falha e a regra de F4 no pré-registro assume um rótulo de impacto que não existe. Entra restrita só com emenda registrada antes de qualquer resultado, que teria de dizer: (a) eventos de F4 = NFP, CPI, Fed, BCE e BoE (sem BoJ), lista fixa no lugar do "rótulo da fonte"; (b) K passa de 10 para 11; (c) cobertura medida = 365 de 367 (99,5%). A fonte técnica desse caminho está pronta e o step 16 poderia partir de `events.parquet`.
2. **Contato no User-Agent do BLS.** Para rodar de novo, passar `--contact` com o e-mail ou a URL que ele aceitar expor ao BLS. Não gravei nenhum contato no código.
