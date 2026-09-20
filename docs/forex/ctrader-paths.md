# cTrader: caminhos para o step 4 (pesquisa sem credenciais)

Data: 2026-09-20. Escopo: leitura de documentação pública, download dos contratos e do `.proto` públicos e `docker manifest inspect` sem login. Nenhuma conta criada, nenhum login, nenhuma credencial usada. Este documento alimenta o step 4 de `docs/plans/forex-fast-bot.md`; o spike com dinheiro de mentira (demo) continua dependendo do que o Igor cria (seção 5).

Convenções: cada afirmação leva a URL da fonte. **[inferência]** é conclusão minha a partir dos fatos. **[não verificado]** é o que não consegui confirmar. Onde não achei algo, digo que não achei.

## Resumo

1. **Correção ao plano:** o "cTrader Console" hoje se chama **cTrader CLI** (anunciado em 13/08/2026). A imagem Docker segue `ghcr.io/spotware/ctrader-console`. O que o plano supôs se confirma: roda headless em Linux/Docker, tem comandos de ordem, posição e preço e **não exige aplicativo da Open API** (só cTID + arquivo de senha + número da conta). O que o plano não previa: os comandos de trading são **one-shot** (um processo e um login por comando), sem stream de cotação, e exigem `--password` na linha de comando (o README da própria Spotware desaconselha isso).
2. **Caminho recomendado: (a) cliente próprio em asyncio sobre o `.proto`** para o runner. O CLI entra como sonda do dia 0 na demo (não espera aprovação) e como ferramenta independente de "fechar tudo".
3. **O prazo de aprovação do aplicativo não é documentado** (só "manual, por e-mail"). Registrar hoje. Corte proposto: sem aprovação em 10 dias úteis, o canário de encanamento roda via CLI em polling.
4. **Fusion:** nada na documentação pública proíbe posições de 1 a 30 minutos, mas os contratos dão poder discricionário de anular trades e encerrar a conta por "abusive trading", sem defini-lo. Comissão da conta Zero confirmada com uma nuance que afeta o simulador (seção 3.4). Entidade que atende brasileiros: **não achei**.
5. **Docker 29.8.1 instalado; `docker manifest inspect` respondeu sem login** (pacote público). Nada foi baixado.

---

## 1. cTrader CLI (imagem `ghcr.io/spotware/ctrader-console`)

### 1.1 O que é de fato

- Nome atual: cTrader CLI, binário `ctrader-cli`; a Spotware o lançou em 13/08/2026 e o descreve como ferramenta para "perform data, trading and algo tasks", inclusive por agentes de IA. Fonte: https://www.spotware.com/news/ctrader-cli/
- A imagem Docker é a "official Docker image for cTrader CLI, a headless Linux runtime that executes cBots, backtests, optimizations and trading commands without requiring cTrader Desktop or a GUI". Fonte: https://github.com/spotware/ctrader-console-docker (README; li o bruto em https://raw.githubusercontent.com/spotware/ctrader-console-docker/main/README.md)
- **Faz as duas coisas.** (i) Roda cBots `.algo` (`run`, `stop`, `cbots`, `metadata`, `backtest`, `optimize`, `create`, `build`) e (ii) expõe comandos de conta, mercado, ordem e posição: `accounts`, `account-stats`, `symbols`, `symbol`, `price`, `prices`, `candles`, `orders`, `order place-market|place-limit|place-stop|place-stop-limit|modify|cancel`, `positions`, `position modify|close|close-partial`, `deals`, `orders-history`, `exposure`, alertas e indicadores. Fonte: https://github.com/spotware/CLI-references (README de referência; li o bruto em https://raw.githubusercontent.com/spotware/CLI-references/main/README.md)
- SL e TP entram nos comandos: `order place-market ... --sl=<sl> --tp=<tp>` e `position modify ... --sl --tp`. O caso de uso oficial "Place a protected order" é "Open a market order with stop loss and take profit, verify the position and adjust the protection". Fontes: CLI-references (acima) e https://help.ctrader.com/ctrader-cli/use-cases/
- cBots: `.algo` compilado para .NET 8 (C# ou Python). A imagem traz .NET SDK e Python 3.12. Fontes: https://help.ctrader.com/ctrader-cli/cbots/ e README da imagem (tabela "What is in the image").
- Python cBots aceitam `requirements.txt` (pandas, numpy), "only when running locally or on a VPS". Fonte: https://help.ctrader.com/ctrader-algo/documentation/third-party-python-packages/ . Se o `numba` do nosso núcleo instala dentro dessa imagem: **[não verificado]**.

### 1.2 Dá para operar de um processo Python externo?

Sim, por `subprocess`, mas com um modelo de uso ruim para um runner 24 horas. **[inferência]** onde marcado.

- Os comandos de ordem, posição, preço e candles são **one-shot**: `--ctid --password --account -q` roda um comando e sai. Fonte: README da imagem, seção Recipes ("Interactive-route commands ... take `--password=…` together with `-q`, which runs one command and exits") e CLI-references. Há também um shell interativo (`ctrader-cli` sem argumentos) em que as credenciais persistem (cabeçalho do CLI-references). Dirigi-lo por pipe/pty seria um remendo. **[inferência]**
- `price`, `prices` e `candles` são retratos, não stream ("Show the current bid and ask for one symbol"). O único modo contínuo é `run` de um cBot: "Launch a cBot and stream its output to stdout. Blocks until the cBot exits." Fonte: CLI-references. **[inferência]** Cotação em tempo real por CLI = polling.
- **A senha vai na linha de comando** nos comandos de trading. O mesmo README diz: "Avoid `--password=` on the command line — it lands in shell history, `docker inspect` and `ps`". A contradição está na própria documentação (seções "Recipes" e "Credentials & security" do README da imagem).
- Formato de saída dos comandos de trading (JSON, tabela): **não documentado** nas páginas que li; só existe `--report-json` para backtest e optimize. **[não verificado]** Parsear texto é frágil. **[inferência]**
- Latência e limite de logins por comando: nada documentado. O README lista "Exit code 85: The authentication token expired. Re-run". **[não verificado]** se logins em rajada são limitados.
- Sem sandbox no Linux: "Linux has no algo sandbox ... a cBot can do anything the container user can". Fonte: README da imagem.

### 1.3 Precisa de aplicativo da Open API?

**Não.** A autenticação documentada é cTID (usuário ou e-mail) + senha em arquivo + número da conta: https://help.ctrader.com/ctrader-cli/setup/ ("The primary authentication approach uses credential files rather than API applications"). Procurei "Open API" e "token" em setup, faq, cbots, use-commands, troubleshooting, ai, skills e use-cases da CLI e não há exigência de aplicativo. O FAQ diz: "You provide your own cTrader ID, a credentials file, and a hosting environment": https://help.ctrader.com/ctrader-cli/faq/

Suporte a 2FA do cTID na CLI: **não documentado** (não achei). **[não verificado]** Testar na demo antes de contar com isso.

### 1.4 Headless em Linux/Docker

Sim. `linux/amd64` e `linux/arm64`, sem X server: README da imagem. O caso de uso oficial "Run a cBot around the clock" é "start it on a VPS": https://help.ctrader.com/ctrader-cli/use-cases/ . Tamanho medido no registro, sem baixar: 22 camadas, **~318 MB comprimidos** (amd64). Tags são imutáveis, exceto `latest`; o README manda fixar por versão ou digest.

### 1.5 Licença e termos

- README da imagem: "The cTrader CLI is proprietary software of Spotware Systems Ltd and is distributed under the cTrader terms of use" (link para https://ctrader.com/terms-of-use/).
- Li esse texto (bruto). Ele trata do site e do marketplace de cBots. A única regra de acesso automatizado é a cláusula 17, sobre aplicações de terceiros que interagem com **o Website** ("scripts designed to scrape or extract data from the Website"). Nada sobre CLI nem trading automatizado.
- O EULA da plataforma (https://www.spotware.com/eula/) dá licença "personal and non-commercial use", lista Windows, Mac, iOS e Android (Linux não aparece) e diz, na 4.3, que quem usa "API, connector, cBot ... or other automated tool" o faz "at your sole risk" e que ações iniciadas por essas ferramentas são "deemed authorised".
- FAQ: a CLI "is included with cTrader"; não achei custo separado: https://help.ctrader.com/ctrader-cli/faq/
- **[inferência]** Bot próprio, com dinheiro próprio, é uso pessoal e cabe. Uma futura plataforma que opere contas de terceiros sairia do "non-commercial". Qual termo governa a imagem Linux é ambíguo: vale uma pergunta por escrito à Spotware, sem ser bloqueio.

### 1.6 Outros produtos da Spotware, descartados

Os **MCP servers** (remoto para cTrader Web, local para cTrader Desktop) são para agentes de IA com supervisão humana ("Users are solely responsible for verifying outputs, supervising strategies"): https://help.ctrader.com/ctrader-ai-agent-connect/ . O remoto usa token da sessão do cTrader Web: "Re-authenticate in cTrader Web and update the token in your AI client configuration" (https://help.ctrader.com/ctrader-ai-agent-connect/faq/). **[inferência]** Não servem para runner sem gente olhando.

---

## 2. Open API (cTrader Open API, protobuf)

### 2.1 Criar o aplicativo e aprovação

- Portal https://openapi.ctrader.com/ → login com o cTID → página Applications → "Add new app" → formulário → Save. Status inicial "submitted". "After the application is reviewed by Spotware, you will be contacted via email. The message will either confirm the application approval or request further details." Fonte: https://help.ctrader.com/open-api/api-application/
- **Prazo: a documentação oficial não informa.** Uma página de terceiro (ClickAlgo) afirma "24-48 hours"; não é da Spotware e não vale como prazo (https://clickalgo.com/open-api-configuration). Um usuário perguntou em 2021 quanto demora a ativação e o fórum não mostra resposta da Spotware (https://community.ctrader.com/forum/connect-api-support/34517/).
- A revisão é criteriosa: "Spotware carefully evaluates new Open API services and there is a higher chance of your application getting approved if you explicitly describe why it is needed and what it will allow users to do". Fonte: https://help.ctrader.com/open-api/creating-new-app/
- App não aprovado não autentica: erro `CH_CLIENT_AUTH_FAILURE = 101; // Open API client is not activated or wrong client credentials`. Fonte: `OpenApiModelMessages.proto`, https://raw.githubusercontent.com/spotware/openapi-proto-messages/main/OpenApiModelMessages.proto
- Redirect URIs só depois da aprovação; o padrão é "only for the playground environment". Fonte: api-application.
- Termos da Open API (https://help.ctrader.com/open-api/terms-of-use/): gratuita ("reserves the right to change the pricing policy without prior notice"), licença "limited, non-exclusive, non-assignable", "as is", uso justo com direito de restringir ou remover o app, e proibição de embutir execução de trades "on behalf of the trader" sem aprovação explícita dele. Bot do próprio dono na própria conta cabe. **[inferência]**

### 2.2 Fluxo de autenticação

Fonte de tudo abaixo: https://help.ctrader.com/open-api/account-authentication/ (OAuth 2.0).

1. Conexão nova → `ProtoOAApplicationAuthReq` com `clientId` e `clientSecret` (o app precisa estar aprovado).
2. Consentimento do usuário no navegador: `https://id.ctrader.com/my/settings/openapi/grantingaccess/?client_id=...&redirect_uri=...&scope=trading`. Escopos: `accounts` (só leitura) e `trading`. O usuário escolhe quais contas do cTID liberar. **Passo humano, uma vez.**
3. O `code` devolvido vale 1 minuto e é trocado por token em `GET https://openapi.ctrader.com/apps/token` (`grant_type=authorization_code`).
4. Resposta: `accessToken` com `expiresIn` 2.628.000 s (~30,4 dias) e `refreshToken`, "no expiry time itself".
5. `ProtoOAGetAccountListByAccessTokenReq` → `ctidTraderAccountId`; depois `ProtoOAAccountAuthReq` por conta.
6. Renovação: `grant_type=refresh_token` no mesmo endpoint ou `ProtoOARefreshTokenReq`; devolve **novo par** e "the old values ... are automatically invalidated". O FAQ confirma: o refresh token vale "forever until you use it to refresh an access token or if you re-authorise your cTrader ID and trading accounts": https://help.ctrader.com/open-api/faq/ . Dá para renovar antes ou depois de expirar.
7. **Playground**: no portal, Applications → Playground, escolhe o escopo e "Get token" e devolve access e refresh token do **próprio cTID**, sem montar redirect URI. Feito para desenvolver e testar. É o caminho de menor atrito para o Igor.

**[inferência]** O refresh rotaciona: gravar o par novo de forma atômica **antes** de usá-lo; se o runner morrer no meio, o refresh antigo já foi invalidado e a saída é reautorizar à mão. Renovar com folga (por exemplo aos 20 dias) e alertar. Dois processos compartilhando o mesmo refresh token se derrubam mutuamente.

**[não verificado]** Se o mesmo access token serve as contas demo e live do cTID (as conexões são separadas por host).

### 2.3 Endpoints, conexões e limites

- Endpoints: `live.ctraderapi.com:5035` e `demo.ctraderapi.com:5035` (protobuf; JSON na 5036), TCP ou WebSocket, TLS obrigatório. "Demo and live environments are fully separated": conta demo só no host demo. Fontes: https://help.ctrader.com/open-api/proxies-endpoints/ e https://help.ctrader.com/open-api/connection/
- **Limites:** máximo de **50 requisições por segundo por conexão** (não históricas) e **5 por segundo por conexão** (dados históricos). Fonte: https://help.ctrader.com/open-api/ . São por conexão, "no matter how many users are authorized through it" (staff, 26/06/2023): https://community.ctrader.com/forum/connect-api-support/41177/ . Estouro devolve `REQUEST_FREQUENCY_EXCEEDED` (108) ou `BLOCKED_PAYLOAD_TYPE` com `retryAfter` em segundos (proto oficial).
- Conexões: "At most, you should create two connections: one for demo accounts and one for live accounts. Each connection can support an unlimited number of accounts of a certain type." Fonte: connection. Não achei limite rígido além dessa recomendação.
- **Heartbeat** (`ProtoHeartbeatEvent`) pelo menos a cada 10 s, senão a Spotware desconecta por inatividade. A API fica indisponível em manutenções de fim de semana. Fonte: https://help.ctrader.com/open-api/faq/
- Enquadramento: 4 bytes de tamanho + `ProtoMessage` serializado; a doc manda inverter os bytes do tamanho em máquinas little-endian (https://help.ctrader.com/open-api/protocol-buffers-json/). **[inferência]** Tamanho em ordem de rede; confirmar no probe.
- **[inferência]** Para 10 pares, poucas ordens por dia e cotações por assinatura, 50 req/s sobra por ordens de grandeza; o limite que aperta é o de 5/s se puxarmos histórico.

### 2.4 Mensagens que precisamos (proto oficial)

Fonte: `OpenApiMessages.proto` e `OpenApiModelMessages.proto` em https://github.com/spotware/openapi-proto-messages (li os brutos). Confirmei a existência de cada mensagem abaixo com `grep`.

| Necessidade | Mensagem | Nota |
|---|---|---|
| Autenticar | `ProtoOAApplicationAuthReq`, `ProtoOAGetAccountListByAccessTokenReq`, `ProtoOAAccountAuthReq`, `ProtoOARefreshTokenReq` | seção 2.2 |
| Símbolos | `ProtoOASymbolsListReq`, `ProtoOASymbolByIdReq` | `digits`, `pipPosition`, `minVolume`, `stepVolume`, `lotSize`, `swapLong/Short`, campos de comissão |
| Cotações | `ProtoOASubscribeSpotsReq` → `ProtoOASpotEvent` | `bid` e `ask` em 1/100000 do preço; o 1º evento traz o último preço mesmo com mercado fechado (proto e FAQ); `timestamp` só com `subscribeToSpotTimestamp` |
| Barras | `ProtoOASubscribeLiveTrendbarReq`, `ProtoOAGetTrendbarsReq` | **trendbars são só bid**: "It is not possible to get trendbars based on ask prices" (staff, 08/07/2023, https://community.ctrader.com/forum/connect-api-support/41268/). Para M1 bid/ask ao vivo, montar as barras dos spots. **[inferência]** |
| Ordem nova | `ProtoOANewOrderReq` | `orderType` MARKET, LIMIT, STOP, STOP_LIMIT, MARKET_RANGE; `volume` em 1/100 de unidade ("1000 in protocol means 10.00 units"), então 0,01 lote de EURUSD = 1.000 unidades = 100.000 no protocolo **[conta minha]**; `clientOrderId` (até 50) e `label` (até 100) |
| SL/TP na ordem a mercado | `relativeStopLoss`, `relativeTakeProfit` | `stopLoss`/`takeProfit` absolutos são "Not supported for MARKET orders"; os relativos são int64 em 1/100000 do preço (BUY: SL = entrada − relativo). Staff: "For Marker Orders you need to set the relativeStopLoss" (31/08/2022, https://community.ctrader.com/forum/connect-api-support/38778/) |
| Alterar SL/TP | `ProtoOAAmendPositionSLTPReq` | `positionId`, `stopLoss`, `takeProfit` absolutos, trailing opcional |
| Fechar | `ProtoOAClosePositionReq` | `positionId` e `volume` |
| Reconciliar | `ProtoOAReconcileReq` → `ProtoOAReconcileRes` | posições e ordens pendentes abertas; `returnProtectionOrders` opcional |
| Eventos de execução | `ProtoOAExecutionEvent` | `executionType`: ORDER_ACCEPTED, FILLED, PARTIAL_FILL, REJECTED, CANCELLED, EXPIRED...; traz `position`, `order`, `deal`, `errorCode` e `isServerEvent` ("e.g. stop-out") |
| Erros | `ProtoOAOrderErrorEvent`, `ProtoOAErrorRes` | códigos úteis: `PROTECTION_IS_TOO_CLOSE_TO_MARKET`, `MARKET_CLOSED`, `NOT_ENOUGH_MONEY`, `TRADING_DISABLED`, `OA_AUTH_TOKEN_EXPIRED` |
| Conta | `ProtoOATraderReq`, `ProtoOATraderUpdatedEvent`, `ProtoOAMarginChangedEvent` | `ProtoOAAccountType`: HEDGED ou NETTED; **qual modo a conta Fusion cTrader usa: [não verificado]**, muda a reconciliação |

**Pegadinha com risco de segurança:** SL/TP absoluto não vale em ordem a mercado. Há relatos em fórum de SL/TP relativo não aplicado com MARKET_RANGE e de `AmendPositionSLTP` pouco confiável depois do fill; achei só o resumo de uma busca e não localizei o tópico primário. **[não verificado]** Por isso o step 4 precisa de uma matriz de teste (seção 4.4) e o runner precisa conferir `position.stopLoss` no `ExecutionEvent` e no `Reconcile` e, se faltar, fechar a posição a mercado.

### 2.5 Como o SL fica no servidor se o cliente cair

- Documentação do cTrader: os campos "Server take profit and Server stop loss ... remain active even if your cTrader app is closed or disconnected", e "Trailing stop loss works even when your cTrader app is closed or disconnected". Fonte: https://help.ctrader.com/ctrader/interface/trade-watch/ (o texto é sobre a proteção avançada do aplicativo).
- No protocolo: `ProtoOAPosition.stopLoss` ("Current stop loss price"), o tipo de ordem `STOP_LOSS_TAKE_PROFIT` e `ProtoOAReconcileReq.returnProtectionOrders` indicam que a proteção é uma ordem no servidor atrelada à posição. **[inferência]** Não achei frase oficial que diga literalmente que o SL colocado pela Open API sobrevive à queda do cliente. **Esse é o teste central do step 4.**
- O SL não é garantia de preço. Cláusula 8.10 do contrato da Fusion: em volatilidade, notícia e gap, "Buy/Sell Stop and Stop Loss orders may not be filled at requested/declared price but instead at the next best available price" e "placing a Stop Loss order will not necessarily limit the Client's losses at the intended amount". Fonte: seção 3.2.
- Dead-man switch no servidor: **não achei**. Quem desliga o bot travado é o runner (step 23).

### 2.6 SDK Python `ctrader-open-api`

Fontes: https://pypi.org/pypi/ctrader-open-api/json e https://github.com/spotware/OpenApiPy (API do GitHub).

- Última versão utilizável: **0.9.2, de 2024-06-26**. A 0.9.3 (2024-08-06) foi **yanked**. Nenhuma release desde então; último commit no repositório: 2024-08-07 (um revert); 13 issues abertas; licença MIT.
- Pins duros: `Twisted==24.3.0`, `pyOpenSSL==24.1.0`, `protobuf==3.20.1`, `requests==2.32.3`, `inputimeout==1.0.4`; `python >=3.8,<4.0`. É baseado em Twisted (reactor), não em asyncio.
- **[inferência]** Não usar: protobuf 3.20.1 (2022) tende a brigar com o protobuf que outras dependências do backend pedem, e o reactor do Twisted não se encaixa no runner em asyncio. Não testei a resolução de dependências no venv do backend. **[não verificado]**

### 2.7 O `.proto` oficial é estável?

Fonte: https://github.com/spotware/openapi-proto-messages (API do GitHub, log de commits e diffs dos 4 últimos).

- Licença MIT, 40 estrelas, último commit 2025-11-13 (correção de typo). Releases numerados até 91 (2024-07-15).
- Em jul e ago de 2025 removeram só as mensagens `ProtoOAv1PnLChange*` (5 payloads) e ajustaram comentários ("MARKER" → "MARKET"). Nada disso nos afeta.
- **[inferência]** Estável para o nosso uso (protobuf evolui por campos novos). Não li o diff de todos os releases antigos.
- Mitigação: copiar os 4 `.proto` para o repositório, fixar o commit e registrar o hash; acompanhar os releases do repositório, como o próprio FAQ manda ("Please follow the Open API Proto message files repository and its releases").

---

## 3. Fusion Markets

### 3.1 A conta cTrader permite Open API e robôs?

- A Spotware diz que a Open API é aberta a "anyone registered with a cTrader-affiliated broker" e suporta "all trading accounts of any cTrader-affiliated brokers": https://help.ctrader.com/open-api/
- A Fusion diz que no cTrader o trader pode "create their own powerful automated strategies or install their own EAs or trading bots" e responde "Yes" a "Does cTrader support algorithmic trading?" (cAlgo/cBots): https://fusionmarkets.com/Platforms/cTrader . A página da conta Zero chama a execução de "Perfect for EAs": https://fusionmarkets.com/Trading/Zero-Trading-Account
- **A Fusion não menciona a Open API nas páginas que li.** Se ela está habilitada nas contas cTrader Zero, incluindo para residente no Brasil: **[não verificado]**, vai para as perguntas (seção 6). A conta cTrader da Fusion é só Zero ("Fusion Markets' clients have access to Fusion Markets' Zero"; a Classic não existe no cTrader): mesma página.

### 3.2 Termos sobre scalping e "algorithmic tools" (posições de 1 a 30 minutos)

- As páginas comerciais que li não trazem política de scalping (nem proibindo nem liberando). **Não achei.**
- Li o *Client Services Agreement* da **Fusion Markets International Ltd** (Seychelles, FSA, SD096), versão de 30/04/2025: https://fusionmarkets.com/static_images/Client_Services_Agreement_FSA_30_Apr_2025_pdf_810d65e55d.pdf , e o *Financial Product Service Terms* da **Gleneagle Securities** (Vanuatu, VFSC), 28/02/2022: https://strapi-content-264190062813.s3.ap-southeast-2.amazonaws.com/Financial_Product_Terms_VFSC_e970634294.pdf . **Nenhum define scalping, arbitragem de latência, tempo mínimo de posição nem frequência máxima de ordens** (busquei essas palavras nos dois). O que existe é discricionário:
  - **CSA 8.9 (erro manifesto):** a empresa "may amend the details of affected transactions ... and/or declare any or all affected transactions as void".
  - **CSA 8.10:** stops podem preencher no "next best available price", e "the same applies when a trading strategy is deemed as abusive, because it is aiming towards potential riskless profit or another strategy deemed by the Company to be abusive".
  - **CSA 13.2:** se suspeitar de "abusive behavior", a empresa pode "void and/or cancel part or all of the Client's abusive trading transactions, close all and any of the Client's trading accounts and terminate this Agreement". A 13.1 define o alvo como abuso de mercado (Seychelles Securities Act).
  - **CSA 12.6:** "expert advisor" ou hospedagem com acesso em tempo real à conta: a empresa não dá garantias e o uso é por conta e risco do cliente.
  - **Vanuatu 5.12(d):** proibido instruir ordens que causem "disorderly market or otherwise prejudicing the integrity or efficiency of the market".
- **[inferência]** Nenhuma cláusula lida proíbe posição de minutos. O risco é o poder discricionário sem definição. Ele pesa mais nas famílias que operam perto de picos de spread e notícia (F3, F4), onde "erro manifesto" e "abuso" são mais fáceis de alegar. Pergunta por escrito antes de dinheiro real.
- Não achei cláusula de proteção de saldo negativo no texto do CSA; os dois documentos abrem com o aviso de que a perda "may vastly exceed the amount of your initial deposit". **[não verificado]** se há proteção na prática; vai para as perguntas.

### 3.3 Entidade que atende brasileiros

**Não achei qual entidade abre conta para residente no Brasil.**

- A Fusion tem três entidades: Fusion Markets International Ltd (Seychelles, FSA, licença SD096), Gleneagle Securities (Vanuatu, VFSC) e FMGP Trading Group (Austrália, ASIC, só clientes australianos). A página não mapeia país a entidade e não menciona Brasil. A lista de países que a Fusion não aceita (Afeganistão, Congo, Irã, Iraque, Japão, Mianmar, Nova Zelândia, Coreia do Norte, Ontário, Palestina, Rússia, Espanha, Somália, Sudão, Síria, Ucrânia, Iêmen, EUA) **não inclui o Brasil**. Fonte: https://fusionmarkets.com/About-us/Regulations
- A página de depósitos oferece **Pix em BRL** (instantâneo, mínimo US$ 1, taxa 0): https://fusionmarkets.com/Trading/Deposit-options
- **[inferência]** Por exclusão (ASIC é só Austrália), o cliente do Brasil cairia na Seychelles ou em Vanuatu; a Fusion não escreve qual. O cadastro real ou a resposta por escrito dirão.
- CVM (alerta de 2024): "inexiste qualquer oferta relacionada ao mercado de CFDs, inclusive Forex, registrada na CVM. Também não há corretora autorizada pela CVM que atue nesse mercado. Sendo assim, qualquer oferta de CFD realizada no Brasil (incluindo Forex) no presente momento, é IRREGULAR", mas "pessoas residentes no Brasil podem investir no exterior, por livre procura". Fonte: https://www.gov.br/investidor/pt-br/educacional/publicacoes-educacionais/alertas/alerta_cvm_forex_2024.pdf . Não é parecer jurídico; o Igor informou que o parecer já existe.

### 3.4 Comissão da conta Zero em cTrader

**Confirmado, com uma nuance que o simulador não modela.**

- Fusion: "At $2.25 commissions per side and 0.0 spreads"; FAQ: "Starting with $2.25 per 1 standard lot or equivalent ($4.50 per round turn)". Fonte: https://fusionmarkets.com/Trading/Zero-Trading-Account
- No cTrader a comissão é por **valor nocional na moeda base do par**: "In cTrader, commissions are charged at a rate of $2.25 per $100,000 of notional volume traded (both in the base currency traded)". Exemplos da própria Fusion: EURUSD 1 lote = EUR 2,25 por lado; AUDUSD 1 lote = AUD 2,25 por lado; depois convertido para a moeda da conta. Fonte: https://fusionmarkets.com/Platforms/cTrader
- Lote mínimo 0,01 em pares de moeda: página da conta Zero. Com 0,01 lote de EURUSD a comissão é ~EUR 0,0225 por lado. **[conta minha]**
- **Achado para o laboratório:** `backend/app/fx/sim/costs.py` modela `commission_usd_per_lot_per_side=2.25` fixo em USD (`FUSION_ZERO`, `STRESS`). Pela regra da Fusion o valor em dólar varia com a moeda base: com preços de exemplo (EUR ~1,17, GBP ~1,35, AUD ~0,66), EURUSD sai ~US$ 2,6 (+17%), GBPUSD ~US$ 3,0 (+35%), AUDUSD ~US$ 1,5 (-34%), USDJPY US$ 2,25 exato. É pequeno frente ao spread, mas o custo foi congelado no pré-registro; se for corrigido, precisa de emenda registrada **antes** do step 19. **[inferência]** de que a diferença não muda o veredito; vale medir.

---

## 4. Recomendação

### 4.1 Comparação dos caminhos

Horas são **estimativas minhas**, sem medição, incluindo teto +50% quando indicado. O plano já prevê 6-10h para o step 4 e 17-26h para os steps 21-24 (runner).

| | (a) Open API própria em asyncio | (b) CLI em polling (subprocess) | (c) Python cBot dentro da CLI |
|---|---|---|---|
| Depende de aprovação da Spotware | **Sim** (prazo desconhecido) | Não | Não |
| Cotação e eventos | stream (`SpotEvent`, `ExecutionEvent`) | polling de `price`/`candles`, sem eventos | stream dentro do cBot |
| SL/TP no servidor | por mensagem; ordem a mercado exige `relative*` (testar) | `--sl --tp` em `place-market` (documentado) | `ExecuteMarketOrder(..., sl_pips, tp_pips)` (documentado) |
| Segredo | token OAuth de escopo `trading`, só nas contas escolhidas, gerenciável no cTID | **senha do cTID** em argv a cada comando | senha do cTID em arquivo |
| Encaixe no plano (venue falso, journal, risco, kill-switch, vários bots) | total | parcial, latência por chamada desconhecida | ruim: código roda no runtime da Spotware |
| Horas: sonda descartável | 8-14h (teto ~21h) | 3-6h | 12-24h, incerteza alta (numba **[não verificado]**) |
| Horas: runner de canário | o do plano (17-26h) | 8-14h para wrapper, reconciliação e kill-switch | não estimo |
| Pior caso | app negado ou parado; rotação de token derruba o runner; SL a mercado não aplicado | login por comando limitado ou lento; saída sem formato estável; produto lançado há 5 semanas muda | acopla a estratégia ao runtime; dev loop com build .NET; sem teste unitário fácil |

### 4.2 Decisão

1. **(a) é o caminho de produção.** É o único que dá stream, eventos de execução e reconciliação de verdade, mantém o segredo revogável e encaixa na interface neutra de venue do plano.
2. **(b) vira duas coisas úteis já**: sonda do dia 0 na demo, sem esperar aprovação (valida SL/TP em ordem a mercado, SL sobrevivendo, custo e latência reais), e ferramenta **independente** de emergência (`position close --all --yes`, `order cancel --all --yes`), por autenticar com a senha do cTID e não com o token Open API. Usar o mesmo refresh token em dois processos os derrubaria mutuamente, então o "fechar tudo" não deve depender do token do runner. **[inferência]** Custo dessa escolha: a senha do cTID em mais um lugar; usar cTID dedicado ao bot reduz o dano.
3. **(c) fora.** **MCP servers** fora (seção 1.6). **OpenApiPy** fora (seção 2.6).
4. **Regra de corte (julgamento meu):** se o aplicativo não estiver aprovado 10 dias úteis depois de submetido, o canário de encanamento (entra, sai depois de N minutos, com SL) roda via (b) em polling, atrás da mesma interface de venue, e a troca para (a) depois é encaixe. O trade-off: (b) aceita a senha em argv dentro de um container isolado e cTID dedicado.
5. **Nada de dinheiro real antes de:** o teste de SL sobrevivendo à queda em cada caminho usado, e as respostas da Fusion às perguntas 1 a 3 da seção 6.

Canário seguro em ambos os caminhos: lote 0,01 (mínimo da Fusion, `minVolume`/`stepVolume` lidos do símbolo), SL obrigatório em toda ordem, perda diária de 1% (~US$ 5) aplicada **no runner** (não achei teto diário no servidor; **[não verificado]**) e conferência de `position.stopLoss` depois de cada fill.

### 4.3 Efeito nos steps do plano

- **Step 4:** trocar "cTrader Console" por "cTrader CLI"; acrescentar que os comandos de trading são one-shot com `--password` (o "Teste" do step 4 vale igual: matar o processo e conferir SL vivo). O trecho "sem aplicativo da Open API" está correto.
- **Step 1:** submeter o aplicativo hoje é o item de maior prazo incerto.
- **Step 19:** decidir se corrige a comissão por moeda base (seção 3.4) via emenda registrada.

### 4.4 Matriz de teste da Open API na demo (step 4)

1. Ordem a mercado com `relativeStopLoss`/`relativeTakeProfit`; conferir `position.stopLoss/takeProfit` no `ExecutionEvent` e no `Reconcile`.
2. Ordem a mercado sem SL + `AmendPositionSLTP` logo no fill; medir a janela sem proteção.
3. `MARKET_RANGE` com SL absoluto (o proto só proíbe absoluto em MARKET; comportamento em MARKET_RANGE **[não verificado]**).
4. `kill -9` do cliente com posição aberta; conferir no cTrader Web que o SL continua e que dispara (SL curto de propósito).
5. Reconectar e reconciliar sem duplicar ordem (`clientOrderId`).
6. Refresh de token com persistência atômica, inclusive matando o processo entre a resposta e a gravação.
7. Heartbeat, queda forçada de rede e reconexão com reassinatura de spots; comportamento na manutenção do fim de semana.
8. Erros: `PROTECTION_IS_TOO_CLOSE_TO_MARKET`, `MARKET_CLOSED`, volume fora do `stepVolume`.
9. Custo e slippage medidos contra o simulado (alimenta o step 25).

---

## 5. O que o Igor precisa fazer e entregar (sem senhas)

Regra: senha do cTID, `client secret`, access token e refresh token **nunca** vão para chat, repositório ou log; ficam num arquivo `.env` fora do git ou num cofre de segredos que ele cria. Para o assistente vão só identificadores não secretos e o resultado dos passos.

| # | O que | Onde | Entrega ao assistente |
|---|---|---|---|
| 1 | Conta demo Fusion, tipo Zero, cTrader, USD, US$ 500 (já em andamento) | Fusion Client Hub (https://hub.fusionmarkets.com/) | número de login da conta demo (não a senha) |
| 2 | cTID. **Recomendo um cTID dedicado ao bot**, só com as contas do bot | https://id.ctrader.com | confirmação de que existe; e se o cTID tem 2FA (afeta o CLI, seção 1.3) |
| 3 | **Registrar o aplicativo na Open API hoje**: nome, descrição detalhada e honesta (uso pessoal, um único usuário que é o dono, trading próprio, demo antes de live, sem redistribuição), redirect URI qualquer que ele controle (o padrão só serve ao Playground) | https://openapi.ctrader.com/ → Applications → Add new app | `client_id` (não é segredo) e a data de submissão; o `client secret` fica com ele |
| 4 | Se a Spotware pedir detalhes, responder no e-mail do cadastro; sem resposta em 5 dias úteis, cobrar pelos canais de suporte da Open API (Discord ou e-mail listados em https://help.ctrader.com/open-api/faq/) | e-mail do cTID | aviso quando aprovar |
| 5 | Depois de aprovado: gerar o token de acesso da demo no **Playground** (escopo `trading`, só a conta demo) | portal → Applications → Playground | nada; ele grava os tokens no `.env` |
| 6 | Para a sonda do dia 0 via CLI: criar o arquivo de senha do cTID **ele mesmo**, `chmod 600`, fora do repositório e no `.gitignore` | máquina local | só o caminho do arquivo |
| 7 | Mandar as perguntas da seção 6 à Fusion e guardar as respostas por escrito (PDF ou e-mail, **fora do repositório**) | suporte da Fusion (e-mail do Client Hub) | resumo das respostas |
| 8 | Guardar a versão do contrato que valer para a conta dele (Client Services Agreement da entidade que aparecer no cadastro), com data | Client Hub | nome da entidade e versão do contrato |
| 9 | (Opcional) Perguntar à Spotware qual termo rege a imagem Docker no Linux (EULA de "uso pessoal, não comercial" ou os termos do ctrader.com) | suporte da Spotware | resposta |

---

## 6. Perguntas por escrito à Fusion (enviar em inglês)

> Hello. I am an individual resident in Brazil and I am approaching Fusion Markets on my own initiative. I plan to open a cTrader Zero account (USD) and trade small sizes with an automated program. Could you confirm in writing:
>
> 1. **Entity and agreement.** Which legal entity and regulator would open and hold a live account for a Brazilian resident, and which version of the Client Agreement applies? Are there any restrictions on Brazilian residents (products, leverage, funding, withdrawals)?
> 2. **Open API.** Is the cTrader Open API (protobuf, live.ctraderapi.com and demo.ctraderapi.com) enabled on the cTrader Zero live and demo accounts? Are orders sent through the Open API, through cBots, or through the cTrader CLI (Docker on a cloud server) treated any differently from manual orders?
> 3. **Strategy restrictions.** Automated trading with holding times from about 1 to 30 minutes and up to a few dozen orders per day: is it permitted without restriction? Is there any minimum holding time, maximum order rate or maximum number of open orders? Your Client Services Agreement (30 April 2025, clauses 8.9, 8.10, 12.6 and 13.2) allows voiding trades and closing accounts for "abusive" strategies without defining them. What specific behaviours do you treat as abusive? Would you notify me before voiding or adjusting trades or closing the account?
> 4. **Execution.** Do you apply last look, price-staleness or latency-based rejections on cTrader Zero? What is the typical slippage on market orders? Does the demo account replicate the live pricing and execution?
> 5. **Costs.** Please confirm the commission is USD 2.25 per USD 100,000 notional per side in the base currency of the pair (so EURUSD 1 lot = EUR 2.25), whether any minimum commission per order applies to 0.01 lot orders, and where I can find the swap table and the triple-swap day for EURUSD, GBPUSD, USDJPY, AUDUSD, EURJPY, GBPJPY and EURGBP.
> 6. **Risk terms.** What leverage would apply to me by default, and how do I request a change? What is the margin call and stop-out level? Do you provide negative balance protection? Is the account hedging or netting?
> 7. **Stop loss.** Confirm that stop loss and take profit orders attached to positions are held on your trading server and stay active if my client disconnects. When are they not honoured or removed (gaps, news, "abusive" strategies)?
> 8. **Funding and records.** Which funding methods are available for my country and which entity name appears on the receiving side? How do I withdraw, and at what cost? How can I export the full deal history (with commissions and swaps) for tax reporting?
> 9. **Maintenance.** What are the weekend maintenance windows of the trading server and of the Open API, and how are open positions and stop losses handled during them?
>
> Thank you.

Registrar data, canal, quem respondeu e a resposta. Sem resposta em 7 dias conta como "não confirmado" (mesmo critério de `docs/forex/step5-contas-checklist.md`).

---

## 7. Docker

- `docker --version` → `Docker version 29.8.1, build 4a63305d74` (instalado).
- `docker manifest inspect ghcr.io/spotware/ctrader-console` **respondeu sem login** (nenhum login tentado). Devolveu um índice OCI com `linux/amd64` (`sha256:f5dc111c7c47...`), `linux/arm64` (`sha256:11668fdd70bb...`) e dois manifests de atestado de proveniência (`unknown/unknown`, 837 bytes). Também obtive um token anônimo direto do registro (`ghcr.io/token`, sem credenciais) e li as tags e o manifest do amd64, o que confirma que o pacote é público. Não verifiquei se o Docker local tem credenciais salvas para o ghcr.io.
- amd64: 22 camadas, ~318 MB comprimidos. **Nada foi baixado nem executado.**
- Config da imagem `latest`: criada em 2026-09-18, base Ubuntu 24.04, `Entrypoint` `/usr/local/bin/ctrader-cli-entrypoint`. Há 32 tags, entre elas `5.9.10`, `5.9.11`, `5.10.0-alpha` e `5.10.1`. O README ainda diz que `latest` é `5.9.11` e que Ubuntu 24.04 vem só a partir da 5.10, então **`latest` provavelmente já é a linha 5.10 [inferência]**; fixar por versão ou digest no spike.

---

## 8. O que ficou sem verificar

- **Prazo de aprovação do aplicativo** na Spotware (nenhuma fonte oficial; o "24-48h" é de terceiro).
- Open API habilitada nas contas cTrader Zero da Fusion, e para residente no Brasil.
- **Qual entidade da Fusion atende brasileiros**, alavancagem padrão para o Brasil, proteção de saldo negativo e modo hedging/netting da conta.
- Política real da Fusion sobre scalping, latência e tempo mínimo de posição (os contratos lidos não definem). Li o CSA de 30/04/2025; o site pode ter versão mais nova.
- CLI: suporte a cTID com 2FA, formato de saída dos comandos de trading, latência e limites de login por comando, comportamento sob login em rajada.
- Se o mesmo access token serve demo e live.
- Se o SL/TP colocado por Open API (ordem a mercado com `relative*`, `AmendPositionSLTP`, `MARKET_RANGE`) é aplicado e sobrevive à queda do cliente. Só existem a inferência do protocolo e relatos de fórum que não consegui abrir na fonte primária.
- Se `numba` roda dentro do Python cBot da imagem; se `protobuf==3.20.1` do OpenApiPy conflita com o venv do backend.
- Limite rígido de conexões da Open API além da recomendação de duas.
- Qual termo (EULA ou ctrader.com/terms-of-use) governa a imagem Docker no Linux.
- A CVM: não achei ato declaratório específico da Fusion na busca; isso não é prova de ausência.

---

## 9. Resultado do teste na demo Fusion pelo CLI (2026-09-20, feito no mesmo dia da pesquisa)

Conta demo Fusion Markets **10139135** (cTrader, hedge, 1.000 USD, 1:500), cTID com login por senha (o cadastro por Google foi convertido pelo Igor), imagem `ghcr.io/spotware/ctrader-console@sha256:285484fa…4ba` (5.10.1), um container por comando, senha lida de arquivo dentro do container. Script: `backend/scripts/ctrader_demo.sh`.

| Item | Resultado |
|---|---|
| Login pelo CLI | Passou, **sem bloqueio de 2FA**. O cTID listou a demo da Spotware (5918970, EUR) e a da Fusion (10139135). |
| Regras do EURUSD | Lote mínimo 1.000 unidades (0,01 lote), passo 1.000, lote 100.000, 5 casas, pip 0,0001. Spread de 0,6 pip logo depois da abertura de domingo. |
| Ordem a mercado com SL e TP | 1.000 unidades, compra. O CLI avisa que **SL e TP de ordem a mercado saem como pips relativos ao preço executado** (limitação da Open API): confirma o desenho do adaptador (proteção relativa no ato, emenda para o nível exato depois). |
| **Stop no servidor sem cliente conectado** | **Sim.** Num login novo, com o processo que abriu a ordem já encerrado, a posição mostrava `stopLoss 1.14701` e `takeProfit 1.15151` gravados. É a evidência de que o stop é do servidor, não do cliente. |
| Custo real de 1 ciclo de 0,01 lote | Entrada 1.14856 (ask), saída 1.1485 (bid): −0,6 pip de spread, comissão −0,03 USD por lado (−0,06 no ciclo), líquido −0,12 USD. A comissão bate com "2,25 na moeda base": EUR 0,0225 × 1,1485 ≈ US$ 0,026 por lado, e US$ 2,25 fixo daria 0,0225 (arredondaria para 0,02). É consistente, mas o arredondamento a 2 casas não distingue com folga. |
| Fechar tudo | `position close all yes` funcionou e a conta ficou sem posição. |

**Não coberto:** o cliente morrer com a posição aberta e o preço tocar o stop (só provei que o stop fica no servidor); `MARKET_RANGE`; refresh de token; o meu adaptador contra o servidor real (depende da aprovação do app da Open API); o sinal da comissão em `closePositionDetail`; slippage medido em volume (uma ordem só).
