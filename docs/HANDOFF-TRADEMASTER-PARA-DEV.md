# Handoff técnico — TRADEMASTER

**Data:** 21/10/2026

**Branch de trabalho:** `feat/forex-rebuild` — commit `f588cc9f69030118a6de4fc37dc856864bc28fd0`

**Base de comparação:** `origin/main` em `3da806d3890228621e51da0c34ca9220096aea32`

**Estado remoto:** a branch local e `origin/feat/forex-rebuild` estavam no mesmo commit na auditoria.
**Status de produção:** não verificado nesta entrega; este documento não autoriza deploy, ordem real ou ativação de bot.

---

## 1. O que o Igor quer que o produto seja

TRADEMASTER deve permitir que uma pessoa:

1. Escolha um ativo sem enfrentar uma tela cheia de parâmetros técnicos.
2. Estude o mercado daquele ativo, incluindo preço, volatilidade, volume/microestrutura quando existir e contexto de custos.
3. Rode pesquisa quantitativa para descobrir padrões e gerar hipóteses de estratégia específicas para o ativo.
4. Treine e compare modelos de forma causal, com dados separados entre descoberta, calibração, holdout e forward/shadow.
5. Adapte uma estratégia ao ativo escolhido somente quando houver evidência suficiente.
6. Acione uma busca de oportunidades no universo inteiro de ativos elegíveis.
7. Só depois de gates explícitos, permita que bots operem em ambiente de demonstração/Testnet, sempre com stop, alvo, limite de perda e kill switch.

O objetivo não é prometer previsão de mercado ou ligar robôs em todos os ativos. O produto precisa distinguir visualmente e tecnicamente:

| Estado | Significado | Pode enviar ordem? |
|---|---|---:|
| Exploração | hipótese, padrão ou score ainda sem validação | Não |
| Validação | backtest causal, custos e holdout | Não |
| Shadow | sinal calculado e registrado, sem ordem | Não |
| Demo/Testnet | experimento autorizado, com limites independentes | Somente se os gates passarem |
| Live | fora do escopo atual | Não |

---

## 2. Decisões e autorizações já dadas pelo Igor

- Atualizar dependências quando necessário para viabilizar o projeto.
- Usar **Testnet/demo**, nunca operar capital real nesta fase.
- Infraestrutura alvo é Railway; não trocar por Vercel, Render, Fly, Supabase ou outro provedor.
- Resolver autenticação JWT e acesso administrativo.
- Permitir seleção ampla de ativos, estudo por ativo, treinamento/adaptação de estratégia e um botão de busca de oportunidades em todos os ativos elegíveis.
- Manter a experiência simples: selecionar ativo → estudar → validar estratégia → acompanhar oportunidade. Parâmetros avançados ficam recolhidos.
- Pesquisar padrões com abordagem multidisciplinar: estatística, mercado, volume/microestrutura, custos, risco e execução.
- Antes de enviar qualquer operação, exigir gain/alvo, stop, limites de perda e governança de risco.

### Credenciais

Não existem chaves, segredos, tokens, senhas ou cookies neste documento. Qualquer credencial que tenha sido digitada em conversa deve ser tratada como potencialmente exposta: **revogar/rotacionar antes de uso**, criar outra no cofre/Railway Variables e nunca inserir em código, arquivo `.env` rastreado, log, print ou commit.

---

## 3. O que já existe na branch

Esta branch é grande: **541 commits, 1.077 arquivos e aproximadamente 198 mil linhas adicionadas** contra `origin/main`. Ela contém, entre outros itens:

- Backend FastAPI, autenticação, rotas de trading, controles Testnet/paper/live e migrações Alembic.
- Caminho principal de execução Binance com controles de risco, OCO/proteção e requisito de feed websocket para iniciar execução.
- Pesquisa Forex extensa, organizada em protocolos, registros, relatórios de descoberta, calibração, holdout, shadow e sensibilidade em `docs/forex/`.
- Runner C13-A para experimento de gerenciamento parcial/trailing em cTrader demo: `backend/scripts/research/fx_c13_demo.py`.
- Pipeline C16 para captura/replay de book de ofertas do cTrader: `backend/scripts/research/fx_c16_microstructure.py` e `backend/scripts/research/ctrader_cbots/`.
- Interface de pesquisa de microestrutura no frontend.
- Configuração Railway com `preDeployCommand = "alembic upgrade head"`.

### Fontes de verdade relevantes

| Assunto | Arquivo/diretório |
|---|---|
| Configuração e JWT/admin | `backend/app/config.py` |
| APIs de trading | `backend/app/api/v1/trading.py` |
| APIs de pesquisa | `backend/app/api/v1/research.py` |
| Executor seguro FX | `backend/app/fx/runner/executor.py` |
| Reconciliação FX | `backend/app/fx/runner/reconcile.py` |
| Journal de execução | `backend/app/fx/runner/journal.py` |
| Integração cTrader CLI | `backend/app/fx/runner/ctrader_cli.py` |
| Experimento C13-A | `backend/scripts/research/fx_c13_demo.py` |
| Pesquisa C16 | `backend/scripts/research/fx_c16_microstructure.py` |
| cBot C16 | `backend/scripts/research/ctrader_cbots/C16DepthProbe.cs` |
| Protocolo C13 | `docs/forex/c13-preregistration.md` |
| Relatório de risco C13-A | `docs/forex/c13-a-partial-trailing-test.md` |
| Protocolo C16 | `docs/forex/c16-preregistration.md` |
| Resultado C16 | `docs/forex/c16-report.md` |

---

## 4. Revisão adversarial: bloqueadores encontrados

### P0 — não executar C13-A até corrigir

#### P0.1 — split de ordens pode deixar exposição órfã ou duplicada

O C13 abre as pernas `bank` e `runner` diretamente pelo `CliVenue`, mas só grava o estado depois das duas ordens. Se houver queda de processo, timeout ou falha de rede após a primeira perna, pode haver posição aberta na corretora que não consta no `demo_state`. No restart, o C13 só sincroniza posições que já conhece; ele não faz inventário global da corretora. Os IDs usam timestamp e não ficam persistidos antes do envio, abrindo chance de duplicação após reinício.

O executor genérico já tem os elementos corretos: grava intenção antes da ordem e possui reconciliação de posições desconhecidas. O C13 os contorna.

**Correção obrigatória:**

1. Fazer C13 reutilizar `Executor`, `Journal` e `reconcile`, ou extrair deles um `enter_split()` seguro.
2. Persistir uma intenção durável e um `split_group_id` antes de cada perna, com ID determinístico, por exemplo `c13:{run_id}:{signal_id}:{role}`.
3. Após timeout ou resultado ambíguo, marcar o grupo como `UNCERTAIN`; bloquear nova entrada até consultar/adotar/fechar a posição no broker.
4. Fazer reconciliação inventariar **todas** as posições da conta, inclusive as que não estão no estado local.
5. Persistir estado de forma atômica, com lock e `fsync`, ou trocar por banco transacional.
6. Criar testes de crash: antes do intent, depois do intent, depois do fill da primeira perna, antes de salvar estado, falha da segunda perna, reinício e reconciliação.

#### P0.2 — cTrader demo não é o mesmo plano do Binance Testnet

O C13 usa conta demo cTrader e CLI própria. Ele não passa pelos armamentos, limites e verificações do fluxo Binance Testnet. Chamar ambos simplesmente de “Testnet” mascara uma fronteira operacional importante.

**Correção obrigatória:**

1. Tratar como serviço/experimento separado: `ctrader-demo`.
2. Exigir conta permitida por ID, variável de ambiente explícita, preflight somente-demo, lock de processo e kill switch próprio.
3. Proibir qualquer configuração que aponte para conta real.
4. Criar runbook: iniciar, verificar posições, pausar, reconciliar, encerrar e recuperar após queda.

---

## 5. Riscos altos antes de confiar em pesquisa ou automação

### P1.1 — parâmetros de risco do C13 não são os do candidato vencedor

O relatório do C13-A aponta um candidato de aproximadamente 2% por operação e 5% de risco aberto. O runner está configurado para 5% por entrada, 15% aberto e stop diário de 30%. Isso configura outro experimento, sem forward test pré-registrado.

**Correção:** congelar manifesto de release contendo commit/hash, dados, universo, parâmetros, custos, risco, período, critérios de parada e autorização. Sem isso, voltar ao envelope 2%/5% para o próximo teste independente.

### P1.2 — risco e alvo usam preço pré-fill

O C13 calcula stop, target e risco na cotação vista antes da ordem; ele não recalcula usando o preço executado. Slippage pode ampliar a perda máxima e alterar a relação risco/retorno.

**Correção:** depois de cada fill, confirmar stop no broker, recalcular risco monetário e R com preço real e taxa de conversão; em caso de excesso, emendar proteção ou encerrar a posição. Cobrir slippage e fill parcial em teste.

### P1.3 — C16 quebra o universo pré-registrado

O protocolo C16 inicial define 7 pares FX. O Python usa por padrão 37 símbolos. Isso muda o experimento e pode fazer a captura falhar se algum ativo não tiver DOM disponível na conta.

**Correção:** instituir um `C16_MANIFEST` único. `--symbols` deve aceitar somente o universo congelado ou exigir `--amendment-file` versionado, hashado e com novo `run_id`.

### P1.4 — C16 não detecta dados localmente perdidos

O pipeline registra `source_event_sequence`, mas não valida gaps, duplicatas nem inversão de ordem. Tampouco mantém uma fila WAL limitada para absorver bursts sem bloquear a captura. Um relatório pode ser calculado sobre uma amostra incompleta sem que isso invalide a execução.

**Correção:** por símbolo e partição, registrar hash do arquivo, contagem, primeira/última sequência, lacunas, duplicatas, versão/hash do cBot e imagem de execução. O avaliador deve recusar automaticamente dados incompletos.

---

## 6. Melhorias necessárias antes de produção

### P2.1 — C16 não possui regressão automatizada suficiente

Não há testes específicos de replay/captura C16 ou do cBot no repositório. A API foi testada, mas o pipeline real de book não foi validado contra cTrader neste ambiente.

**Fazer:** fixtures de stream com gap/duplicata/desordem, teste de causalidade, teste de joins, teste do manifesto, e smoke real do cBot em cTrader demo.

### P2.2 — loop C13 continua após erro amplo

Falhas recorrentes de sincronização, stop ou corretora podem ficar apenas no log local e o loop continua.

**Fazer:** classificar erro recuperável/fatal; erro de proteção, reconciliação ou estado deve pausar novas entradas, alertar o operador e exigir reconciliação explícita.

### P2.3 — fallback de JWT/admin é perigoso se ambiente estiver mal configurado

Se `APP_ENV` não for produção, há fallback de senha administrativa e JWT efêmero. Não foi feita leitura das variáveis do Railway, portanto não é prova de que o deploy esteja vulnerável, mas a aplicação deve falhar fechada.

**Fazer:** validar enum de ambiente; em Railway exigir `APP_ENV=production`, `JWT_SECRET`, senha/admin e origens permitidas no boot; remover fallback de senha.

### P2.4 — release precisa ser reprodutível e revisável

O backend usa versões por faixa, e o diff contra `main` apresenta 2.422 alertas de whitespace. Uma branch com 541 commits não deve ser aprovada como um pacote monolítico.

**Fazer:** limpar `git diff --check`, fixar resolução de dependências, e separar PRs em: core de execução, C13, C16, docs/dados e frontend.

---

## 7. Segurança e fronteiras que não podem ser quebradas

- Nunca registrar ou exibir segredo em logs, docs, testes, terminal, commit ou interface.
- Rotacionar credenciais que tenham passado por conversa antes de uso efetivo.
- Não habilitar `LIVE`; o escopo atual é paper, Testnet Binance e cTrader demo segregado.
- Nenhuma pesquisa, score ou modelo pode enviar ordem por padrão.
- Alterações de API devem começar por OpenAPI 3.1, validação de request/response e testes de autorização.
- Proteger toda mutação crítica com autenticação administrativa, auditoria e confirmação server-side.
- Railway é o ambiente de deploy. Não usar SSH manual para “consertar produção”; o deploy deve vir de commit e migração no pre-deploy.

---

## 8. Plano recomendado de continuidade

### Fase 0 — travar o risco atual

1. Desabilitar qualquer start automático C13.
2. Declarar C13 como `ctrader-demo` e C16 como `research-only` na UI, API e documentação.
3. Criar checklist operacional e rotação de credenciais.
4. Corrigir boot fail-closed de produção.

**Aceite:** não existe caminho de C13/C16 para ordem sem preflight, ambiente explícito e operador autenticado.

### Fase 1 — corrigir execução C13

1. Implementar `enter_split()` durável e recuperável.
2. Unificar journal, reconciliação e IDs de cliente.
3. Validar risco real após fill e proteção confirmada no broker.
4. Criar kill switch, alertas e estados `PENDING`, `UNCERTAIN`, `PROTECTED`, `CLOSED`.

**Aceite:** testes determinísticos cobrem crash, timeout, fill parcial, slippage, ordem duplicada, posição órfã e restart; não há nova entrada enquanto existir estado incerto.

### Fase 2 — tornar C16 cientificamente auditável

1. Congelar universo inicial em 7 pares e registrar amendment para toda mudança.
2. Implementar integridade de sequência e partição.
3. Criar fila/WAL de captura e teste de burst.
4. Materializar features com ponto temporal causal e documentar granularidade.
5. Adicionar testes do replay e smoke cTrader demo.

**Aceite:** cada relatório C16 possui manifesto verificável, input hashado, cobertura de sequência e rejeita dados inválidos.

### Fase 3 — motor de descoberta e estratégia por ativo

1. Definir universo elegível por corretora, liquidez, custo e disponibilidade de dados.
2. Para cada ativo: criar perfil de mercado, dataset versionado, baseline simples e modelo candidato.
3. Impor pipeline: descoberta → calibração → holdout congelado → shadow prospectivo → decisão humana.
4. Medir retorno líquido de custo, drawdown, estabilidade por regime, turnover, slippage e sensibilidade de parâmetros.
5. Produzir explicação curta para usuário: padrão detectado, evidência, risco, período e por que não está aprovado quando falhar.

**Aceite:** não existe estratégia “aprovada” sem holdout, custo, registros de versão e shadow prospectivo suficiente.

### Fase 4 — produto simples para o usuário

Fluxo recomendado:

1. `Selecionar ativo`.
2. `Estudar mercado`.
3. Resultado em três cartões: contexto, padrões/hipóteses e estratégias validadas.
4. `Treinar/adaptar estratégia` somente abre configurações avançadas quando necessário.
5. `Buscar oportunidades` varre apenas o universo elegível e mostra: ativo, sinal, confiança, custo, risco, estado de validação e ação permitida.
6. Execução fica sempre separada, com confirmação explícita e somente para estratégia liberada em demo/Testnet.

**Não construir:** uma tela que permita escolher qualquer ativo, clicar “treinar” e ligar robô sem evidência e sem limites.

### Fase 5 — release controlado

1. PRs pequenos, OpenAPI atualizado, lint, testes e `git diff --check` limpo.
2. Migrações testadas em banco vazio e upgrade a partir do estado anterior.
3. Deploy Railway pelo commit; validar health, migração, autenticação e rota exata após deploy.
4. Rodar primeiro shadow, depois demo/Testnet com limites baixos e janela forward definida.
5. Liberar expansão de universo somente depois de relatório auditável do lote inicial.

---

## 9. Validações já executadas na auditoria

| Verificação | Resultado |
|---|---:|
| Parse AST Python | 967 arquivos, sem falha |
| Lint dos caminhos críticos | Aprovado |
| Testes existentes cTrader/runner FX | 108 aprovados |
| Testes C13 executados | 19 de 25 aprovados; 6 pendentes de execução completa |
| Gates Railway, risco e start do motor | 4 aprovados |
| Testes de migração | 11 aprovados |
| API de microestrutura | 45 aprovados |
| Frontend de research | 2 aprovados |

Limitações da auditoria:

- Não foi executada a suíte completa.
- Não houve cTrader/Docker real disponível neste ambiente.
- Não foi verificada a configuração efetiva de variáveis/deploy no Railway.
- A varredura de segredos em arquivos rastreados não encontrou chave hardcoded, mas não substitui auditoria de histórico com ferramenta dedicada.

---

## 10. Primeiro pacote de trabalho para o próximo dev

Não começar por IA nova ou por mais ativos. Primeiro entregar, em uma PR isolada:

1. Bloqueio de start C13 sem manifesto e ambiente `ctrader-demo` explícito.
2. `enter_split()` com journal durável, IDs determinísticos e reconciliação global.
3. Recalcular risco/target após fill e garantir stop confirmado.
4. Testes de falha e restart correspondentes.
5. OpenAPI e documentação operacional atualizados.

Depois, em uma segunda PR, atacar C16: universo fixo, integridade de captura, manifestos e testes de replay.

**Gate final:** até os itens P0/P1 estarem concluídos e comprovados, C13 não é candidato a operação e C16 não é candidato a estratégia executável.
