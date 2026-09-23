# Fast105 — pré-registro prospectivo do veto fixo `spot_return_1s_bps >= 0`

Registrado em `2026-09-23T23:30:31Z`, depois do diagnóstico Fast104.

Fast105 não é estratégia aprovada. É uma observação prospectiva fail-closed para
testar se a melhor pista do Fast104 sobrevive fora das 13 janelas usadas na
descoberta.

Fast105 não autoriza Demo, Testnet, LIVE, envio de ordem, commit de ledger ou
operação manual.

## Origem da hipótese

Fast101 mostrou que o candidato de operação maior tinha sinal, mas ainda era
frágil:

- modelo: `hist_gradient_payoff`;
- target: `10 bps`;
- trailing distance: `15 bps`;
- top fraction: `0.05`;
- horizonte: `300s`;
- one-position net @ `1 bps`: `+13.763261447660312` bps;
- one-position net @ `2 bps`: `-28.236738552339688` bps;
- positivo em `11/13` janelas, mas fast24 e fast36 quebravam a tese.

Fast104 testou vetos fixos de outlier com thresholds calculados só no treino.
O melhor diagnóstico foi:

- política: `spot_return_1s_not_crash30`;
- regra materializada recorrente: `spot_return_1s_bps >= 0`;
- net @ `1 bps`: `+105.4141767734227` bps;
- net @ `2 bps`: `+68.4141767734227` bps;
- positivo em `11/13` janelas;
- pior janela @ `1 bps`: `-42.99999999999976` bps.

Conclusão de partida: o filtro parece remover parte importante do regime ruim,
mas ainda não é robusto o suficiente para ser chamado de estratégia. Ele precisa
de validação prospectiva em WALs posteriores a este pré-registro.

## Safety boundary

Durante todo o Fast105:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

Se qualquer implementação futura tentar enviar ordem ou gravar ledger real com
base neste pré-registro, isso é defeito de segurança.

## Hipótese congelada

Treino congelado:

- fast16: `backend/data/microstructure_v1/prospective-wal-fast16-18m`;
- fast20: `backend/data/microstructure_v1/prospective-wal-fast20-30m`;
- fast21: `backend/data/microstructure_v1/prospective-wal-fast21-30m`;
- fast22: `backend/data/microstructure_v1/prospective-wal-fast22-30m`;
- fast24: `backend/data/microstructure_v1/prospective-wal-fast24-30m`;
- fast28: `backend/data/microstructure_v1/prospective-wal-fast28-30m`;
- fast29: `backend/data/microstructure_v1/prospective-wal-fast29-30m`;
- fast32: `backend/data/microstructure_v1/prospective-wal-fast32-30m`;
- fast35: `backend/data/microstructure_v1/prospective-wal-fast35-30m`;
- fast36: `backend/data/microstructure_v1/prospective-wal-fast36-30m`;
- fast82: `backend/data/microstructure_v1/prospective-wal-fast82-30m`;
- fast83: `backend/data/microstructure_v1/prospective-wal-fast83-30m`;
- fast89: `backend/data/microstructure_v1/prospective-wal-fast89-30m`.

Configuração congelada:

- modelo: `hist_gradient_payoff`;
- target: `10 bps`;
- trailing distance: `15 bps`;
- horizonte: `300s`;
- initial stop: `20 bps`;
- top fraction: `0.05`;
- decision stride: `5s`;
- max book staleness: `1000ms`;
- latency: `100ms`;
- custo esperado: `1 bps`;
- custo de stress: `2 bps`;
- mercado inicial: `BTCUSDT` perpetual;
- avaliação: `one-position`, escolhendo por `model_score`.

Regra congelada de entrada:

1. treinar o `hist_gradient_payoff` somente nas 13 janelas históricas acima;
2. pontuar cada nova janela prospectiva sem re-treinar;
3. selecionar o top `5%` por `model_score`;
4. aplicar veto fixo: `spot_return_1s_bps >= 0.0`;
5. aplicar uma posição por vez até o `exit_time_ms`.

Nenhum outro threshold, feature, lado, sessão, densidade, horizonte, top-p,
trailing ou custo entra no Fast105.

## Janela prospectiva

São elegíveis apenas decisões com:

```text
decision_time > 2026-09-23T23:30:31Z
```

Qualquer decisão anterior a esse timestamp fica fora, mesmo que seja
materializada depois.

WALs usados no treino não podem ser usados como confirmação prospectiva. Se
forem passados ao runner, devem resultar em zero decisão elegível ou em status
fail-closed.

## Gates

### Gate de falha precoce

A hipótese é encerrada como falha se qualquer condição ocorrer:

- após pelo menos `10` trades one-position prospectivos,
  `stress_total <= -20 bps`;
- qualquer janela prospectiva com pelo menos `3` trades fechar com
  `stress_total <= -15 bps`;
- `spot_return_1s_bps` precisar ser aproximado, imputado ou substituído;
- qualquer implementação usar dado anterior a `2026-09-23T23:30:31Z` como
  confirmação prospectiva.

### Gate mínimo para candidato de shadow offline

Mesmo que pareça bom antes, não há promoção antes de:

- pelo menos `30` trades one-position prospectivos;
- pelo menos `3` janelas prospectivas distintas;
- stress médio > `+1.5 bps/trade`;
- expected médio > `+2.5 bps/trade`;
- pelo menos `75%` das janelas positivas em stress;
- nenhuma janela suficiente com stress total <= `-15 bps`;
- nenhuma regra alterada desde este documento.

Passar esse gate não aprova Testnet. Passar esse gate só autoriza escrever um
novo relatório propondo shadow offline, ainda `research_only`.

### Gate para Testnet

Fast105 não tem gate de Testnet. Testnet exigiria novo relatório, novo
pré-registro de execução, auditoria de book/freshness/fill/slippage e
autorização explícita do Igor.

## O que NÃO será feito

- Trocar o veto `spot_return_1s_bps >= 0.0` depois de ver dado novo.
- Trocar o top fraction `0.05`.
- Treinar incluindo a janela prospectiva.
- Usar fast16–fast89 como confirmação prospectiva.
- Promover pelo agregado positivo sem número mínimo de trades e janelas.
- Chamar Fast105 de estratégia antes dos gates acima.

## Próxima implementação

O próximo passo técnico é criar um runner Fast105 que:

- valide o hash deste pré-registro e o evento do registro;
- treine somente nas 13 janelas congeladas;
- aceite WALs prospectivos passados explicitamente por `--validation-wal-roots`;
- descarte decisões anteriores ao timestamp deste documento;
- aplique a regra congelada;
- liquide offline com o replay executável já usado no Fast104;
- aplique one-position;
- calcule gates;
- mantenha `orders_sent=false` sempre.
