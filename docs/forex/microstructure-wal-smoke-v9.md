# Smoke v9 — WAL prospectivo com book, mark e spot

Data da captura: 2026-09-23 UTC  
Escopo: BTCUSDT, Binance público, research-only  
Resultado: pipeline prospectivo validado tecnicamente; evidência estatística ainda insuficiente

## Por que isto foi feito

As rodadas 12 e 13 rejeitaram preço/volume agregado, trailing e calendário simples. O próximo
avanço válido precisa de informação que não existia nesses testes: book/depth, spread real,
microprice, spot-perp e, quando disponível, liquidações. Esta captura curta valida se o recorder e
o materializador conseguem produzir essas features sem ordens e sem conta.

## Captura

Comando executado:

`PYTHONPATH=. ../backend/.venv/bin/python scripts/research/record_microstructure.py --root data/microstructure_v1/prospective-wal-smoke-v9 --symbol BTCUSDT --duration-seconds 90 --include-spot-trades --audit-status-path data/microstructure_v1/prospective-wal-smoke-v9-audits/evidence-gate-status.json --audit-rolling-days 1 --audit-interval-seconds 30`

Arquivos gerados:

| Stream | Linhas | Primeiro evento | Último evento | SHA-256 |
|---|---:|---|---|---|
| futures trade | 2.849 | 2026-09-23T02:25:09.390Z | 2026-09-23T02:26:38.094Z | `8313283417d55be6e0118cd217501c1dbebb7f3ec9f128ad4077e3df24161895` |
| depth | 831 | 2026-09-23T02:25:13.922574Z | 2026-09-23T02:26:38.265Z | `a349029e4cf93acf16a420a5f37a8a37aebd275b971ca89d296b35c1dd8e3c8b` |
| mark price | 65 | 2026-09-23T02:25:13Z | 2026-09-23T02:26:37Z | `96306a6e0fd4e594838fefdc49a98c2fd04722bfd102e8a8c3e8ba38e5e5dcd0` |
| spot trade | 2.660 | 2026-09-23T02:25:09.460Z | 2026-09-23T02:26:38.373Z | `59f89c09a1ef543179d3bd089693e6e746088ae1bcf08031bffdc53c7b7f6ead` |

O evidence gate segue rejeitado, como esperado: é só ~90 segundos, não um dia completo e muito menos
60 dias. Status auditado: `PARTIAL`, `eligible=false`, `order_submission_allowed=false`,
`execution_authorization=none`.

## Correção técnica necessária

Ao tentar materializar o smoke, o loader de `research_dataset.py` falhou em timestamps WAL ISO-8601
com precisão mista (`2026-09-23T02:25:16Z` sem microssegundos versus timestamps com microssegundos).
Foi corrigido para usar parsing `format="mixed"` em todos os `event_time`.

## Validação de features

Depois da correção, foi gerado um dataset intervalar smoke:

- arquivo: `backend/data/microstructure_v1/smoke-v9-research-rows.parquet`;
- SHA-256: `1f4221c47cdcd29a3b1d17791531b83adb68dc553f81ad7b561e74a35552797d`;
- decisões: 7;
- linhas: 14 (`BUY`/`SELL`);
- colunas: 143;
- book fresco em todas as decisões: `book_available_min=1.0`;
- maior idade do book: 84 ms;
- inclui `mark_index_basis_bps`;
- inclui `spot_perp_basis_bps`.

Todos os targets desse smoke curto ficaram `0`, então não há qualquer leitura de edge aqui. O valor
deste teste é provar que a próxima rodada pode usar as features certas quando houver cobertura.

## Decisão

Ainda não temos estratégia aprovada. Mas o próximo caminho técnico está claro: acumular WAL
prospectivo completo com `trade`, `depth`, `mark_price` e `spot_trade`, depois pré-registrar uma
rodada que use book/microprice/spot-perp em amostra prospectiva. Liquidações ficam opcionais até
existirem eventos suficientes; a falta delas não pode ser preenchida com hipótese.

Nenhuma ordem foi enviada e este resultado não autoriza Demo, Testnet ou LIVE.
