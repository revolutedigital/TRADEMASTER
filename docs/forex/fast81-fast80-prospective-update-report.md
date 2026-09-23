# Fast81 — update prospectivo do Fast80

Resposta objetiva: não temos uma estratégia aprovada.

O Fast81 foi criado para tirar a dúvida do campo subjetivo: ele procura WALs
locais depois do pré-registro Fast80 e, quando existirem, roda a cadeia
Fast60 → Fast61 → Fast80 sem enviar ordem e sem escrever ledger.

## Safety boundary

O runner mantém:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`;
- `approved_strategy=false`.

## Resultado local atual

Comando executado:

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast81_fast80_prospective_update.py \
  --output-root backend/data/microstructure_v1/reports/fast81-fast80-prospective-update-current
```

Resumo:

- status: `FAST81_NO_POST_PREREG_WAL_ROOTS`;
- WAL roots encontrados: `10`;
- WAL roots com trades: `10`;
- WAL roots pós-pré-registro Fast80: `0`;
- último trade local em WAL: `2026-09-23T14:22:00.799000+00:00`;
- timestamp do pré-registro Fast80: `2026-09-23T18:40:07+00:00`;
- Fast60/Fast61/Fast80 não foram rodados, porque não há dado prospectivo
  local posterior ao pré-registro.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

O que existe agora é uma hipótese rara de vol+flow pronta para observação
prospectiva. Ela só pode virar candidata se novos WALs posteriores ao
pré-registro produzirem settlements scoreáveis e o Fast80 passar os gates
mínimos congelados.

## Arquivos e hashes

- Fast81 script:
  `e3a9a99c816a082df1b29e97742f526f035048d12738490df339a30da3d7dcfa`;
- Fast81 tests:
  `521795ff622318cecdec34e044336c8ffed296279ddff1cc2a8191a4c7c620bf`;
- Fast81 report:
  `be453f68013acb761bfa5f09a9ba6e2cce22f756bb57f5ef4ac331646e528501`.

## Validação

```bash
PYTHONPATH=backend ./backend/.venv/bin/pytest \
  backend/tests/unit/test_fast81_fast80_prospective_update.py -q --no-cov
```

Resultado: `2 passed`.

```bash
./backend/.venv/bin/ruff check \
  backend/scripts/research/run_fast81_fast80_prospective_update.py \
  backend/tests/unit/test_fast81_fast80_prospective_update.py
```

Resultado: `All checks passed!`.

Observação: o mesmo teste sem `--no-cov` também passou funcionalmente
(`2 passed`), mas falhou a suíte por cobertura global do projeto
(`3.49% < 80%`) por ter sido uma execução focal de apenas um arquivo.
