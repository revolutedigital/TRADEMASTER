# Fast178 — Fast177 edge-2 prospective observation run

Fast178 congelou uma variação do Fast177 com cutoff de edge em `2.0` bps,
mantendo o mesmo modelo e o mesmo regime `ETHUSDT_qv_30m >= 100000000.0`.

Âncora prospectiva:

`2026-09-24T10:42:20.437000-03:00`

Status:

`FAST178_WAITING_FOR_POST_PREREGISTRATION_DATA`

## Resposta curta

O experimento está pré-registrado, tem registry próprio e o runner validou o
hash do pré-registro. Com a partição local atual até
`2026-09-24T13:35:59.999000+00:00`, ainda não existe decisão pós-âncora com
outcome completo de `900s`.

Isso é esperado: a primeira decisão estritamente posterior à âncora é
`2026-09-24T13:45:00Z`, e ela só pode ser avaliada depois da janela futura
fechar em torno de `2026-09-24T14:00:00Z`.

## Política congelada

Entrada:

```text
abs(prediction) >= 2.0
ETHUSDT_qv_30m >= 100000000.0
```

Direção:

- `LONG` se `prediction > 0`;
- `SHORT` se `prediction < 0`.

Custo:

- `2.0` bps por decisão offline.

## Resultado do run

| Métrica | Valor |
|---|---:|
| Training rows | `1,710` |
| Features | `47` |
| Última data comum local | `2026-09-24` |
| Prospective rows pós-âncora | `0` |
| Observações elegíveis | `0` |
| Ordens enviadas | `0` |

## Interpretação

Fast178 não prova nada ainda. Ele apenas congela o próximo teste honesto: se o
problema do Fast177 é o cutoff de edge alto demais, o edge mínimo igual ao custo
explícito (`2.0` bps) deve gerar mais entradas futuras. O resultado só conta em
dados posteriores à âncora.

Não há estratégia operável ainda.

## Segurança

Fast178 é observação prospectiva offline:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `dry_run=true`.

Nenhum trade foi enviado. Nenhum ledger foi alterado. O resultado não autoriza
Demo, Testnet nem LIVE.

## Artefatos

- pré-registro:
  `docs/forex/fast178-fast177-edge2-prospective-preregistration.md`;
- registry:
  `docs/forex/fast178-registry.jsonl`;
- runner:
  `backend/scripts/research/run_fast178_fast177_edge2_observation.py`;
- teste:
  `backend/tests/unit/test_fast178_fast177_edge2_observation.py`;
- report JSON:
  `backend/data/microstructure_cross_asset_v1/reports/fast178-fast177-edge2-observation/report.json`.

Hashes:

- pré-registro:
  `ba2ababfcb97d1a1fad8fdf07302cbb95be7abfb9b417ac2df09a56c1cf2dd42`;
- report:
  `f6eecb906a3d95ed13f84c0b483de665ed4d1ad577f95c1574bdc09ce1835674`;
- summary:
  `387eaca0f9a2962f395fd885e7f1fe49371a25a5ff9024f4d3b0d7e9f6efeb9d`;
- prospective diagnostics:
  `7f6b6966fce13433d12f6de5c59629dc79a7b79b9d30d88c77abc0cdb5c96c82`;
- prospective diagnostic rows:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- observations:
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`;
- observation decision rows:
  `21eaa26cdc3998b17fb045e9f3b11a21ed6a69bcdb090ca6b0f86f7667bb9c3e`.
