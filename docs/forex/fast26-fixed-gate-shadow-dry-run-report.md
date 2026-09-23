# Rodada 26 — shadow dry-run online da regra fixa

Data: 2026-09-23  
Mercado: BTCUSDT futures, WAL prospectivo Fast24  
Resultado: a regra fixa gerou os mesmos 52 sinais em modo online/dry-run; nenhuma ordem foi enviada

## Pergunta

Depois do replay executável da Rodada 25, faltava provar que a candidata não depende do construtor
offline de labels para selecionar sinais. Esta rodada criou um runner shadow-only que:

- consome eventos cronológicos;
- calcula features causais no timestamp de decisão;
- aplica a regra congelada;
- emite decisões `would_enter`;
- não grava ledger;
- não envia ordem.

## Regra congelada

Fonte:

`backend/data/microstructure_v1/reports/book-regime-gate-confirmation-fast24-spot-age-f20-f21-f22-to-f24/report.json`

Política:

```text
Side: SELL
Base: directed_spot_perp_basis_bps >= 5,2032340279142275
Gate: spot_update_age_ms <= 140,48 ms
Horizon: 120s
Target: 5 bps
Stop inicial: 20 bps
```

## O que foi implementado

Script novo:

`backend/scripts/research/run_book_fixed_gate_shadow_policy.py`

O runner:

- valida que o relatório-fonte é `research_only=true`;
- exige `order_submission_allowed=false`;
- exige `execution_authorization=none`;
- exige `approved_strategy=false`;
- calcula `policy_sha256`;
- grava `shadow_decisions.jsonl` em dry-run;
- não tem modo de envio de ordem.

## Dry-run em Fast24

Comando:

`PYTHONPATH=backend backend/.venv/bin/python backend/scripts/research/run_book_fixed_gate_shadow_policy.py --policy-report backend/data/microstructure_v1/reports/book-regime-gate-confirmation-fast24-spot-age-f20-f21-f22-to-f24/report.json --wal-root backend/data/microstructure_v1/prospective-wal-fast24-30m --symbol BTCUSDT --start 2026-09-23T05:21:45Z --end 2026-09-23T05:49:15Z --decision-stride-seconds 5 --side SELL --output-root backend/data/microstructure_v1/reports/book-fixed-gate-shadow-policy-fast24-dry-run`

Resultado:

| Campo | Valor |
|---|---:|
| Decisões | 330 |
| Sinais `would_enter=true` | 52 |
| Policy SHA-256 | `82407bcc5d7fb895f51cc59aebd5e8e6444b06b0c2d9fea7120d36a043800e03` |
| `committed` | `false` |
| `dry_run` | `true` |
| `order_submission_allowed` | `false` |
| `execution_authorization` | `none` |

Isso bate com a seleção da confirmação Fast24 e com o replay executável: os mesmos 52 sinais.

Exemplo de decisão emitida:

```json
{
  "side": "SELL",
  "horizon_seconds": 120,
  "probability": 1.0,
  "threshold": 1.0,
  "would_enter": true
}
```

Aqui `probability=1.0` não é probabilidade calibrada de modelo; é codificação binária para registrar
uma regra fixa em formato compatível com shadow decisions.

## Decisão

A candidata agora tem três provas separadas:

1. seleção offline congelada;
2. replay executável por bid/ask;
3. emissão online/dry-run por features causais.

Ainda não é estratégia aprovada para Testnet real nem LIVE.

Próximo passo correto:

1. abrir uma partição prospectiva de shadow;
2. rodar esse runner em fluxo contínuo;
3. liquidar os sinais com o replay executável;
4. só depois avaliar Testnet.

Nenhuma ordem foi enviada.

## Artefatos

- Runner shadow:
  `backend/scripts/research/run_book_fixed_gate_shadow_policy.py`
- Testes:
  `backend/tests/unit/test_book_fixed_gate_shadow_policy.py`
- Relatório dry-run:
  `backend/data/microstructure_v1/reports/book-fixed-gate-shadow-policy-fast24-dry-run/report.json`
  (`bc87313609c0d0f5fbaef51c030e9ddd5d425f50e06faf8ac206bf339e3bfddd`)
- Decisões dry-run:
  `backend/data/microstructure_v1/reports/book-fixed-gate-shadow-policy-fast24-dry-run/shadow_decisions.jsonl`
  (`4600e4568aa3f63d8212593f83d35608d95e84e03689dfcee89e2de62c62b548`)
