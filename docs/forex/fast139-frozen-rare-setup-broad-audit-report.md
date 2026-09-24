# Fast139 — frozen rare-setup broad audit

Fast139 auditou as regras raras já congeladas em Fast80 e Fast68 no universo
broad/near-threshold materializado pelo Fast92 all-local. O objetivo foi
responder:

> Setup raro salva a família de microestrutura quando abrimos o universo para
> entradas perto do threshold?

Status:

`FAST139_FROZEN_RARE_SETUP_NO_LATE_AUDIT_COVERAGE`

## Protocolo

Fonte:

- input:
  `backend/data/microstructure_v1/reports/fast92-broad-universe-scout-all-local/independent_settlements.jsonl`;
- linhas liquidadas: `6,579`;
- universo: decisões broad near-threshold, não estratégia deployável.

Regras congeladas testadas:

1. `fast80_rare_vol_flow_near200`
   - `realized_vol_30s_bps >= 45.38807193200911`;
   - `flow_imbalance_5s <= -0.2193072079643347`;
   - usando todo o universo near-threshold.

2. `fast80_rare_vol_flow_top_p_only`
   - mesma regra Fast80;
   - exige também `probability >= threshold`.

3. `fast68_directional_gate_near200`
   - `quote_volume_30s >= 79775575.91570444`;
   - `realized_vol_30s_bps >= 45.38807193200911`;
   - `realized_vol_30s_bps <= 351.4988811749231`;
   - `directed_flow_imbalance_30s >= -0.4391006273798949`;
   - usando todo o universo near-threshold.

4. `fast68_directional_gate_top_p_only`
   - mesma regra Fast68;
   - exige também `probability >= threshold`.

Execução:

- aplica regra congelada;
- aplica one-position por label;
- separa labels tardios `fast82+`;
- exige pelo menos `5` trades tardios para haver audit tardio interpretável.

## Resultado

| Regra | Raw selecionados | One-position trades | Stress total | Labels | Trades tardios |
|---|---:|---:|---:|---:|---:|
| `fast80_rare_vol_flow_near200` | `16` | `5` | `-10.4311` | `2` | `0` |
| `fast80_rare_vol_flow_top_p_only` | `14` | `5` | `-10.4311` | `2` | `0` |
| `fast68_directional_gate_near200` | `19` | `6` | `-21.0788` | `1` | `0` |
| `fast68_directional_gate_top_p_only` | `19` | `6` | `-21.0788` | `1` | `0` |

Nenhuma regra alcançou o mínimo de `5` trades tardios. Pior: nas janelas onde
elas dispararam, antes do recorte tardio, todas ficaram negativas em stress.

## Interpretação

Fast139 enfraquece a tese de que as regras raras congeladas em Fast79/Fast80 ou
Fast68 salvam o universo broad de microestrutura.

Leitura honesta:

- as regras são raras demais;
- não há cobertura em `fast82+`;
- quando há trades, o stress fica negativo;
- restringir para `top_p_only` não melhora;
- abrir para `near200` também não melhora.

Isso não prova que nenhum setup raro exista, mas mata estas regras congeladas
como caminho imediato. O próximo caminho, se seguir microestrutura, precisa ser
seleção walk-forward de novas regras com late holdout explícito — não reaproveitar
Fast80/Fast68 como se já fossem sólidos.

## Segurança

Fast139 é pesquisa offline:

- `research_only=true`;
- `approved_strategy=false`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `committed=false`;
- `dry_run=true`.

Nenhum trade foi enviado. Nenhum ledger foi alterado. O resultado não autoriza
Demo, Testnet nem LIVE.

## Artefatos

- runner:
  `backend/scripts/research/run_fast139_frozen_rare_setup_broad_audit.py`;
- teste:
  `backend/tests/unit/test_fast139_frozen_rare_setup_broad_audit.py`;
- report JSON:
  `backend/data/microstructure_v1/reports/fast139-frozen-rare-setup-broad-audit/report.json`;
- linhas one-position:
  `backend/data/microstructure_v1/reports/fast139-frozen-rare-setup-broad-audit/variant-one-position-rows.jsonl`.

Hashes:

- report:
  `803ee8a5aeb0e727ad50257c056ff1168b84dc1719a39735c9377eb54f4ae361`;
- linhas one-position:
  `f05c9cd7b704617364ddbb64ae2a45d2124aef5b5d81b73393ee4f0bd523e1a1`.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast139 descarta as regras raras congeladas Fast80/Fast68 como caminho imediato.
Continuar a busca exige uma nova seleção walk-forward, com late holdout separado
desde o início.
