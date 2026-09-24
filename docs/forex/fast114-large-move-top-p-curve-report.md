# Fast114 — curva fina de top-p no ramo large-move

Resposta objetiva: ainda não temos estratégia. A calibração fina do top-p não
encontrou uma faixa mágica entre raridade e qualidade.

Fast114 testou a lacuna deixada por Fast98/Fast100/Fast107:

- Fast98 avaliou `top 5%`, `10%` e `20%`;
- Fast100/Fast107 mostraram que abrir para `20%` piora;
- Fast114 testou a curva fina `1%`, `2%`, `3%`, `5%`, `7.5%` e `10%` no ramo
  de operações maiores.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `approved_strategy=false`;
- sem Demo;
- sem Testnet;
- sem LIVE;
- sem envio de ordem.

## Configuração

Runner:

`backend/scripts/research/run_book_payoff_model_scout.py`

Janelas:

- fast16, fast20, fast21, fast22, fast24, fast28, fast29, fast32, fast35,
  fast36, fast82, fast83, fast89.

Parâmetros:

- horizonte: `300s`;
- target / breakeven activation: `10` e `15` bps;
- trailing distance: `15` bps;
- top fractions: `0.01`, `0.02`, `0.03`, `0.05`, `0.075`, `0.10`;
- modelos: `hist_gradient_payoff`, `ridge_payoff`;
- latency: `100ms`;
- custo de decisão: `1 bps`;
- custo de stress: `2 bps`;
- mínimo de janelas: `10`;
- mínimo operacional por janela: `3` execuções.

## Resultado

Status:

`PAYOFF_MODEL_FAILED_REPLICATION`

Motivos:

- `no_one_position_candidate_replicated_all_windows_at_1bps_cost`;
- `do_not_promote_to_strategy`.

Melhor candidato:

- modelo: `hist_gradient_payoff`;
- target: `10 bps`;
- trailing distance: `15 bps`;
- top fraction: `5%`.

Resumo one-position:

| Métrica | Valor |
|---|---:|
| Execuções | `42` |
| Net total @ 1 bps | `+13.763261447660312` bps |
| Net total @ 2 bps | `-28.236738552339688` bps |
| Janelas positivas @ 1 bps | `11/13` |
| Janelas positivas @ 2 bps | `11/13` |
| Menor janela @ 1 bps | `-14.226994710722115` bps |
| Janelas com mínimo de 3 execuções | `9/13` |
| Exit reasons | `1 BREAKEVEN_STOP`, `7 INITIAL_STOP`, `32 TIMEOUT`, `2 TRAILING_STOP` |

Resumo independente:

| Métrica | Valor |
|---|---:|
| Execuções | `364` |
| Net total @ 1 bps | `+553.0084850636933` bps |
| Net total @ 2 bps | `+189.00848506369334` bps |
| Janelas positivas @ 1 bps | `7/13` |
| Janelas positivas @ 2 bps | `7/13` |
| Janelas com mínimo de 3 execuções | `13/13` |

## Tabela dos melhores candidatos

Ordenado por janelas positivas em one-position sob stress e depois total:

| Modelo | Target | Top | Execuções OP | Net OP @1bps | Net OP @2bps | Pos OP @1bps | Pos OP @2bps | Min exec windows |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| hist_gradient_payoff | `10` | `5%` | `42` | `+13.7633` | `-28.2367` | `11/13` | `11/13` | `9/13` |
| hist_gradient_payoff | `15` | `7.5%` | `49` | `+7.0904` | `-41.9096` | `9/13` | `9/13` | `10/13` |
| hist_gradient_payoff | `15` | `2%` | `29` | `-40.8961` | `-69.8961` | `9/13` | `8/13` | `6/13` |
| hist_gradient_payoff | `15` | `1%` | `22` | `+5.6330` | `-16.3670` | `8/13` | `8/13` | `3/13` |
| hist_gradient_payoff | `10` | `3%` | `36` | `-37.9028` | `-73.9028` | `8/13` | `8/13` | `8/13` |
| hist_gradient_payoff | `15` | `3%` | `32` | `-78.0062` | `-110.0062` | `8/13` | `8/13` | `7/13` |

## Por que o melhor ainda não é estratégia

O candidato `target 10 / trail 15 / top 5%` melhorou a frequência de janelas
positivas, mas falhou em três pontos que importam para operação:

1. `net @2bps` ficou negativo no modo one-position;
2. só `9/13` janelas tiveram o mínimo operacional de 3 execuções;
3. a janela `fast36` quebrou a robustez:
   - execuções: `8`;
   - gross: `-105.81595768577692` bps;
   - net @ 1 bps: `-113.81595768577692` bps;
   - net @ 2 bps: `-121.81595768577692` bps;
   - exits: `1 BREAKEVEN_STOP`, `6 INITIAL_STOP`, `1 TRAILING_STOP`.

O problema, portanto, não é só "calibrar top-p". A região top `5%` continua
sendo a melhor das testadas, mas ainda contém regimes onde a entrada erra feio
e o trailing não salva.

## Decisão

Não aprovar estratégia. Não Demo. Não Testnet. Não LIVE.

Fast114 encerra a hipótese simples de que uma calibragem fina de top-p dentro do
mesmo modelo large-move resolveria o edge. A melhor faixa segue sendo `top 5%`,
mas ela precisa de uma mudança estrutural: ou mais dados/janelas, ou um modelo
de regime anterior à entrada que detecte janelas tipo `fast36` antes do trade.

## Comando executado

```bash
FAST114_WALS='fast16=backend/data/microstructure_v1/prospective-wal-fast16-18m,fast20=backend/data/microstructure_v1/prospective-wal-fast20-30m,fast21=backend/data/microstructure_v1/prospective-wal-fast21-30m,fast22=backend/data/microstructure_v1/prospective-wal-fast22-30m,fast24=backend/data/microstructure_v1/prospective-wal-fast24-30m,fast28=backend/data/microstructure_v1/prospective-wal-fast28-30m,fast29=backend/data/microstructure_v1/prospective-wal-fast29-30m,fast32=backend/data/microstructure_v1/prospective-wal-fast32-30m,fast35=backend/data/microstructure_v1/prospective-wal-fast35-30m,fast36=backend/data/microstructure_v1/prospective-wal-fast36-30m,fast82=backend/data/microstructure_v1/prospective-wal-fast82-30m,fast83=backend/data/microstructure_v1/prospective-wal-fast83-30m,fast89=backend/data/microstructure_v1/prospective-wal-fast89-30m'

PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_book_payoff_model_scout.py \
  --wal-roots "$FAST114_WALS" \
  --output-root backend/data/microstructure_v1/reports/fast114-large-move-top-p-curve-13w \
  --horizon-seconds 300 \
  --target-bps-values 10,15 \
  --trailing-distance-bps-values 15 \
  --top-fractions 0.01,0.02,0.03,0.05,0.075,0.10 \
  --decision-stride-seconds 5 \
  --latency-ms 100 \
  --minimum-windows 10 \
  --minimum-executable-per-window 3
```

## Arquivos e hashes

- report:
  `backend/data/microstructure_v1/reports/fast114-large-move-top-p-curve-13w/report.json`;
- report sha256:
  `3647861afd28b61db040fdbb1a9868eb7e0752964ad8a9720d9aa39ffb4e8a3b`.
