# Fast104 — diagnóstico de veto fixo para outliers do candidato maior

Resposta objetiva: ainda não temos estratégia aprovada. Temos uma hipótese
melhor e mais concreta: o ramo de operações maiores melhora muito quando a
entrada evita micro-regimes de queda imediata, mas ainda falha em fast24 e
fast36. Isso é candidato para pré-registro novo, não permissão para operar.

Fast101 havia mostrado que o candidato `target 10 bps`, trailing distance
`15 bps`, top `5%`, horizonte `300s`, tinha sinal em `11/13` janelas, mas era
destruído por outliers, principalmente fast36. Fast104 testou se vetos causais
simples, com thresholds calculados somente no treino de cada fold, removem
risco de cauda suficiente.

## Safety boundary

Mantido:

- `research_only=true`;
- `order_submission_allowed=false`;
- `execution_authorization=none`;
- `orders_sent=false`;
- `approved_strategy=false`;
- `dry_run=true`;
- sem Demo;
- sem Testnet;
- sem LIVE;
- sem envio de ordem.

## Comando

```bash
PYTHONPATH=backend ./backend/.venv/bin/python \
  backend/scripts/research/run_fast104_fixed_outlier_veto_audit.py
```

Configuração default:

- janelas: fast16, fast20, fast21, fast22, fast24, fast28, fast29, fast32,
  fast35, fast36, fast82, fast83, fast89;
- modelo: `hist_gradient_payoff`;
- horizonte: `300s`;
- alvo: `10 bps`;
- trailing distance: `15 bps`;
- top fraction: `0.05`;
- latency: `100ms`;
- custo de lucro: `1 bps`;
- custo de stress: `2 bps`;
- mínimo: `11` janelas;
- mínimo por janela: `3` execuções.

## Resultado

Status:

`FAST104_FIXED_OUTLIER_VETO_FAILED_REPLICATION`

Motivos gravados no report:

- `not_approved_strategy`;
- `research_only`;
- `fixed_outlier_veto_discovery_diagnostic`;
- `positive_result_requires_fresh_preregistration`;
- `best_policy:spot_return_1s_not_crash30`;
- `profit_positive_holdouts:11/13`;
- `stress_positive_holdouts:11/13`;
- `do_not_promote_to_strategy`.

Melhor política encontrada:

`spot_return_1s_not_crash30`

Regra:

```text
spot_return_1s_bps >= q0.30 do conjunto selecionado no treino
```

Na prática, nos holdouts impressos, o threshold materializado foi `0.0`. Ou
seja: o veto remove entradas quando o retorno spot de 1 segundo já vem negativo.

## Comparação contra sem gate

| Política | Execuções | Net @ 1 bps | Net @ 2 bps | Positivas @ 1 bps | Positivas @ 2 bps | Janelas com >=3 execuções | Pior janela @ 1 bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_gate` | `42` | `+13.763261447660312` | `-28.236738552339688` | `11/13` | `11/13` | `9/13` | `-113.81595768577692` |
| `spot_return_1s_not_crash30` | `37` | `+105.4141767734227` | `+68.4141767734227` | `11/13` | `11/13` | `9/13` | `-42.99999999999976` |

Isso é materialmente melhor que o candidato cru:

- reduz a perda de fast36 de `-113.81595768577692` para `-42.99999999999976`;
- mantém positivo no agregado mesmo com `2 bps` de custo;
- preserva `11/13` janelas positivas.

Mas ainda não passa como estratégia:

- fast24 segue negativo;
- fast36 segue negativo;
- só `9/13` janelas têm o mínimo de `3` execuções;
- o teste é diagnóstico pós-observação dos outliers, não pré-registrado;
- o próprio report marca `do_not_promote_to_strategy`.

## Resultado por holdout da melhor política

| Holdout | Sinais antes do gate | Sinais depois do gate | Execuções one-position | Net @ 1 bps | Net @ 2 bps |
|---|---:|---:|---:|---:|---:|
| fast16 | `15` | `14` | `1` | `+4.343264375832244` | `+3.343264375832244` |
| fast20 | `30` | `24` | `3` | `+23.845763044504306` | `+20.845763044504306` |
| fast21 | `30` | `23` | `4` | `+38.597892432473756` | `+34.597892432473756` |
| fast22 | `30` | `29` | `2` | `+12.13462360289153` | `+10.13462360289153` |
| fast24 | `30` | `29` | `3` | `-14.016812888603907` | `-17.01681288860391` |
| fast28 | `30` | `22` | `2` | `+11.731181626076834` | `+9.731181626076834` |
| fast29 | `30` | `24` | `3` | `+7.601176116289523` | `+4.601176116289523` |
| fast32 | `27` | `17` | `1` | `+5.66881965552305` | `+4.66881965552305` |
| fast35 | `30` | `23` | `4` | `+9.926685299494059` | `+5.926685299494059` |
| fast36 | `26` | `6` | `3` | `-42.99999999999976` | `-45.99999999999976` |
| fast82 | `29` | `21` | `4` | `+22.347112316766506` | `+18.347112316766506` |
| fast83 | `29` | `19` | `4` | `+17.269432583112973` | `+13.269432583112973` |
| fast89 | `28` | `26` | `3` | `+8.96503860906158` | `+5.96503860906158` |

## Outras políticas testadas

| Política | Execuções | Net @ 1 bps | Net @ 2 bps | Positivas @ 1 bps | Positivas @ 2 bps | Janelas com >=3 execuções | Pior janela @ 1 bps |
|---|---:|---:|---:|---:|---:|---:|---:|
| `spread_widening_5s_low80` | `39` | `+89.4462599187955` | `+50.446259918795505` | `10/13` | `10/13` | `9/13` | `-30.64422527491125` |
| `realized_vol_30s_low80` | `33` | `+105.7253125338982` | `+72.7253125338982` | `10/13` | `10/13` | `8/13` | `-14.016812888603907` |
| `quote_volume_1s_low80` | `35` | `+70.02303852094197` | `+35.02303852094197` | `10/13` | `10/13` | `7/13` | `-41.99999999999864` |
| `spot_quote_volume_1s_low80` | `34` | `+54.88428292726081` | `+20.884282927260813` | `10/13` | `9/13` | `7/13` | `-41.99999999999976` |
| `directed_spot_return_5s_not_crash30` | `38` | `+40.84341417588057` | `+2.843414175880568` | `9/13` | `9/13` | `9/13` | `-28.698716147088888` |
| `spread_low_and_spot_return_not_crash` | `35` | `+47.172392516055794` | `+12.172392516055794` | `9/13` | `9/13` | `9/13` | `-28.698716147088888` |

## Decisão

Não aprovar estratégia.

O que temos agora é:

1. uma família de entrada mais promissora: operação maior com target `10 bps`,
   trailing distance `15 bps`, top `5%`, horizonte `300s`;
2. um veto de regime promissor: não entrar quando o `spot_return_1s_bps` já está
   negativo no momento da entrada;
3. evidência de que custo continua crucial: sem gate, `1 bps` ainda fica
   positivo, mas `2 bps` já vira negativo;
4. evidência de que trailing stop sozinho não resolve entrada ruim: fast36
   continua perdendo mesmo depois do veto.

## Próximo passo recomendado

Pré-registrar Fast105 antes de olhar o resultado:

- entrada base: `hist_gradient_payoff`, target `10`, trailing `15`, top `5%`,
  horizonte `300s`;
- filtro obrigatório: `spot_return_1s_bps >= 0`;
- testar em janelas novas/baixadas depois do pré-registro;
- critério mínimo para virar estratégia de Testnet:
  - net positivo @ `2 bps`;
  - pelo menos `12/13` ou equivalente em novas janelas positivas;
  - nenhuma janela com perda catastrófica;
  - mínimo de execuções por janela;
  - ainda sem LIVE.

## Verificações

```bash
./backend/.venv/bin/python -m pytest \
  backend/tests/unit/test_fast104_fixed_outlier_veto_audit.py -q --no-cov

./backend/.venv/bin/python -m ruff check \
  backend/scripts/research/run_fast104_fixed_outlier_veto_audit.py \
  backend/tests/unit/test_fast104_fixed_outlier_veto_audit.py
```

Resultado:

- `3 passed`;
- `All checks passed!`.

## Arquivos e hashes

- report:
  `backend/data/microstructure_v1/reports/fast104-fixed-outlier-veto-audit/report.json`;
- policy holdouts:
  `backend/data/microstructure_v1/reports/fast104-fixed-outlier-veto-audit/policy_holdouts.jsonl`;
- report sha256:
  `17a21f43e4c63fb27460c063cfa49c18ef88ada3aaeefe58e462711c4af90d35`;
- policy holdouts sha256:
  `84b010e3ab5dadf903b2ccec1c916e8cf56441e7da1830a43f11b5daf7677483`.
