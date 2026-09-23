"""Latency-adjusted scout helpers keep selection explicit and safe."""

from __future__ import annotations

import pytest

from datetime import UTC, datetime

import pandas as pd

from app.services.research.research_dataset import ResearchDatasetConfig
from scripts.research.run_latency_adjusted_top_p_scout import (
    _consistent_label_config,
    run_portfolio_replay,
    select_portfolio_candidate,
)


def _result(
    *,
    feature_set: str,
    horizon_seconds: int,
    selected_count: int,
    target_rate: float,
    mean_roc_auc: float,
    top_p_monotonic: bool,
) -> dict[str, object]:
    return {
        "summary": {
            "feature_set": feature_set,
            "horizon_seconds": horizon_seconds,
            "mean_roc_auc": mean_roc_auc,
            "top_p_monotonic": top_p_monotonic,
            "tails": {
                "top_1%": {
                    "selected_count": selected_count,
                    "target_rate": target_rate,
                }
            },
        }
    }


def test_select_portfolio_candidate_prefers_monotonic_minimum_sample() -> None:
    results = [
        _result(
            feature_set="flow",
            horizon_seconds=120,
            selected_count=200,
            target_rate=0.80,
            mean_roc_auc=0.70,
            top_p_monotonic=False,
        ),
        _result(
            feature_set="flow_price_session",
            horizon_seconds=300,
            selected_count=75,
            target_rate=0.40,
            mean_roc_auc=0.75,
            top_p_monotonic=True,
        ),
    ]

    selection = select_portfolio_candidate(results, minimum_selected=50)

    assert selection["result_index"] == 1
    assert selection["selection_tier"] == 2
    assert selection["selection_tier_reason"] == "monotonic_and_minimum_selected"


def test_select_portfolio_candidate_falls_back_to_best_sampled_rate() -> None:
    results = [
        _result(
            feature_set="flow",
            horizon_seconds=120,
            selected_count=20,
            target_rate=0.90,
            mean_roc_auc=0.60,
            top_p_monotonic=True,
        ),
        _result(
            feature_set="flow_price",
            horizon_seconds=120,
            selected_count=80,
            target_rate=0.30,
            mean_roc_auc=0.80,
            top_p_monotonic=False,
        ),
    ]

    selection = select_portfolio_candidate(results, minimum_selected=50)

    assert selection["result_index"] == 1
    assert selection["selection_tier_reason"] == "minimum_selected_only"


def test_select_portfolio_candidate_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        select_portfolio_candidate([], minimum_selected=50)


def test_legacy_manifest_config_hash_uses_default_label_config(tmp_path) -> None:
    manifest = tmp_path / "research_rows.manifest.json"
    default_config = ResearchDatasetConfig()
    manifest.write_text(
        f'{{"schema_version":1,"config_sha256":"{default_config.sha256}"}}\n',
        encoding="utf-8",
    )

    label_config, source = _consistent_label_config([manifest])

    assert source == "legacy_manifest_default_config_sha256"
    assert label_config == default_config.path_label_config()


def test_old_legacy_manifest_hash_is_reported_as_inferred(tmp_path) -> None:
    manifest = tmp_path / "research_rows.manifest.json"
    manifest.write_text(
        '{"schema_version":1,"config_sha256":"bcb2a332e1a1bb3fca3d122eefaa990ec4f72788886e54c39aaa82fb41e93b3a"}\n',
        encoding="utf-8",
    )

    label_config, source = _consistent_label_config([manifest])

    assert source == "legacy_manifest_inferred_path_label_config"
    assert label_config == ResearchDatasetConfig().path_label_config()


def test_portfolio_replay_uses_manifest_cost_overrides(tmp_path) -> None:
    trade_root = tmp_path / "trades"
    partition = trade_root / "date=2026-01-01"
    partition.mkdir(parents=True)
    events = pd.DataFrame(
        {
            "event_time": [
                datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC),
                datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
                datetime(2026, 1, 1, 0, 0, 2, tzinfo=UTC),
            ],
            "sequence_id": [1, 2, 3],
            "product": ["usdm_perpetual", "usdm_perpetual", "usdm_perpetual"],
            "price": [100.0, 101.0, 101.0],
            "quantity": [1.0, 1.0, 1.0],
            "is_buyer_maker": [False, False, False],
        }
    )
    events.to_parquet(partition / "events.parquet", index=False)
    predictions = pd.DataFrame(
        {
            "decision_time_ms": [
                int(datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC).timestamp() * 1000)
            ],
            "side": ["BUY"],
            "horizon_seconds": [2],
            "probability": [1.0],
            "threshold_top_1%": [0.5],
            "threshold_top_2%": [0.5],
            "threshold_top_5%": [0.5],
            "threshold_top_10%": [0.5],
        }
    )

    results = run_portfolio_replay(
        predictions,
        trade_root=trade_root,
        latency_ms=0,
        expected_round_trip_bps=1.0,
        stress_round_trip_bps=2.0,
        output_root=tmp_path / "out",
    )

    best = max(results, key=lambda row: row["summary"]["expected_mean_bps"])
    assert best["summary"]["expected_mean_bps"] > 90
    assert best["summary"]["stress_mean_bps"] == pytest.approx(
        best["summary"]["expected_mean_bps"] - 1.0
    )
