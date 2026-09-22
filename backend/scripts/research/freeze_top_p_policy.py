"""Freeze a deterministic research-only top-p shadow policy artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.research.top_p_model import freeze_top_p_policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "backend" / "data" / "microstructure_v1" / "research-v1",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=("expected", "stress"), default="stress")
    parser.add_argument("--horizon-seconds", type=int, choices=(120, 300), required=True)
    parser.add_argument(
        "--feature-set",
        choices=(
            "flow",
            "flow_price",
            "flow_price_session",
            "flow_book",
            "flow_price_book",
            "flow_price_book_session",
        ),
        required=True,
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        choices=(0.01, 0.02, 0.05, 0.10),
        required=True,
    )
    parser.add_argument("--calibration-date", type=date.fromisoformat, required=True)
    parser.add_argument("--embargo-seconds", type=int, default=300)
    arguments = parser.parse_args()

    paths = sorted(arguments.dataset_root.glob("date=*/research_rows.parquet"))
    if not paths:
        parser.error("no research dataset partitions found")
    frame = pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)
    target_column = (
        "paid_expected_before_stop" if arguments.target == "expected" else "paid_stress_before_stop"
    )
    frame["target"] = frame[target_column].astype("int8")
    artifact = freeze_top_p_policy(
        frame,
        horizon_seconds=arguments.horizon_seconds,
        feature_set=arguments.feature_set,
        tail_fraction=arguments.tail_fraction,
        calibration_date=arguments.calibration_date.isoformat(),
        dataset_manifest_sha256=_combined_sha256(arguments.dataset_root, paths),
        target_column="target",
        embargo_seconds=arguments.embargo_seconds,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = arguments.output.with_suffix(f"{arguments.output.suffix}.tmp")
    temporary_path.write_text(artifact.to_json(), encoding="utf-8")
    temporary_path.replace(arguments.output)
    print(  # noqa: T201
        json.dumps(
            {
                "output": str(arguments.output),
                "model_sha256": artifact.model_sha256,
                "research_only": True,
                "order_submission_allowed": False,
                "execution_authorization": "none",
            },
            sort_keys=True,
        )
    )
    return 0


def _combined_sha256(root: Path, paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
