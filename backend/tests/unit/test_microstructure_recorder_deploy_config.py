"""Deployment config must collect every WAL stream required by the evidence gate."""

from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]


def test_recorder_deploy_commands_include_required_spot_trade_stream() -> None:
    dockerfile = (BACKEND_ROOT / "Dockerfile.recorder").read_text(encoding="utf-8")
    railway_config = (BACKEND_ROOT / "railway.recorder.toml").read_text(encoding="utf-8")

    assert "--include-spot-trades" in dockerfile
    assert "--include-spot-trades" in railway_config
    assert "--audit-status-path" in dockerfile
    assert "--audit-status-path" in railway_config
    assert "/data/microstructure-v1/prospective-audits/evidence-gate-status.json" in dockerfile
    assert (
        "/data/microstructure-v1/prospective-audits/evidence-gate-status.json"
        in railway_config
    )
