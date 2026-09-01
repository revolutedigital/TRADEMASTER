"""Deployment configuration invariants for the Railway backend service."""

import tomllib
from pathlib import Path


def test_railway_runs_database_migrations_before_starting_the_api() -> None:
    config_path = Path(__file__).parents[2] / "railway.toml"
    config = tomllib.loads(config_path.read_text())

    assert config["deploy"]["preDeployCommand"] == "alembic upgrade head"
    assert "uvicorn app.main:app" in config["deploy"]["startCommand"]
