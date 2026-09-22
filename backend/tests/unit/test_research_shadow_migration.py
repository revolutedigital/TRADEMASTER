"""Migration 021 adds only the constrained shadow-evidence table."""

import importlib.util
from pathlib import Path

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

from app.models import research_experiment  # noqa: F401
from app.models.base import Base


def load_migration(filename: str):
    path = Path(__file__).parents[2] / "alembic" / "versions" / filename
    spec = importlib.util.spec_from_file_location(filename, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run(connection, filename: str, direction: str) -> None:
    with Operations.context(MigrationContext.configure(connection)):
        getattr(load_migration(filename), direction)()


def test_shadow_migration_chains_matches_model_and_rolls_back() -> None:
    migration = load_migration("021_research_shadow_signals.py")
    assert (migration.revision, migration.down_revision) == ("021", "020")
    engine = sa.create_engine("sqlite://")
    with engine.begin() as connection:
        run(connection, "020_research_experiment_registry.py", "upgrade")
        run(connection, "021_research_shadow_signals.py", "upgrade")
        migrated = {
            column["name"]
            for column in sa.inspect(connection).get_columns("research_shadow_signals")
        }
        modelled = {
            column.name for column in Base.metadata.tables["research_shadow_signals"].columns
        }
        assert migrated == modelled
        run(connection, "021_research_shadow_signals.py", "downgrade")
        assert "research_shadow_signals" not in sa.inspect(connection).get_table_names()
