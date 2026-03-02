"""Generate data catalog documentation from SQLAlchemy models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.base import Base
# Import all models to register them
import app.models.market  # noqa: F401
import app.models.trade  # noqa: F401
import app.models.portfolio  # noqa: F401
import app.models.signal  # noqa: F401
import app.models.audit  # noqa: F401
import app.models.api_key  # noqa: F401
import app.models.alert  # noqa: F401
import app.models.journal  # noqa: F401
import app.models.lineage  # noqa: F401


def generate_catalog() -> str:
    lines = [
        "# TradeMaster Data Catalog",
        "",
        "Auto-generated documentation of all database tables.",
        "",
        f"**Total tables: {len(Base.metadata.tables)}**",
        "",
    ]

    for table_name in sorted(Base.metadata.tables):
        table = Base.metadata.tables[table_name]
        lines.append(f"## `{table_name}`")
        lines.append("")
        lines.append("| Column | Type | Nullable | PK | FK |")
        lines.append("|--------|------|----------|----|----|")

        for col in table.columns:
            pk = "Yes" if col.primary_key else ""
            fk = ", ".join(str(f) for f in col.foreign_keys) if col.foreign_keys else ""
            nullable = "Yes" if col.nullable else "No"
            lines.append(f"| `{col.name}` | {col.type} | {nullable} | {pk} | {fk} |")

        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    catalog = generate_catalog()
    output_path = Path(__file__).parent.parent.parent / "docs" / "data-catalog.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(catalog)
    print(f"Data catalog written to {output_path}")
