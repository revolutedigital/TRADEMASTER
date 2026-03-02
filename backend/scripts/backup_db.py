#!/usr/bin/env python3
"""Database backup utility for TradeMaster."""

import os
import subprocess
import sys
from datetime import datetime


def backup_database():
    """Create a pg_dump backup of the TradeMaster database."""
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://trademaster:trademaster@localhost:5432/trademaster",
    )

    # Parse URL for pg_dump
    # Format: postgresql://user:pass@host:port/dbname
    backup_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backups")
    os.makedirs(backup_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = os.path.join(backup_dir, f"trademaster_{timestamp}.sql")

    print(f"Backing up database to {backup_file}...")

    try:
        result = subprocess.run(
            ["pg_dump", database_url, "-f", backup_file],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            size_mb = os.path.getsize(backup_file) / (1024 * 1024)
            print(f"Backup complete: {backup_file} ({size_mb:.1f} MB)")
        else:
            print(f"Backup failed: {result.stderr}", file=sys.stderr)
            sys.exit(1)

    except FileNotFoundError:
        print("Error: pg_dump not found. Install PostgreSQL client tools.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    backup_database()
