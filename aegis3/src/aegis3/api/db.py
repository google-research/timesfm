from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from aegis3.config import settings

engine = create_engine(settings.db_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session() -> Iterator[Session]:
    with SessionLocal() as s:
        yield s


def migrate() -> None:
    """Apply all SQL files in db/migrations in lexical order."""
    here = Path(__file__).resolve().parents[3]
    mig_dir = here / "db" / "migrations"
    files = sorted(mig_dir.glob("*.sql"))
    if not files:
        print(f"no migrations found at {mig_dir}", file=sys.stderr)
        sys.exit(1)
    with engine.begin() as conn:
        for f in files:
            print(f"applying {f.name}")
            conn.execute(text(f.read_text()))
    print("done.")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "migrate":
        migrate()
    else:
        print("usage: python -m aegis3.api.db migrate", file=sys.stderr)
        sys.exit(2)
