"""docs/database_schema.md is generated from the DDL; it must not drift."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "gen_schema_doc", ROOT / "scripts" / "gen_schema_doc.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_schema_doc_is_current() -> None:
    gen = _load_generator()
    expected = gen.render()
    actual = (ROOT / "docs" / "database_schema.md").read_text()
    assert actual == expected, (
        "docs/database_schema.md is stale — run "
        "`python scripts/gen_schema_doc.py`")


def test_schema_doc_covers_every_column() -> None:
    """Every column in every table of the live DDL appears in the doc."""
    import sqlite3

    from simulator import database

    gen = _load_generator()
    doc = gen.render()
    conn = sqlite3.connect(":memory:")
    conn.executescript(database.SCHEMA)
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%'")]
    assert tables, "no tables parsed from SCHEMA"
    for table in tables:
        assert f"## `{table}`" in doc, f"table {table} missing from doc"
        for row in conn.execute(f"PRAGMA table_info({table})"):
            assert f"| `{row[1]}` |" in doc, f"{table}.{row[1]} missing from doc"
