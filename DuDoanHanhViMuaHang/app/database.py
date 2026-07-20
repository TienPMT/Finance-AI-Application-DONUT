import sqlite3
from pathlib import Path
from typing import Any

DEFAULT_DATABASE_PATH = Path("database/app.sqlite")


def connect_database(database_path: Path = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("pragma foreign_keys = on")
    return connection


def fetch_one(
    connection: sqlite3.Connection,
    query: str,
    params: tuple[Any, ...] = (),
) -> dict[str, Any]:
    row = connection.execute(query, params).fetchone()
    if row is None:
        return {}
    return dict(row)


def fetch_all(
    connection: sqlite3.Connection,
    query: str,
    params: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(query, params).fetchall()]
