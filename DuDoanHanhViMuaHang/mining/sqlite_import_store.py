import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from app.database import connect_database
from mining.models import ScoredPattern
from mining.score_patterns import build_product_weights


def write_import_database(
    database_path: Path,
    products: pd.DataFrame,
    patterns: list[ScoredPattern],
    dataset_name: str,
    minsup: float,
    sequence_count: int,
) -> int:
    temp_path = create_temp_database_path(database_path)
    try:
        with connect_database(temp_path) as connection:
            rebuild_schema(connection)
            insert_products(connection, products)
            run_id = insert_mining_run(
                connection,
                dataset_name=dataset_name,
                minsup=minsup,
                sequence_count=sequence_count,
                product_count=len(products),
                pattern_count=len(patterns),
            )
            insert_patterns(connection, run_id, patterns)
            connection.commit()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_path, database_path)
        return run_id
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def create_temp_database_path(database_path: Path) -> Path:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(
        prefix=f".{database_path.name}.",
        suffix=".tmp",
        dir=database_path.parent,
        delete=False,
    )
    temp_file.close()
    return Path(temp_file.name)


def rebuild_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        drop table if exists patterns;
        drop table if exists products;
        drop table if exists mining_runs;

        create table products (
            item_id integer primary key,
            product_source_id text not null,
            product_name text not null,
            unit_price real not null,
            normalized_weight real not null,
            transaction_count integer not null,
            country text
        );

        create table mining_runs (
            id integer primary key autoincrement,
            dataset_name text not null,
            minsup real not null,
            sequence_count integer not null,
            product_count integer not null,
            pattern_count integer not null,
            run_timestamp text not null default current_timestamp
        );

        create table patterns (
            id integer primary key autoincrement,
            mining_run_id integer not null references mining_runs(id) on delete cascade,
            antecedent text not null,
            consequent text not null,
            support integer not null,
            support_ratio real not null,
            weight_value real not null,
            pattern_score real not null,
            raw_pattern text not null
        );
        """
    )


def insert_products(connection: sqlite3.Connection, products: pd.DataFrame) -> None:
    weights = build_product_weights(products)
    rows = []
    for row in products.itertuples(index=False):
        rows.append(
            (
                int(row.item_id),
                str(row.product_source_id),
                str(row.product_name),
                float(row.unit_price),
                weights[int(row.item_id)],
                int(row.transaction_count),
                str(row.country) if pd.notna(row.country) else "",
            )
        )
    connection.executemany(
        """
        insert into products (
            item_id, product_source_id, product_name, unit_price,
            normalized_weight, transaction_count, country
        )
        values (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def insert_mining_run(
    connection: sqlite3.Connection,
    dataset_name: str,
    minsup: float,
    sequence_count: int,
    product_count: int,
    pattern_count: int,
) -> int:
    cursor = connection.execute(
        """
        insert into mining_runs (
            dataset_name, minsup, sequence_count, product_count, pattern_count
        )
        values (?, ?, ?, ?, ?)
        """,
        (dataset_name, minsup, sequence_count, product_count, pattern_count),
    )
    return int(cursor.lastrowid)


def insert_patterns(
    connection: sqlite3.Connection,
    mining_run_id: int,
    patterns: list[ScoredPattern],
) -> None:
    rows = [
        (
            mining_run_id,
            json.dumps(list(pattern.antecedent)),
            json.dumps(list(pattern.consequent)),
            pattern.support,
            pattern.support_ratio,
            pattern.weight_value,
            pattern.pattern_score,
            pattern.raw_pattern,
        )
        for pattern in patterns
    ]
    connection.executemany(
        """
        insert into patterns (
            mining_run_id, antecedent, consequent, support, support_ratio,
            weight_value, pattern_score, raw_pattern
        )
        values (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
