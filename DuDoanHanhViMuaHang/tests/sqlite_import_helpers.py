from pathlib import Path

import pandas as pd

from app.database import connect_database, fetch_one


def write_fixture_files(tmp_path: Path) -> tuple[Path, Path]:
    products_path = tmp_path / "products.csv"
    patterns_path = tmp_path / "patterns-raw.txt"
    pd.DataFrame(
        [
            {
                "item_id": 1,
                "product_source_id": "A",
                "product_name": "Apple",
                "unit_price": 2.0,
                "transaction_count": 5,
                "country": "United Kingdom",
            },
            {
                "item_id": 6,
                "product_source_id": "B",
                "product_name": "Bread",
                "unit_price": 10.0,
                "transaction_count": 3,
                "country": "United Kingdom",
            },
            {
                "item_id": 7,
                "product_source_id": "C",
                "product_name": "Coffee",
                "unit_price": 5.0,
                "transaction_count": 2,
                "country": "France",
            },
        ]
    ).to_csv(products_path, index=False)
    patterns_path.write_text(
        "1 -1 6 -1 #SUP: 3\n6 7 -1 #SUP: 2\n",
        encoding="utf-8",
    )
    return products_path, patterns_path


def fetch_table_counts(database_path: Path) -> dict[str, int]:
    with connect_database(database_path) as connection:
        return fetch_one(
            connection,
            """
            select
                (select count(*) from products) as product_count,
                (select count(*) from patterns) as pattern_count,
                (select count(*) from mining_runs) as run_count
            """,
        )
