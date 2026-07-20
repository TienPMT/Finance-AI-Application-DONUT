import json
from pathlib import Path

import pandas as pd
import pytest

from app.database import connect_database, fetch_one
from mining.import_sqlite import import_patterns_to_sqlite
from mining.score_patterns import ProductCatalogError
from tests.sqlite_import_helpers import fetch_table_counts, write_fixture_files


def test_importer_rebuilds_expected_sqlite_tables(tmp_path: Path) -> None:
    products_path, patterns_path = write_fixture_files(tmp_path)
    database_path = tmp_path / "app.sqlite"

    first = import_patterns_to_sqlite(
        products_path=products_path,
        patterns_path=patterns_path,
        database_path=database_path,
        dataset_name="fixture",
        minsup=0.5,
        sequence_count=6,
    )
    second = import_patterns_to_sqlite(
        products_path=products_path,
        patterns_path=patterns_path,
        database_path=database_path,
        dataset_name="fixture",
        minsup=0.5,
        sequence_count=6,
    )

    assert first.product_count == 3
    assert first.pattern_count == 1
    assert second.product_count == 3
    assert second.pattern_count == 1
    with connect_database(database_path) as connection:
        pattern = fetch_one(connection, "select * from patterns")
        run = fetch_one(connection, "select * from mining_runs")

    assert fetch_table_counts(database_path) == {
        "product_count": 3,
        "pattern_count": 1,
        "run_count": 1,
    }
    assert json.loads(pattern["antecedent"]) == [1]
    assert json.loads(pattern["consequent"]) == [6]
    assert pattern["support"] == 3
    assert pattern["support_ratio"] == 0.5
    assert pattern["pattern_score"] == 3.0
    assert run["dataset_name"] == "fixture"
    assert run["minsup"] == 0.5


def test_failed_import_preserves_existing_database(tmp_path: Path) -> None:
    products_path, patterns_path = write_fixture_files(tmp_path)
    database_path = tmp_path / "app.sqlite"
    import_patterns_to_sqlite(
        products_path=products_path,
        patterns_path=patterns_path,
        database_path=database_path,
        dataset_name="fixture",
        minsup=0.5,
        sequence_count=6,
    )
    before_bytes = database_path.read_bytes()
    bad_products = tmp_path / "bad-products.csv"
    pd.DataFrame(
        [
            {
                "item_id": 1,
                "product_source_id": "A",
                "product_name": "Apple",
                "unit_price": 2.0,
                "country": "United Kingdom",
            }
        ]
    ).to_csv(bad_products, index=False)

    with pytest.raises(ProductCatalogError, match="missing columns"):
        import_patterns_to_sqlite(
            products_path=bad_products,
            patterns_path=patterns_path,
            database_path=database_path,
            dataset_name="bad",
            minsup=0.5,
            sequence_count=6,
        )

    assert database_path.read_bytes() == before_bytes
    assert fetch_table_counts(database_path) == {
        "product_count": 3,
        "pattern_count": 1,
        "run_count": 1,
    }


def test_importer_rejects_duplicate_product_ids_before_db_swap(tmp_path: Path) -> None:
    products_path, patterns_path = write_fixture_files(tmp_path)
    products = pd.read_csv(products_path)
    duplicate_products = tmp_path / "duplicate-products.csv"
    pd.concat([products, products.head(1)], ignore_index=True).to_csv(
        duplicate_products,
        index=False,
    )

    with pytest.raises(ProductCatalogError, match="duplicate item IDs"):
        import_patterns_to_sqlite(
            products_path=duplicate_products,
            patterns_path=patterns_path,
            database_path=tmp_path / "app.sqlite",
            dataset_name="duplicate",
            minsup=0.5,
            sequence_count=6,
        )


def test_importer_filters_patterns_below_recorded_minsup(tmp_path: Path) -> None:
    products_path, patterns_path = write_fixture_files(tmp_path)
    database_path = tmp_path / "app.sqlite"
    patterns_path.write_text("1 -1 6 -1 #SUP: 1\n", encoding="utf-8")

    result = import_patterns_to_sqlite(
        products_path=products_path,
        patterns_path=patterns_path,
        database_path=database_path,
        dataset_name="fixture",
        minsup=0.9,
        sequence_count=10,
    )

    assert result.pattern_count == 0
    with connect_database(database_path) as connection:
        counts = fetch_one(
            connection,
            """
            select
                (select count(*) from patterns) as pattern_count,
                (select pattern_count from mining_runs) as recorded_pattern_count
            """,
        )
    assert counts == {"pattern_count": 0, "recorded_pattern_count": 0}


@pytest.mark.parametrize("minsup", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_importer_rejects_invalid_minsup(tmp_path: Path, minsup: float) -> None:
    products_path, patterns_path = write_fixture_files(tmp_path)

    with pytest.raises(ValueError, match="minsup must be in"):
        import_patterns_to_sqlite(
            products_path=products_path,
            patterns_path=patterns_path,
            database_path=tmp_path / "app.sqlite",
            dataset_name="fixture",
            minsup=minsup,
            sequence_count=6,
        )
