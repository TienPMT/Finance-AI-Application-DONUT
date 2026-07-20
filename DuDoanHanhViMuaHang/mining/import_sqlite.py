import argparse
import math
from pathlib import Path

import pandas as pd

from app.database import DEFAULT_DATABASE_PATH
from mining.models import SqliteImportResult
from mining.parse_patterns import iter_parse_patterns_file
from mining.score_patterns import ProductCatalogError, score_patterns
from mining.sqlite_import_store import write_import_database

REQUIRED_PRODUCT_COLUMNS = {
    "item_id",
    "product_source_id",
    "product_name",
    "unit_price",
    "transaction_count",
    "country",
}


def import_patterns_to_sqlite(
    products_path: Path,
    patterns_path: Path,
    database_path: Path = DEFAULT_DATABASE_PATH,
    dataset_name: str = "unknown",
    minsup: float = 0.0,
    sequence_count: int = 1,
) -> SqliteImportResult:
    validate_import_metadata(minsup, sequence_count)
    products = pd.read_csv(products_path)
    validate_product_catalog(products)
    min_support = math.ceil(minsup * sequence_count)
    scored = score_patterns(
        iter_parse_patterns_file(patterns_path),
        products,
        sequence_count=sequence_count,
        min_support=min_support,
    )
    run_id = write_import_database(
        database_path=database_path,
        products=products,
        patterns=scored,
        dataset_name=dataset_name,
        minsup=minsup,
        sequence_count=sequence_count,
    )
    return SqliteImportResult(
        database_path=database_path,
        product_count=len(products),
        pattern_count=len(scored),
        run_id=run_id,
    )


def validate_product_catalog(products: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_PRODUCT_COLUMNS - set(products.columns))
    if missing:
        raise ProductCatalogError(f"Product catalog missing columns: {missing}")
    if products.empty:
        raise ProductCatalogError("Product catalog is empty")
    item_ids = products["item_id"].astype(int)
    duplicated = sorted(item_ids[item_ids.duplicated()].unique().tolist())
    if duplicated:
        raise ProductCatalogError(f"Product catalog has duplicate item IDs: {duplicated}")
    if products["transaction_count"].astype(int).lt(0).any():
        raise ProductCatalogError("Product transaction_count must be non-negative")


def validate_import_metadata(minsup: float, sequence_count: int) -> None:
    if not math.isfinite(minsup) or minsup <= 0 or minsup > 1:
        raise ValueError("minsup must be in (0, 1]")
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import scored CM-SPADE output into SQLite.")
    parser.add_argument("--products", required=True, type=Path)
    parser.add_argument("--patterns", required=True, type=Path)
    parser.add_argument("--database", default=DEFAULT_DATABASE_PATH, type=Path)
    parser.add_argument("--dataset-name", default="unknown")
    parser.add_argument("--minsup", required=True, type=float)
    parser.add_argument("--sequence-count", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = import_patterns_to_sqlite(
        products_path=args.products,
        patterns_path=args.patterns,
        database_path=args.database,
        dataset_name=args.dataset_name,
        minsup=args.minsup,
        sequence_count=args.sequence_count,
    )
    print(f"database: {result.database_path}")
    print(f"product_count: {result.product_count}")
    print(f"pattern_count: {result.pattern_count}")


if __name__ == "__main__":
    main()
