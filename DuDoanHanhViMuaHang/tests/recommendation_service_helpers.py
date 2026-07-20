from pathlib import Path

import pandas as pd

from mining.models import ScoredPattern
from mining.sqlite_import_store import write_import_database


def write_recommendation_fixture(database_path: Path) -> None:
    products = pd.DataFrame(
        [
            product_row(1, "A", "Apple", 2.0, 8, "United Kingdom"),
            product_row(2, "B", "Bread", 4.0, 6, "United Kingdom"),
            product_row(6, "C", "Coffee", 10.0, 4, "France"),
            product_row(7, "D", "Dates", 5.0, 12, "France"),
            product_row(8, "E", "Eggs", 1.0, 20, "Germany"),
            product_row(9, "F", "Flour", 1.0, 1, "Germany"),
        ]
    )
    patterns = [
        pattern((1,), (6,), 3, 0.3, 1.0, 3.0, "1 -1 6 -1 #SUP: 3"),
        pattern((1, 2), (6,), 4, 0.4, 1.0, 4.0, "1 2 -1 6 -1 #SUP: 4"),
        pattern((1,), (7,), 2, 0.2, 0.5, 1.0, "1 -1 7 -1 #SUP: 2"),
        pattern((7,), (8,), 5, 0.5, 0.1, 0.5, "7 -1 8 -1 #SUP: 5"),
        pattern((7,), (9,), 5, 0.5, 0.1, 0.5, "7 -1 9 -1 #SUP: 5"),
    ]
    write_import_database(
        database_path=database_path,
        products=products,
        patterns=patterns,
        dataset_name="fixture",
        minsup=0.2,
        sequence_count=10,
    )


def product_row(
    item_id: int,
    source_id: str,
    name: str,
    unit_price: float,
    transaction_count: int,
    country: str,
) -> dict[str, object]:
    return {
        "item_id": item_id,
        "product_source_id": source_id,
        "product_name": name,
        "unit_price": unit_price,
        "transaction_count": transaction_count,
        "country": country,
    }


def pattern(
    antecedent: tuple[int, ...],
    consequent: tuple[int, ...],
    support: int,
    support_ratio: float,
    weight_value: float,
    pattern_score: float,
    raw_pattern: str,
) -> ScoredPattern:
    return ScoredPattern(
        antecedent=antecedent,
        consequent=consequent,
        support=support,
        support_ratio=support_ratio,
        weight_value=weight_value,
        pattern_score=pattern_score,
        raw_pattern=raw_pattern,
    )
