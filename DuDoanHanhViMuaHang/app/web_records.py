import json
import sqlite3

from app.repositories import PatternRecord, ProductRecord


def product_from_row(row: sqlite3.Row) -> ProductRecord:
    return ProductRecord(
        item_id=int(row["item_id"]),
        product_source_id=str(row["product_source_id"]),
        product_name=str(row["product_name"]),
        unit_price=float(row["unit_price"]),
        normalized_weight=float(row["normalized_weight"]),
        transaction_count=int(row["transaction_count"]),
        country=str(row["country"] or ""),
    )


def pattern_from_row(row: sqlite3.Row) -> PatternRecord:
    return PatternRecord(
        id=int(row["id"]),
        antecedent=tuple(int(value) for value in json.loads(row["antecedent"])),
        consequent=tuple(int(value) for value in json.loads(row["consequent"])),
        support=int(row["support"]),
        support_ratio=float(row["support_ratio"]),
        weight_value=float(row["weight_value"]),
        pattern_score=float(row["pattern_score"]),
        raw_pattern=str(row["raw_pattern"]),
    )
