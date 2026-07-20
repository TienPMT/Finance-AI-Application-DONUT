import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.database import connect_database


@dataclass(frozen=True)
class ProductRecord:
    item_id: int
    product_source_id: str
    product_name: str
    unit_price: float
    normalized_weight: float
    transaction_count: int
    country: str

    @property
    def fallback_score(self) -> float:
        return self.transaction_count * self.normalized_weight


@dataclass(frozen=True)
class PatternRecord:
    id: int
    antecedent: tuple[int, ...]
    consequent: tuple[int, ...]
    support: int
    support_ratio: float
    weight_value: float
    pattern_score: float
    raw_pattern: str


class SqliteRecommendationRepository:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def list_patterns(self) -> list[PatternRecord]:
        with connect_database(self.database_path) as connection:
            run_id = self._latest_run_id(connection)
            if run_id is None:
                return []
            rows = connection.execute(
                """
                select id, antecedent, consequent, support, support_ratio,
                       weight_value, pattern_score, raw_pattern
                from patterns
                where mining_run_id = ?
                order by id
                """,
                (run_id,),
            ).fetchall()
        return [self._pattern_from_row(row) for row in rows]

    def get_products_by_ids(self, item_ids: set[int]) -> dict[int, ProductRecord]:
        if not item_ids:
            return {}
        placeholders = ",".join("?" for _ in item_ids)
        with connect_database(self.database_path) as connection:
            rows = connection.execute(
                f"""
                select item_id, product_source_id, product_name, unit_price,
                       normalized_weight, transaction_count, country
                from products
                where item_id in ({placeholders})
                """,
                tuple(sorted(item_ids)),
            ).fetchall()
        return {int(row["item_id"]): self._product_from_row(row) for row in rows}

    def list_fallback_products(
        self,
        excluded_item_ids: set[int],
        limit: int,
    ) -> list[ProductRecord]:
        with connect_database(self.database_path) as connection:
            rows = connection.execute(
                """
                select item_id, product_source_id, product_name, unit_price,
                       normalized_weight, transaction_count, country
                from products
                order by (transaction_count * normalized_weight) desc,
                         transaction_count desc,
                         item_id asc
                """
            ).fetchall()
        products = [self._product_from_row(row) for row in rows]
        return [
            product
            for product in products
            if product.item_id not in excluded_item_ids
        ][:limit]

    def _latest_run_id(self, connection: sqlite3.Connection) -> int | None:
        row = connection.execute("select max(id) as id from mining_runs").fetchone()
        if row is None or row["id"] is None:
            return None
        return int(row["id"])

    def _pattern_from_row(self, row: sqlite3.Row) -> PatternRecord:
        return PatternRecord(
            id=int(row["id"]),
            antecedent=tuple(int(item_id) for item_id in json.loads(row["antecedent"])),
            consequent=tuple(int(item_id) for item_id in json.loads(row["consequent"])),
            support=int(row["support"]),
            support_ratio=float(row["support_ratio"]),
            weight_value=float(row["weight_value"]),
            pattern_score=float(row["pattern_score"]),
            raw_pattern=str(row["raw_pattern"]),
        )

    def _product_from_row(self, row: sqlite3.Row) -> ProductRecord:
        return ProductRecord(
            item_id=int(row["item_id"]),
            product_source_id=str(row["product_source_id"]),
            product_name=str(row["product_name"]),
            unit_price=float(row["unit_price"]),
            normalized_weight=float(row["normalized_weight"]),
            transaction_count=int(row["transaction_count"]),
            country=str(row["country"] or ""),
        )
