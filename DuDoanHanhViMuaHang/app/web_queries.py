import sqlite3
from dataclasses import dataclass
from pathlib import Path

from app.database import connect_database
from app.repositories import PatternRecord, ProductRecord
from app.web_records import pattern_from_row, product_from_row


class DataUnavailableError(RuntimeError):
    pass

@dataclass(frozen=True)
class DashboardStats:
    dataset_name: str
    minsup: float
    sequence_count: int
    product_count: int
    pattern_count: int
    run_timestamp: str


@dataclass(frozen=True)
class PatternView:
    pattern: PatternRecord
    antecedent_products: tuple[ProductRecord, ...]
    consequent_products: tuple[ProductRecord, ...]
    itemsets_products: tuple[tuple[ProductRecord, ...], ...]


class RetailAnalyticsQueries:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def dashboard_stats(self) -> DashboardStats:
        self._require_database()
        try:
            with connect_database(self.database_path) as connection:
                row = connection.execute(
                    """
                    select dataset_name, minsup, sequence_count, product_count,
                           pattern_count, run_timestamp
                    from mining_runs
                    order by id desc
                    limit 1
                    """
                ).fetchone()
        except sqlite3.Error as exc:
            raise self._unavailable() from exc
        if row is None:
            raise self._unavailable()
        return DashboardStats(
            dataset_name=str(row["dataset_name"]),
            minsup=float(row["minsup"]),
            sequence_count=int(row["sequence_count"]),
            product_count=int(row["product_count"]),
            pattern_count=int(row["pattern_count"]),
            run_timestamp=str(row["run_timestamp"]),
        )

    def list_products(
        self,
        query: str = "",
        min_price: float | None = None,
        max_price: float | None = None,
        min_popularity: int = 0,
        sort: str = "popular",
        limit: int = 60,
    ) -> list[ProductRecord]:
        self._require_database()
        clauses = ["transaction_count >= ?"]
        params: list[object] = [min_popularity]
        if query.strip():
            clauses.append(
                "(lower(product_name) like ? or lower(product_source_id) like ?)"
            )
            term = f"%{query.strip().lower()}%"
            params.extend([term, term])
        if min_price is not None:
            clauses.append("unit_price >= ?")
            params.append(min_price)
        if max_price is not None:
            clauses.append("unit_price <= ?")
            params.append(max_price)
        order_by = {
            "popular": "transaction_count desc, item_id asc",
            "price_asc": "unit_price asc, item_id asc",
            "price_desc": "unit_price desc, item_id asc",
        }[sort]
        params.append(limit)
        try:
            with connect_database(self.database_path) as connection:
                rows = connection.execute(
                    f"""
                    select item_id, product_source_id, product_name, unit_price,
                           normalized_weight, transaction_count, country
                    from products
                    where {" and ".join(clauses)}
                    order by {order_by}
                    limit ?
                    """,
                    tuple(params),
                ).fetchall()
        except sqlite3.Error as exc:
            raise self._unavailable() from exc
        return [product_from_row(row) for row in rows]

    def get_products(self, item_ids: set[int]) -> list[ProductRecord]:
        if not item_ids:
            return []
        self._require_database()
        placeholders = ",".join("?" for _ in item_ids)
        try:
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
        except sqlite3.Error as exc:
            raise self._unavailable() from exc
        products = {int(row["item_id"]): product_from_row(row) for row in rows}
        return [products[item_id] for item_id in sorted(item_ids) if item_id in products]

    def list_patterns(self, limit: int = 25) -> list[PatternView]:
        self._require_database()
        try:
            with connect_database(self.database_path) as connection:
                run = connection.execute(
                    "select max(id) as id from mining_runs"
                ).fetchone()
                if run is None or run["id"] is None:
                    raise self._unavailable()
                rows = connection.execute(
                    """
                    select id, antecedent, consequent, support, support_ratio,
                           weight_value, pattern_score, raw_pattern
                    from patterns
                    where mining_run_id = ?
                    order by pattern_score desc, support desc, id asc
                    limit ?
                    """,
                    (int(run["id"]), limit),
                ).fetchall()
                patterns = [pattern_from_row(row) for row in rows]
                item_ids = {
                    item_id
                    for pattern in patterns
                    for item_id in pattern.antecedent + pattern.consequent
                }
                products = self._products_by_id(connection, item_ids)
        except sqlite3.Error as exc:
            raise self._unavailable() from exc
        return [
            PatternView(
                pattern=pattern,
                antecedent_products=tuple(
                    products[item_id]
                    for item_id in pattern.antecedent
                    if item_id in products
                ),
                consequent_products=tuple(
                    products[item_id]
                    for item_id in pattern.consequent
                    if item_id in products
                ),
                itemsets_products=tuple(
                    tuple(
                        products[item_id]
                        for item_id in itemset
                        if item_id in products
                    )
                    for itemset in self._parse_raw_itemsets(pattern.raw_pattern)
                ),
            )
            for pattern in patterns
        ]

    def _require_database(self) -> None:
        if not self.database_path.is_file():
            raise self._unavailable()

    def _unavailable(self) -> DataUnavailableError:
        return DataUnavailableError(
            "Chưa có dữ liệu khai phá. Hãy chạy pipeline và nhập kết quả vào SQLite."
        )

    @staticmethod
    def _parse_raw_itemsets(raw_pattern: str) -> tuple[tuple[int, ...], ...]:
        itemsets: list[tuple[int, ...]] = []
        current: list[int] = []
        for token in raw_pattern.split():
            if token == "-1":
                if current:
                    itemsets.append(tuple(current))
                    current = []
                continue
            if token == "-2":
                continue
            try:
                item_id = int(token)
            except ValueError:
                continue
            if item_id > 0:
                current.append(item_id)
        if current:
            itemsets.append(tuple(current))
        return tuple(itemsets)

    def _products_by_id(
        self,
        connection: sqlite3.Connection,
        item_ids: set[int],
    ) -> dict[int, ProductRecord]:
        if not item_ids:
            return {}
        placeholders = ",".join("?" for _ in item_ids)
        rows = connection.execute(
            f"""
            select item_id, product_source_id, product_name, unit_price,
                   normalized_weight, transaction_count, country
            from products
            where item_id in ({placeholders})
            """,
            tuple(sorted(item_ids)),
        ).fetchall()
        return {int(row["item_id"]): product_from_row(row) for row in rows}
