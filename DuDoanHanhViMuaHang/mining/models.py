from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CustomerSequence:
    customer_id: str
    itemsets: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class PipelineOutputs:
    products: Path
    transactions_clean: Path
    item_id_map: Path
    sequence_db: Path


@dataclass(frozen=True)
class ParsedPattern:
    itemsets: tuple[tuple[int, ...], ...]
    support: int
    raw_pattern: str

    @property
    def antecedent(self) -> tuple[int, ...]:
        if len(self.itemsets) < 2:
            return ()
        return tuple(
            sorted({item_id for itemset in self.itemsets[:-1] for item_id in itemset})
        )

    @property
    def consequent(self) -> tuple[int, ...]:
        if len(self.itemsets) < 2:
            return ()
        return self.itemsets[-1]


@dataclass(frozen=True)
class ScoredPattern:
    antecedent: tuple[int, ...]
    consequent: tuple[int, ...]
    support: int
    support_ratio: float
    weight_value: float
    pattern_score: float
    raw_pattern: str


@dataclass(frozen=True)
class SqliteImportResult:
    database_path: Path
    product_count: int
    pattern_count: int
    run_id: int
