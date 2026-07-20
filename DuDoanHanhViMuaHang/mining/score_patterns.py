import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

from mining.models import ParsedPattern, ScoredPattern
from mining.parse_patterns import filter_recommendable_patterns, iter_parse_patterns_file


class ProductCatalogError(ValueError):
    pass


def build_product_weights(products: pd.DataFrame) -> dict[int, float]:
    require_columns(products, {"item_id", "unit_price"})
    prices = products[["item_id", "unit_price"]].copy()
    prices["item_id"] = prices["item_id"].astype(int)
    prices["unit_price"] = prices["unit_price"].astype(float)
    max_price = prices["unit_price"].max()
    if max_price <= 0:
        return {int(row.item_id): 0.0 for row in prices.itertuples(index=False)}
    return {
        int(row.item_id): round(float(row.unit_price) / float(max_price), 10)
        for row in prices.itertuples(index=False)
    }


def score_patterns(
    patterns: Iterable[ParsedPattern],
    products: pd.DataFrame,
    sequence_count: int,
    min_support: int = 1,
    max_itemsets: int = 3,
) -> list[ScoredPattern]:
    if sequence_count <= 0:
        raise ValueError("sequence_count must be positive")
    weights = build_product_weights(products)
    scored: list[ScoredPattern] = []
    for pattern in filter_recommendable_patterns(patterns, min_support, max_itemsets):
        if pattern.support > sequence_count:
            raise ValueError(
                "pattern support cannot exceed sequence_count: "
                f"support={pattern.support}, sequence_count={sequence_count}"
            )
        validate_known_items(pattern, weights)
        consequent_weights = [weights[item_id] for item_id in pattern.consequent]
        weight_value = sum(consequent_weights) / len(consequent_weights)
        support_ratio = pattern.support / sequence_count
        scored.append(
            ScoredPattern(
                antecedent=pattern.antecedent,
                consequent=pattern.consequent,
                support=pattern.support,
                support_ratio=round(support_ratio, 10),
                weight_value=round(weight_value, 10),
                pattern_score=round(pattern.support * weight_value, 10),
                raw_pattern=pattern.raw_pattern,
            )
        )
    return scored


def validate_known_items(pattern: ParsedPattern, weights: dict[int, float]) -> None:
    unknown = sorted(
        {
            item_id
            for itemset in pattern.itemsets
            for item_id in itemset
            if item_id not in weights
        }
    )
    if unknown:
        raise ProductCatalogError(f"Unknown item IDs in pattern: {unknown}")


def require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ProductCatalogError(f"Product catalog missing columns: {missing}")


def write_scored_patterns_csv(patterns: list[ScoredPattern], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "antecedent",
                "consequent",
                "support",
                "support_ratio",
                "weight_value",
                "pattern_score",
                "raw_pattern",
            ],
        )
        writer.writeheader()
        for pattern in patterns:
            writer.writerow(
                {
                    "antecedent": list(pattern.antecedent),
                    "consequent": list(pattern.consequent),
                    "support": pattern.support,
                    "support_ratio": pattern.support_ratio,
                    "weight_value": pattern.weight_value,
                    "pattern_score": pattern.pattern_score,
                    "raw_pattern": pattern.raw_pattern,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score CM-SPADE patterns with product weights.")
    parser.add_argument("--patterns", required=True, type=Path)
    parser.add_argument("--products", required=True, type=Path)
    parser.add_argument("--sequence-count", required=True, type=int)
    parser.add_argument("--output", default=Path("data/patterns/patterns-scored.csv"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    products = pd.read_csv(args.products)
    scored = score_patterns(
        iter_parse_patterns_file(args.patterns),
        products,
        sequence_count=args.sequence_count,
    )
    write_scored_patterns_csv(scored, args.output)
    print(f"scored_patterns: {args.output}")
    print(f"pattern_count: {len(scored)}")


if __name__ == "__main__":
    main()
