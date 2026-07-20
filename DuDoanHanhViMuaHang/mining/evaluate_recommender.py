import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.repositories import SqliteRecommendationRepository


def compute_hit_rate_at_k(
    predictions: list[list[int]],
    ground_truth: list[set[int]],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if len(predictions) != len(ground_truth):
        raise ValueError(
            "predictions and ground_truth must have the same length"
        )
    if not predictions:
        return 0.0
    hits = 0
    for preds, truth in zip(predictions, ground_truth):
        top_k = preds[:k]
        if not truth:
            continue
        if any(item_id in truth for item_id in top_k):
            hits += 1
    return hits / len(predictions)


def compute_precision_at_k(
    predictions: list[list[int]],
    ground_truth: list[set[int]],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if len(predictions) != len(ground_truth):
        raise ValueError(
            "predictions and ground_truth must have the same length"
        )
    if not predictions:
        return 0.0
    total_precision = 0.0
    for preds, truth in zip(predictions, ground_truth):
        top_k = preds[:k]
        if not top_k:
            continue
        hits = sum(1 for item_id in top_k if item_id in truth)
        total_precision += hits / len(top_k)
    return total_precision / len(predictions)


def compute_recall_at_k(
    predictions: list[list[int]],
    ground_truth: list[set[int]],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be a positive integer")
    if len(predictions) != len(ground_truth):
        raise ValueError(
            "predictions and ground_truth must have the same length"
        )
    if not predictions:
        return 0.0
    total_recall = 0.0
    for preds, truth in zip(predictions, ground_truth):
        top_k = preds[:k]
        if not truth:
            continue
        hits = sum(1 for item_id in top_k if item_id in truth)
        total_recall += hits / len(truth)
    return total_recall / len(predictions)


RecommendFn = Callable[[list[int], int], list[int]]


def evaluate_recommender(
    recommend_fn: RecommendFn,
    test_baskets: list[tuple[list[int], set[int]]],
    k: int,
) -> dict[str, float]:
    if not test_baskets:
        return {
            "basket_count": 0,
            "hit_rate@{}".format(k): 0.0,
            "precision@{}".format(k): 0.0,
            "recall@{}".format(k): 0.0,
        }
    predictions: list[list[int]] = []
    ground_truth: list[set[int]] = []
    for cart, held_out in test_baskets:
        predictions.append(recommend_fn(cart, k))
        ground_truth.append(held_out)
    return {
        "basket_count": len(test_baskets),
        "hit_rate@{}".format(k): round(compute_hit_rate_at_k(predictions, ground_truth, k), 6),
        "precision@{}".format(k): round(compute_precision_at_k(predictions, ground_truth, k), 6),
        "recall@{}".format(k): round(compute_recall_at_k(predictions, ground_truth, k), 6),
    }


class BaselineRecommender:
    def __init__(self, database_path: Path) -> None:
        self._repository = SqliteRecommendationRepository(database_path)

    def recommend(self, cart: list[int], top_k: int = 5) -> list[int]:
        excluded = set(cart)
        products = self._repository.list_fallback_products(excluded, top_k)
        return [product.item_id for product in products]


def compute_baseline_metrics(
    database_path: Path,
    test_baskets: list[tuple[list[int], set[int]]],
    k: int,
) -> dict[str, float]:
    recommender = BaselineRecommender(database_path)
    return evaluate_recommender(recommender.recommend, test_baskets, k)


def write_metrics_json(
    metrics: dict[str, object],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate recommendation quality offline.")
    parser.add_argument("--database", required=True, type=Path, help="Path to SQLite database")
    parser.add_argument("--test-baskets", required=True, type=Path, help="JSON file with test baskets")
    parser.add_argument("--k", type=int, default=5, help="Top-K cutoff")
    parser.add_argument("--output", type=Path, default=Path("data/reports/evaluation-metrics.json"))
    args = parser.parse_args()

    baskets_data = json.loads(args.test_baskets.read_text(encoding="utf-8"))
    test_baskets: list[tuple[list[int], set[int]]] = [
        (basket["cart"], set(basket["held_out"])) for basket in baskets_data
    ]

    baseline_metrics = compute_baseline_metrics(args.database, test_baskets, args.k)
    print(json.dumps(baseline_metrics, indent=2))
    write_metrics_json(baseline_metrics, args.output)
    print(f"metrics: {args.output}")
