import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.database import connect_database
from app.recommendation_service import RecommendationService
from app.repositories import SqliteRecommendationRepository
from mining.build_sequence_db import build_customer_sequences, write_sequence_db
from mining.evaluate_recommender import (
    BaselineRecommender,
    compute_baseline_metrics,
    evaluate_recommender,
    write_metrics_json,
)
from mining.models import CustomerSequence
from mining.run_cmspade import run_cmspade
from mining.score_patterns import score_patterns, write_scored_patterns_csv
from mining.parse_patterns import iter_parse_patterns_file
from mining.sqlite_import_store import write_import_database


def split_sequences_last_basket(
    sequences: list[CustomerSequence],
) -> tuple[list[CustomerSequence], list[tuple[list[int], set[int]]]]:
    """Split sequences into train (all but last itemset) and test baskets.

    Each test basket is (cart_items, held_out_itemset) where cart_items
    contains all unique item IDs from the training portion. Sequences
    with fewer than 2 itemsets are skipped.
    """
    train_sequences: list[CustomerSequence] = []
    test_baskets: list[tuple[list[int], set[int]]] = []
    for sequence in sequences:
        if len(sequence.itemsets) < 2:
            continue
        train_itemsets = sequence.itemsets[:-1]
        held_out_itemset = sequence.itemsets[-1]
        train_sequences.append(
            CustomerSequence(
                customer_id=sequence.customer_id,
                itemsets=train_itemsets,
            )
        )
        cart_items = sorted(
            {item_id for itemset in train_itemsets for item_id in itemset}
        )
        held_out = set(held_out_itemset)
        test_baskets.append((cart_items, held_out))
    return train_sequences, test_baskets


def _recommend_from_service(
    service: RecommendationService,
    cart: list[int],
    top_k: int,
) -> list[int]:
    return [item.product_id for item in service.recommend(cart, top_k=top_k)]


def _run_single_minsup(
    minsup: float,
    train_seq_path: Path,
    products: pd.DataFrame,
    test_baskets: list[tuple[list[int], set[int]]],
    experiment_dir: Path,
    dataset_name: str,
    verbose: bool,
    timeout_seconds: int,
) -> dict[str, object]:
    patterns_raw_path = experiment_dir / "patterns-raw.txt"
    scored_path = experiment_dir / "patterns-scored.csv"
    database_path = experiment_dir / "app.sqlite"

    t0 = time.monotonic()
    mining_result = run_cmspade(
        train_seq_path,
        patterns_raw_path,
        minsup,
        verbose=verbose,
        timeout_seconds=timeout_seconds,
    )
    mining_runtime = round(time.monotonic() - t0, 3)

    scored = score_patterns(
        iter_parse_patterns_file(patterns_raw_path),
        products,
        sequence_count=mining_result.sequence_count,
        min_support=int(minsup * mining_result.sequence_count),
    )
    write_scored_patterns_csv(scored, scored_path)

    run_id = write_import_database(
        database_path=database_path,
        products=products,
        patterns=scored,
        dataset_name=dataset_name,
        minsup=minsup,
        sequence_count=mining_result.sequence_count,
    )

    k = 5
    baseline_metrics = compute_baseline_metrics(database_path, test_baskets, k)

    service = RecommendationService(SqliteRecommendationRepository(database_path))
    proposed_metrics = evaluate_recommender(
        lambda cart, top_k: _recommend_from_service(service, cart, top_k),
        test_baskets,
        k,
    )

    experiment_result = {
        "experiment": f"minsup-{minsup}",
        "dataset": dataset_name,
        "minsup": minsup,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mining": {
            "sequence_count": mining_result.sequence_count,
            "pattern_count": mining_result.pattern_count,
            "scored_pattern_count": len(scored),
            "runtime_seconds": mining_runtime,
            "run_id": run_id,
        },
        "evaluation": {
            "k": k,
            "test_basket_count": len(test_baskets),
            "baseline": baseline_metrics,
            "proposed": proposed_metrics,
        },
    }

    metrics_path = experiment_dir / "metrics.json"
    write_metrics_json(experiment_result, metrics_path)
    return experiment_result


def run_experiment(
    transactions_path: Path,
    products_path: Path,
    minsup_values: list[float],
    output_dir: Path,
    dataset_name: str = "online-retail-ii",
    verbose: bool = False,
    timeout_seconds: int = 300,
) -> list[dict[str, object]]:
    transactions = pd.read_csv(transactions_path, parse_dates=["invoice_datetime"])
    products = pd.read_csv(products_path)
    sequences = build_customer_sequences(transactions)
    train_sequences, test_baskets = split_sequences_last_basket(sequences)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_seq_path = output_dir / "train-sequence-db.txt"
    write_sequence_db(train_sequences, train_seq_path)

    results: list[dict[str, object]] = []

    for minsup in minsup_values:
        experiment_label = f"minsup-{minsup}"
        experiment_dir = output_dir / experiment_label
        experiment_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = _run_single_minsup(
                minsup=minsup,
                train_seq_path=train_seq_path,
                products=products,
                test_baskets=test_baskets,
                experiment_dir=experiment_dir,
                dataset_name=dataset_name,
                verbose=verbose,
                timeout_seconds=timeout_seconds,
            )
            results.append(result)
            if verbose:
                print(json.dumps(result, indent=2, default=str))
        except Exception as exc:
            error_result = {
                "experiment": experiment_label,
                "dataset": dataset_name,
                "minsup": minsup,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
            }
            results.append(error_result)
            metrics_path = experiment_dir / "metrics.json"
            write_metrics_json(error_result, metrics_path)
            print(f"  {experiment_label}: FAILED - {exc}")

    summary_path = output_dir / "experiment-results.json"
    summary = {
        "dataset": dataset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "minsup_values": minsup_values,
        "test_basket_count": len(test_baskets),
        "train_sequence_count": len(train_sequences),
        "experiments": results,
    }
    write_metrics_json(summary, summary_path)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CM-SPADE recommendation experiments with multiple minsup values."
    )
    parser.add_argument("--transactions", required=True, type=Path)
    parser.add_argument("--products", required=True, type=Path)
    parser.add_argument(
        "--minsup",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05],
        help="Minsup values to test (space-separated)",
    )
    parser.add_argument("--output-dir", default=Path("data/experiments"), type=Path)
    parser.add_argument("--dataset-name", default="online-retail-ii")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(
        transactions_path=args.transactions,
        products_path=args.products,
        minsup_values=args.minsup,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        verbose=args.verbose,
        timeout_seconds=args.timeout_seconds,
    )
    print(f"experiments: {len(results)}")
    for result in results:
        mining = result["mining"]
        eval_data = result["evaluation"]
        print(
            f"  {result['experiment']}: "
            f"patterns={mining['pattern_count']} "
            f"runtime={mining['runtime_seconds']}s "
            f"hit@{eval_data['k']}: baseline={eval_data['baseline']['hit_rate@5']:.4f} "
            f"proposed={eval_data['proposed']['hit_rate@5']:.4f}"
        )


if __name__ == "__main__":
    main()
