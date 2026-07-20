import argparse
from pathlib import Path

import pandas as pd

from mining.models import CustomerSequence


def build_customer_sequences(transactions: pd.DataFrame) -> list[CustomerSequence]:
    sequences: list[CustomerSequence] = []
    sorted_transactions = transactions.sort_values(
        ["customer_id", "invoice_datetime", "invoice_id", "item_id"],
        kind="mergesort",
    )
    for customer_id, customer_rows in sorted_transactions.groupby("customer_id", sort=True):
        itemsets: list[tuple[int, ...]] = []
        grouped_invoices = customer_rows.groupby(
            ["invoice_datetime", "invoice_id"],
            sort=True,
        )
        for _, invoice_rows in grouped_invoices:
            item_ids = tuple(sorted({int(item_id) for item_id in invoice_rows["item_id"]}))
            if item_ids:
                itemsets.append(item_ids)
        if itemsets:
            sequences.append(
                CustomerSequence(customer_id=str(customer_id), itemsets=tuple(itemsets))
            )
    return sequences


def format_sequence(sequence: CustomerSequence) -> str:
    tokens: list[str] = []
    for itemset in sequence.itemsets:
        tokens.extend(str(item_id) for item_id in itemset)
        tokens.append("-1")
    tokens.append("-2")
    return " ".join(tokens)


def write_sequence_db(sequences: list[CustomerSequence], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [format_sequence(sequence) for sequence in sequences]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SPMF-style sequence DB from CSV.")
    parser.add_argument("--transactions", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/processed/sequence-db.txt"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    transactions = pd.read_csv(args.transactions, parse_dates=["invoice_datetime"])
    write_sequence_db(build_customer_sequences(transactions), args.output)
    print(f"sequence_db: {args.output}")


if __name__ == "__main__":
    main()
