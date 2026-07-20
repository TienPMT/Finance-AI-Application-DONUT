import argparse
from pathlib import Path

import pandas as pd

from mining.build_sequence_db import build_customer_sequences, write_sequence_db
from mining.dataset_loader import load_dataset
from mining.models import PipelineOutputs

NON_PRODUCT_SOURCE_IDS = frozenset(
    {"POST", "D", "DOT", "M", "BANK CHARGES", "ADJUST", "ADJUST2"}
)

ROOT = Path(__file__).resolve().parents[1]
APPROVED_INPUT_DIRS = (
    ROOT / ".claude" / "reference" / "datasets",
    ROOT / "data" / "raw",
)
APPROVED_OUTPUT_DIRS = (ROOT / "data" / "processed",)


def clean_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned = cleaned.dropna(
        subset=[
            "invoice_id",
            "customer_id",
            "product_source_id",
            "product_name",
            "quantity",
            "unit_price",
            "invoice_datetime",
        ]
    )
    required_text_columns = [
        "invoice_id",
        "customer_id",
        "product_source_id",
        "product_name",
    ]
    for column in required_text_columns:
        cleaned = cleaned[cleaned[column].astype(str).str.strip() != ""]
    invoice_text = cleaned["invoice_id"].astype(str).str.upper()
    cleaned = cleaned[~invoice_text.str.startswith("C")]
    cleaned = cleaned[(cleaned["quantity"] > 0) & (cleaned["unit_price"] > 0)]
    cleaned = cleaned[~cleaned["product_source_id"].isin(NON_PRODUCT_SOURCE_IDS)]
    cleaned = cleaned.drop_duplicates(
        subset=[
            "invoice_id",
            "customer_id",
            "product_source_id",
            "product_name",
            "quantity",
            "invoice_datetime",
            "unit_price",
            "country",
        ],
        keep="first",
    )
    cleaned = cleaned.sort_values(
        ["customer_id", "invoice_datetime", "invoice_id", "product_source_id"],
        kind="mergesort",
    )
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def build_product_map(
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    products = (
        transactions.sort_values(["product_source_id", "invoice_datetime"], kind="mergesort")
        .groupby("product_source_id", as_index=False)
        .agg(
            product_name=("product_name", "first"),
            unit_price=("unit_price", "median"),
            transaction_count=("invoice_id", "count"),
            country=("country", most_common_text),
        )
        .sort_values("product_source_id", kind="mergesort")
        .reset_index(drop=True)
    )
    products.insert(0, "item_id", range(1, len(products) + 1))
    item_map = products[["product_source_id", "item_id"]].copy()
    mapped = transactions.merge(item_map, on="product_source_id", how="inner")
    mapped = mapped.sort_values(
        ["customer_id", "invoice_datetime", "invoice_id", "item_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return mapped, products, item_map


def most_common_text(values: pd.Series) -> str:
    non_empty = values.dropna().astype(str)
    if non_empty.empty:
        return ""
    return non_empty.mode().sort_values().iloc[0]


def write_pipeline_outputs(
    transactions: pd.DataFrame,
    products: pd.DataFrame,
    item_map: pd.DataFrame,
    output_dir: Path,
) -> PipelineOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = PipelineOutputs(
        products=output_dir / "products.csv",
        transactions_clean=output_dir / "transactions-clean.csv",
        item_id_map=output_dir / "item-id-map.csv",
        sequence_db=output_dir / "sequence-db.txt",
    )
    products.to_csv(outputs.products, index=False)
    transactions.to_csv(outputs.transactions_clean, index=False)
    item_map.to_csv(outputs.item_id_map, index=False)
    write_sequence_db(build_customer_sequences(transactions), outputs.sequence_db)
    return outputs


def run_preprocess_pipeline(
    input_path: Path,
    output_dir: Path,
    dataset_type: str = "online-retail-ii",
    row_limit: int | None = None,
) -> dict[str, Path]:
    validate_row_limit(row_limit)
    raw = load_dataset(input_path, dataset_type=dataset_type, row_limit=row_limit)
    clean = clean_transactions(raw)
    mapped, products, item_map = build_product_map(clean)
    outputs = write_pipeline_outputs(mapped, products, item_map, output_dir)
    return {
        "products": outputs.products,
        "transactions_clean": outputs.transactions_clean,
        "item_id_map": outputs.item_id_map,
        "sequence_db": outputs.sequence_db,
    }


def validate_row_limit(row_limit: int | None) -> None:
    if row_limit is not None and row_limit <= 0:
        raise ValueError("row_limit must be positive when provided")


def validate_cli_paths(input_path: Path, output_dir: Path) -> None:
    resolved_input = input_path.resolve()
    resolved_output = output_dir.resolve()
    if not is_within_any(resolved_input, APPROVED_INPUT_DIRS):
        raise ValueError(
            f"Input path is outside approved project directories: {resolved_input}"
        )
    if not is_within_any(resolved_output, APPROVED_OUTPUT_DIRS):
        raise ValueError(
            f"Output path is outside approved project directories: {resolved_output}"
        )


def is_within_any(path: Path, roots: tuple[Path, ...]) -> bool:
    resolved_roots = [root.resolve() for root in roots]
    return any(path == root or root in path.parents for root in resolved_roots)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess retail invoices for CM-SPADE.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("data/processed"), type=Path)
    parser.add_argument("--dataset-type", default="online-retail-ii")
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument(
        "--allow-external-paths",
        action="store_true",
        help="Allow input/output paths outside approved project data directories.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_row_limit(args.row_limit)
    if not args.allow_external_paths:
        validate_cli_paths(args.input, args.output_dir)
    outputs = run_preprocess_pipeline(
        input_path=args.input,
        output_dir=args.output_dir,
        dataset_type=args.dataset_type,
        row_limit=args.row_limit,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
