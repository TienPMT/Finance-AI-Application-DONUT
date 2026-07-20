from pathlib import Path

import pandas as pd

from mining.build_sequence_db import build_customer_sequences, write_sequence_db
from mining.dataset_loader import normalize_online_retail_frame
from mining.preprocess_data import (
    build_product_map,
    clean_transactions,
    run_preprocess_pipeline,
    validate_cli_paths,
)


def raw_invoice_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Invoice": "10003",
                "StockCode": "C",
                "Description": "Coffee",
                "Quantity": 1,
                "InvoiceDate": "2010-01-02 10:00",
                "Price": 4.5,
                "Customer ID": 2,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10001",
                "StockCode": "B",
                "Description": "Bread",
                "Quantity": 2,
                "InvoiceDate": "2010-01-01 10:00",
                "Price": 2.5,
                "Customer ID": 1,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10001",
                "StockCode": "A",
                "Description": "Apple",
                "Quantity": 3,
                "InvoiceDate": "2010-01-01 10:00",
                "Price": 1.2,
                "Customer ID": 1,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10002",
                "StockCode": "A",
                "Description": "Apple",
                "Quantity": 1,
                "InvoiceDate": "2010-01-02 09:00",
                "Price": 1.3,
                "Customer ID": 1,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10002",
                "StockCode": "A",
                "Description": "Apple",
                "Quantity": 1,
                "InvoiceDate": "2010-01-02 09:00",
                "Price": 1.3,
                "Customer ID": 1,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "c10004",
                "StockCode": "D",
                "Description": "Cancelled",
                "Quantity": 1,
                "InvoiceDate": "2010-01-03 09:00",
                "Price": 9.0,
                "Customer ID": 3,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10005",
                "StockCode": "E",
                "Description": "No Customer",
                "Quantity": 1,
                "InvoiceDate": "2010-01-03 09:00",
                "Price": 9.0,
                "Customer ID": None,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10006",
                "StockCode": "F",
                "Description": "Bad Quantity",
                "Quantity": 0,
                "InvoiceDate": "2010-01-03 09:00",
                "Price": 9.0,
                "Customer ID": 4,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10007",
                "StockCode": "G",
                "Description": "Bad Price",
                "Quantity": 1,
                "InvoiceDate": "2010-01-03 09:00",
                "Price": -1.0,
                "Customer ID": 5,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "   ",
                "StockCode": "H",
                "Description": "Blank Invoice",
                "Quantity": 1,
                "InvoiceDate": "2010-01-03 09:00",
                "Price": 1.0,
                "Customer ID": 6,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10008",
                "StockCode": "   ",
                "Description": "Blank Stock",
                "Quantity": 1,
                "InvoiceDate": "2010-01-03 09:00",
                "Price": 1.0,
                "Customer ID": 7,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10009",
                "StockCode": "I",
                "Description": "   ",
                "Quantity": 1,
                "InvoiceDate": "2010-01-03 09:00",
                "Price": 1.0,
                "Customer ID": 8,
                "Country": "United Kingdom",
            },
        ]
    )


def test_cleaner_removes_invalid_rows_and_normalizes_columns() -> None:
    normalized = normalize_online_retail_frame(raw_invoice_fixture())
    clean = clean_transactions(normalized)

    assert clean["product_source_id"].tolist() == ["A", "B", "A", "C"]
    assert len(clean) == len(clean.drop_duplicates())
    assert clean["invoice_id"].tolist() == ["10001", "10001", "10002", "10003"]
    assert clean["customer_id"].tolist() == ["1", "1", "1", "2"]
    assert clean["invoice_datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist() == [
        "2010-01-01 10:00",
        "2010-01-01 10:00",
        "2010-01-02 09:00",
        "2010-01-02 10:00",
    ]


def test_product_mapper_creates_stable_integer_ids() -> None:
    clean = clean_transactions(normalize_online_retail_frame(raw_invoice_fixture()))
    mapped, products, item_map = build_product_map(clean)

    assert item_map[["product_source_id", "item_id"]].to_dict("records") == [
        {"product_source_id": "A", "item_id": 1},
        {"product_source_id": "B", "item_id": 2},
        {"product_source_id": "C", "item_id": 3},
    ]
    assert products[["item_id", "product_source_id", "product_name"]].to_dict("records") == [
        {"item_id": 1, "product_source_id": "A", "product_name": "Apple"},
        {"item_id": 2, "product_source_id": "B", "product_name": "Bread"},
        {"item_id": 3, "product_source_id": "C", "product_name": "Coffee"},
    ]
    assert mapped["item_id"].tolist() == [1, 2, 1, 3]


def test_sequence_writer_emits_expected_spmf_format(tmp_path: Path) -> None:
    clean = clean_transactions(normalize_online_retail_frame(raw_invoice_fixture()))
    mapped, _, _ = build_product_map(clean)
    sequences = build_customer_sequences(mapped)
    output_path = tmp_path / "sequence-db.txt"

    write_sequence_db(sequences, output_path)

    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "1 2 -1 1 -1 -2",
        "3 -1 -2",
    ]


def test_cli_pipeline_writes_deterministic_outputs_with_row_limit(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "processed"
    raw_invoice_fixture().to_csv(input_path, index=False)

    first = run_preprocess_pipeline(input_path, output_dir, dataset_type="online-retail-ii", row_limit=4)
    second = run_preprocess_pipeline(input_path, output_dir, dataset_type="online-retail-ii", row_limit=4)

    expected_files = {
        "products.csv",
        "transactions-clean.csv",
        "item-id-map.csv",
        "sequence-db.txt",
    }
    assert {path.name for path in first.values()} == expected_files
    assert {path.name for path in second.values()} == expected_files
    assert (output_dir / "sequence-db.txt").read_text(encoding="utf-8").splitlines() == [
        "1 2 -1 1 -1 -2",
        "3 -1 -2",
    ]
    assert len(pd.read_csv(output_dir / "transactions-clean.csv")) == 4


def test_pipeline_rejects_invalid_row_limit(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    raw_invoice_fixture().to_csv(input_path, index=False)

    try:
        run_preprocess_pipeline(input_path, tmp_path / "processed", row_limit=0)
    except ValueError as error:
        assert "row_limit must be positive" in str(error)
    else:
        raise AssertionError("Expected invalid row_limit to fail")


def test_cli_path_validation_blocks_unapproved_external_output(tmp_path: Path) -> None:
    input_path = Path(".claude/reference/datasets/online_retail_II.xlsx")

    try:
        validate_cli_paths(input_path, tmp_path / "processed")
    except ValueError as error:
        assert "outside approved project directories" in str(error)
    else:
        raise AssertionError("Expected unapproved output path to fail")
