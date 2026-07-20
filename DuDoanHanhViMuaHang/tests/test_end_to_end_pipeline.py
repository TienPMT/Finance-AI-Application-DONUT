import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from mining.build_sequence_db import build_customer_sequences, write_sequence_db
from mining.dataset_loader import normalize_online_retail_frame
from mining.parse_patterns import iter_parse_patterns_file
from mining.preprocess_data import build_product_map, clean_transactions, run_preprocess_pipeline
from mining.run_cmspade import run_cmspade
from mining.score_patterns import score_patterns, write_scored_patterns_csv
from mining.sqlite_import_store import write_import_database


def fixture_raw_invoices() -> pd.DataFrame:
    """Small deterministic fixture for E2E pipeline testing.

    Customer 1: invoice 10001 (A, B), invoice 10002 (A) → sequence: {A,B} {A}
    Customer 2: invoice 10003 (C) → sequence: {C}
    """
    return pd.DataFrame(
        [
            {
                "Invoice": "10001",
                "StockCode": "A",
                "Description": "Apple",
                "Quantity": 2,
                "InvoiceDate": "2010-01-01 10:00",
                "Price": 1.5,
                "Customer ID": 1,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10001",
                "StockCode": "B",
                "Description": "Bread",
                "Quantity": 1,
                "InvoiceDate": "2010-01-01 10:00",
                "Price": 2.0,
                "Customer ID": 1,
                "Country": "United Kingdom",
            },
            {
                "Invoice": "10002",
                "StockCode": "A",
                "Description": "Apple",
                "Quantity": 1,
                "InvoiceDate": "2010-01-02 09:00",
                "Price": 1.5,
                "Customer ID": 1,
                "Country": "United Kingdom",
            },
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
        ]
    )


def test_full_pipeline_creates_expected_outputs(tmp_path: Path) -> None:
    """Run the full offline pipeline on fixture data and verify each stage's outputs."""
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "processed"
    fixture_raw_invoices().to_csv(input_path, index=False)

    result = run_preprocess_pipeline(input_path, output_dir, dataset_type="online-retail-ii")

    expected_files = {"products.csv", "transactions-clean.csv", "item-id-map.csv", "sequence-db.txt"}
    assert {path.name for path in result.values()} == expected_files
    assert len(pd.read_csv(result["products"])) == 3
    assert len(pd.read_csv(result["transactions_clean"])) == 4


def test_full_pipeline_sequence_db_format(tmp_path: Path) -> None:
    """Verify the sequence DB matches expected SPMF format for fixture data."""
    clean = clean_transactions(normalize_online_retail_frame(fixture_raw_invoices()))
    mapped, _, _ = build_product_map(clean)
    sequences = build_customer_sequences(mapped)
    seq_path = tmp_path / "sequence-db.txt"
    write_sequence_db(sequences, seq_path)

    lines = seq_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    # Customer 1: {A,B} {A} then Customer 2: {C}
    assert lines == ["1 2 -1 1 -1 -2", "3 -1 -2"]


def _java_available() -> bool:
    return shutil.which("java") is not None and shutil.which("javac") is not None


@pytest.mark.skipif(not _java_available(), reason="Java not available on PATH")
def test_end_to_end_pipeline_with_cmspade(tmp_path: Path) -> None:
    """Full E2E: fixture CSV → preprocess → CM-SPADE → parse → score → SQLite import."""
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "processed"
    fixture_raw_invoices().to_csv(input_path, index=False)

    preprocess_result = run_preprocess_pipeline(
        input_path, output_dir, dataset_type="online-retail-ii"
    )
    seq_db_path = preprocess_result["sequence_db"]

    patterns_raw_path = tmp_path / "patterns-raw.txt"
    scored_path = tmp_path / "patterns-scored.csv"
    database_path = tmp_path / "app.sqlite"

    mining_result = run_cmspade(seq_db_path, patterns_raw_path, minsup=0.5)

    assert patterns_raw_path.exists()
    assert mining_result.sequence_count == 2
    assert mining_result.pattern_count > 0

    products = pd.read_csv(preprocess_result["products"])
    parsed = list(iter_parse_patterns_file(patterns_raw_path))
    assert len(parsed) > 0

    scored = score_patterns(
        parsed,
        products,
        sequence_count=mining_result.sequence_count,
        min_support=int(0.5 * mining_result.sequence_count),
    )
    write_scored_patterns_csv(scored, scored_path)
    assert scored_path.exists()
    assert len(scored) > 0

    run_id = write_import_database(
        database_path=database_path,
        products=products,
        patterns=scored,
        dataset_name="fixture-e2e",
        minsup=0.5,
        sequence_count=mining_result.sequence_count,
    )
    assert run_id is not None


@pytest.mark.skipif(not _java_available(), reason="Java not available on PATH")
def test_end_to_end_web_app_serves_mined_recommendations(tmp_path: Path) -> None:
    """Full E2E: pipeline → SQLite → web app serves pattern-based recommendations."""
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "processed"
    fixture_raw_invoices().to_csv(input_path, index=False)

    preprocess_result = run_preprocess_pipeline(
        input_path, output_dir, dataset_type="online-retail-ii"
    )
    seq_db_path = preprocess_result["sequence_db"]
    patterns_raw_path = tmp_path / "patterns-raw.txt"
    database_path = tmp_path / "app.sqlite"

    mining_result = run_cmspade(seq_db_path, patterns_raw_path, minsup=0.5)
    products = pd.read_csv(preprocess_result["products"])
    parsed = list(iter_parse_patterns_file(patterns_raw_path))
    scored = score_patterns(
        parsed,
        products,
        sequence_count=mining_result.sequence_count,
        min_support=int(0.5 * mining_result.sequence_count),
    )
    write_import_database(
        database_path=database_path,
        products=products,
        patterns=scored,
        dataset_name="fixture-e2e",
        minsup=0.5,
        sequence_count=mining_result.sequence_count,
    )

    client = TestClient(create_app(database_path))

    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "Phân tích bán lẻ" in dashboard.text

    products_page = client.get("/products")
    assert products_page.status_code == 200
    assert "Danh mục sản phẩm" in products_page.text

    patterns_page = client.get("/patterns")
    assert patterns_page.status_code == 200
    assert "Mẫu tuần tự" in patterns_page.text

    # Cart page: pick a product that exists in the fixture (Apple, item_id=1)
    cart_response = client.get("/cart", params={"items": "1"})
    assert cart_response.status_code == 200
    assert "Giỏ hàng" in cart_response.text

    # Recommendation with known cart items
    recommendation = client.post("/recommend", json={"cart_item_ids": [1], "top_k": 3})
    assert recommendation.status_code == 200
    # Must return product recommendations (pattern-based or fallback)
    assert "recommendation-card" in recommendation.text
    assert any(
        name in recommendation.text for name in ["Coffee", "Bread"]
    ), "recommendation should suggest products not in cart"


def test_generated_files_not_leaked_to_git() -> None:
    """Guardrail: no generated or accidental files in git status.

    Any file appearing in git status must belong to a known safe prefix
    (source, tests, docs, config). Everything else — data outputs,
    build artifacts, databases — must be covered by .gitignore.
    """
    import subprocess

    root = Path(__file__).resolve().parents[1]
    safe_prefixes = [
        ".gitignore",
        ".claude/",
        "app/",
        "mining/",
        "tests/",
        "cmspade/src/",
        "README.md",
        "pyproject.toml",
    ]

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )

    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        status_path = line[3:].strip()
        if any(status_path.startswith(prefix) for prefix in safe_prefixes):
            continue
        raise AssertionError(
            f"Unexpected file in git status: {status_path}\n"
            "Either add a safe prefix for this file pattern or ensure "
            ".gitignore covers it."
        )


@pytest.mark.skipif(not _java_available(), reason="Java not available on PATH")
def test_pipeline_error_surfaces_cleanly(tmp_path: Path) -> None:
    """E2E: bad data produces clean error, not a cryptic traceback."""
    bad_input = tmp_path / "bad.csv"
    bad_input.write_text("garbage\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required"):
        run_preprocess_pipeline(bad_input, tmp_path / "out", dataset_type="online-retail-ii")
