from pathlib import Path

import pandas as pd


ONLINE_RETAIL_COLUMNS = {
    "Invoice": "invoice_id",
    "StockCode": "product_source_id",
    "Description": "product_name",
    "Quantity": "quantity",
    "InvoiceDate": "invoice_datetime",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}

ONLINE_RETAIL_SHEETS = ("Year 2009-2010", "Year 2010-2011")


def normalize_online_retail_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(set(ONLINE_RETAIL_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required Online Retail II columns: {missing_columns}")

    normalized = frame.rename(columns=ONLINE_RETAIL_COLUMNS)
    normalized = normalized[list(ONLINE_RETAIL_COLUMNS.values())].copy()
    normalized["invoice_id"] = normalized["invoice_id"].astype("string").str.strip()
    normalized["product_source_id"] = normalized["product_source_id"].astype("string").str.strip()
    normalized["product_name"] = normalized["product_name"].astype("string").str.strip()
    normalized["country"] = normalized["country"].astype("string").str.strip()
    normalized["customer_id"] = normalized["customer_id"].map(format_customer_id)
    normalized["quantity"] = pd.to_numeric(normalized["quantity"], errors="coerce")
    normalized["unit_price"] = pd.to_numeric(normalized["unit_price"], errors="coerce")
    normalized["invoice_datetime"] = pd.to_datetime(
        normalized["invoice_datetime"],
        errors="coerce",
    )
    return normalized


def format_customer_id(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text or None


def load_online_retail_ii(path: Path, row_limit: int | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if path.suffix.lower() in {".csv", ".txt"}:
        frame = pd.read_csv(path, nrows=row_limit)
        return normalize_online_retail_frame(frame)

    frames: list[pd.DataFrame] = []
    remaining_rows = row_limit
    for sheet_name in ONLINE_RETAIL_SHEETS:
        nrows = remaining_rows if remaining_rows is not None else None
        sheet = pd.read_excel(path, sheet_name=sheet_name, nrows=nrows)
        frames.append(sheet)
        if remaining_rows is not None:
            remaining_rows -= len(sheet)
            if remaining_rows <= 0:
                break

    combined = pd.concat(frames, ignore_index=True)
    return normalize_online_retail_frame(combined)


def load_dataset(path: Path, dataset_type: str, row_limit: int | None = None) -> pd.DataFrame:
    if dataset_type != "online-retail-ii":
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    return load_online_retail_ii(path, row_limit=row_limit)
