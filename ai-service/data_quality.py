"""
Data Quality & EDA cho Ta-Feng (rubric "Làm sạch dữ liệu / Phân tích dữ liệu").

KHÔNG dùng pandas (venv không có) — đọc CSV thuần + thống kê tay.
Báo cáo TRUNG THỰC số dòng bị loại ở từng bước làm sạch + thống kê mô tả.

Bước làm sạch (data cleaning pipeline):
  1. drop dòng thiếu CUSTOMER_ID / PRODUCT_* (null/empty)
  2. drop dòng TRANSACTION_DT sai định dạng (parse m/d/Y thất bại)
  3. drop dòng SALES_PRICE <= 0 hoặc không parse được (giá lỗi)
  4. drop dòng AMOUNT (số lượng) <= 0
  5. dedup: cùng (customer, ngày, item) gộp 1 lần mua trong basket
  6. (downstream) chỉ giữ khách >= 2 basket cho holdout

Chạy:  python data_quality.py            (full CSV)
       python data_quality.py 5000       (giới hạn 5000 khách)
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime

from run_tafeng import CSV_PATH


def _parse_dt(s: str):
    try:
        mm, dd, yy = s.split("/")
        return datetime(int(yy), int(mm), int(dd))
    except Exception:
        return None


def analyze(path: str = CSV_PATH, max_customers: int | None = None,
            item_col: str = "PRODUCT_SUBCLASS") -> dict:
    total = 0
    drop_id = drop_date = drop_price = drop_amount = 0
    skipped_demo = 0
    kept = 0

    cust_baskets = defaultdict(set)
    cust_spend = defaultdict(float)
    cust_days = defaultdict(set)
    item_price_sum = defaultdict(float)
    item_price_cnt = defaultdict(int)
    seen_line = set()
    dup_lines = 0
    prices = []
    age_dist = defaultdict(int)

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name.strip().strip('"'): i for i, name in enumerate(header)}
        i_cid, i_item = idx["CUSTOMER_ID"], idx[item_col]
        i_dt, i_price = idx["TRANSACTION_DT"], idx["SALES_PRICE"]
        i_amt = idx["AMOUNT"]
        i_age = idx.get("AGE_GROUP", -1)

        for r in reader:
            total += 1
            cid = r[i_cid].strip()
            item = r[i_item].strip()
            if not cid or not item:
                drop_id += 1
                continue
            if max_customers and cid not in cust_baskets and len(cust_baskets) >= max_customers:
                skipped_demo += 1
                continue
            dt = _parse_dt(r[i_dt])
            if dt is None:
                drop_date += 1
                continue
            try:
                price = float(r[i_price])
            except ValueError:
                price = -1.0
            if price <= 0:
                drop_price += 1
                continue
            try:
                amt = float(r[i_amt])
            except ValueError:
                amt = 0.0
            if amt <= 0:
                drop_amount += 1
                continue

            dord = dt.toordinal()
            line_key = (cid, dord, item)
            if line_key in seen_line:
                dup_lines += 1
                continue
            seen_line.add(line_key)
            kept += 1
            cust_baskets[cid].add((dord, item))
            cust_spend[cid] += price * amt
            cust_days[cid].add(dord)
            item_price_sum[item] += price
            item_price_cnt[item] += 1
            prices.append(price)
            if i_age >= 0:
                age_dist[r[i_age].strip() or "NA"] += 1

    n_cust = len(cust_baskets)
    n_items = len(item_price_cnt)
    basket_sizes = []
    for cid, pairs in cust_baskets.items():
        by_day = defaultdict(int)
        for d, _ in pairs:
            by_day[d] += 1
        basket_sizes.extend(by_day.values())

    n_multi = sum(1 for cid in cust_days if len(cust_days[cid]) >= 2)

    def pct(parts):
        return sorted(parts)

    sp = pct(prices)

    def q(arr, p):
        if not arr:
            return 0.0
        i = int(p * (len(arr) - 1))
        return arr[i]

    seq_lens = [len(cust_days[c]) for c in cust_days]
    spends = sorted(cust_spend.values())

    return {
        "rows": {
            "totalRead": total,
            "dropMissingId": drop_id,
            "dropBadDate": drop_date,
            "dropBadPrice": drop_price,
            "dropBadAmount": drop_amount,
            "duplicateLines": dup_lines,
            "duplicatePct": round(100 * dup_lines / total, 2) if total else 0,
            "skippedDemo": skipped_demo,
            "kept": kept,
            "keptPct": round(100 * kept / total, 2) if total else 0,
        },
        "customers": {
            "nCustomers": n_cust,
            "nWithGe2Baskets": n_multi,
            "ge2Pct": round(100 * n_multi / n_cust, 2) if n_cust else 0,
            "spendMedian": round(q(spends, 0.5), 1),
            "spendP90": round(q(spends, 0.9), 1),
            "spendMax": round(spends[-1], 1) if spends else 0,
        },
        "items": {
            "nItems": n_items,
            "priceMin": round(sp[0], 2) if sp else 0,
            "priceMedian": round(q(sp, 0.5), 2),
            "priceP90": round(q(sp, 0.9), 2),
            "priceMax": round(sp[-1], 2) if sp else 0,
        },
        "baskets": {
            "nBaskets": len(basket_sizes),
            "avgItemsPerBasket": round(sum(basket_sizes) / len(basket_sizes), 2) if basket_sizes else 0,
            "avgBasketsPerCustomer": round(sum(seq_lens) / len(seq_lens), 2) if seq_lens else 0,
        },
        "ageGroups": dict(sorted(age_dist.items())),
        "params": {"itemCol": item_col, "maxCustomers": max_customers},
    }


def _print(rep: dict):
    r = rep["rows"]
    print("=" * 60)
    print("TA-FENG DATA QUALITY / CLEANING REPORT")
    print("=" * 60)
    print(f"  Tổng dòng đọc          : {r['totalRead']:,}")
    print(f"  - loại thiếu ID/item   : {r['dropMissingId']:,}")
    print(f"  - loại sai ngày        : {r['dropBadDate']:,}")
    print(f"  - loại giá <= 0/lỗi    : {r['dropBadPrice']:,}")
    print(f"  - loại số lượng <= 0   : {r['dropBadAmount']:,}")
    print(f"  => GIỮ LẠI             : {r['kept']:,} ({r['keptPct']}%)")
    print(f"  Dòng trùng (cùng KH/ngày/item): {r['duplicateLines']:,} ({r['duplicatePct']}%) → dedup vào basket")
    c = rep["customers"]
    print(f"\n  Khách hàng             : {c['nCustomers']:,} "
          f"(>= 2 basket: {c['nWithGe2Baskets']:,} = {c['ge2Pct']}% → dùng holdout)")
    print(f"  Chi tiêu/khách (median/P90/max): {c['spendMedian']:,} / {c['spendP90']:,} / {c['spendMax']:,}")
    it = rep["items"]
    print(f"\n  Item ({rep['params']['itemCol']}): {it['nItems']:,}")
    print(f"  Giá item (min/median/P90/max): {it['priceMin']} / {it['priceMedian']} / {it['priceP90']} / {it['priceMax']}")
    print(f"    -> chênh giá rất lớn (min→max {it['priceMin']}→{it['priceMax']}) "
          f"=> Value-Aware metric có ý nghĩa")
    b = rep["baskets"]
    print(f"\n  Basket: {b['nBaskets']:,} | TB {b['avgItemsPerBasket']} item/basket | "
          f"{b['avgBasketsPerCustomer']} basket/khách")
    print(f"\n  Phân bố nhóm tuổi: {rep['ageGroups']}")
    print("=" * 60)


if __name__ == "__main__":
    mc = int(sys.argv[1]) if len(sys.argv) > 1 else None
    _print(analyze(max_customers=mc))
