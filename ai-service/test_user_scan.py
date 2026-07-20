"""
Test WCM-SPADE TRỰC TIẾP trên dữ liệu hóa đơn người dùng quét (Donut).

Chứng minh: table menu của hóa đơn ( nm / price / cnt ) khớp format Ta-Feng
( PRODUCT / SALES_PRICE / AMOUNT ). Một hóa đơn = 1 basket. Nhiều hóa đơn của
cùng 1 user theo thời gian = 1 sequence → mining giống hệt Ta-Feng.

INPUT: file JSON, list các hóa đơn đã quét. 2 dạng chấp nhận:

  A) Output thô từ /donut/scan (có khóa "seller" + "menu"):
     [
       {"user": "u1", "date": "2026-01-05",
        "scan": {"seller": {"seller_name": "BHX",
                            "total_cost_number": 120000,
                            "menu": [{"nm": "Sua", "price": "30000", "cnt": "2"},
                                     {"nm": "Banh", "price": "15000", "cnt": "1"}]}}},
       ...
     ]

  B) Dạng rút gọn (tự gõ tay để test nhanh):
     [
       {"user": "u1", "date": "2026-01-05",
        "items": [{"name": "Sua", "price": 30000}, {"name": "Banh", "price": 15000}]},
       ...
     ]

CHẠY:
  python test_user_scan.py receipts.json --min-sup 0.3 --alpha 0.5 --top-k 3
  python test_user_scan.py --demo    
"""

import argparse
import json
import sys
from collections import defaultdict

from wcm_spade import WCMSpade, NextBasketPredictor


def _to_float(x):
    if isinstance(x, (int, float)):
        return float(x)
    if x is None:
        return 0.0
    digits = "".join(c for c in str(x) if c.isdigit())
    return float(digits) if digits else 0.0


def extract_items(receipt):
    """Trả về list (name, price) từ 1 hóa đơn, hỗ trợ cả dạng A và B."""

    if "items" in receipt:
        return [(it.get("name", "?"), _to_float(it.get("price"))) for it in receipt["items"]]

    scan = receipt.get("scan", {})
    seller = scan.get("seller", scan)
    menu = seller.get("menu", [])
    if isinstance(menu, dict):
        menu = [menu]
    out = []
    for it in menu:
        out.append((it.get("nm", "?"), _to_float(it.get("price"))))
    return out


def build_sequences(receipts):
    """user -> ngày(sorted) -> basket(set tên SP). Trả (sequences, item_prices)."""
    by_user = defaultdict(lambda: defaultdict(set))
    price_sum = defaultdict(float)
    price_cnt = defaultdict(int)
    for r in receipts:
        user = r.get("user", "default")
        date = r.get("date", "")
        for name, price in extract_items(r):
            by_user[user][date].add(name)
            price_sum[name] += price
            price_cnt[name] += 1

    sequences = []
    for user, days in by_user.items():
        seq = [tuple(sorted(days[d])) for d in sorted(days.keys())]
        sequences.append(seq)
    item_prices = {n: price_sum[n] / price_cnt[n] for n in price_sum if price_cnt[n]}
    return sequences, item_prices


DEMO = [
    {"user": "u1", "date": "2026-01-05", "items": [{"name": "Sua", "price": 30000}, {"name": "Banh mi", "price": 15000}]},
    {"user": "u1", "date": "2026-01-12", "items": [{"name": "Sua", "price": 30000}, {"name": "Ca phe", "price": 45000}]},
    {"user": "u1", "date": "2026-01-20", "items": [{"name": "Banh mi", "price": 15000}, {"name": "Ca phe", "price": 45000}]},
    {"user": "u2", "date": "2026-01-03", "items": [{"name": "Sua", "price": 32000}, {"name": "Banh mi", "price": 16000}]},
    {"user": "u2", "date": "2026-01-15", "items": [{"name": "Sua", "price": 32000}, {"name": "Ca phe", "price": 50000}]},
    {"user": "u3", "date": "2026-01-08", "items": [{"name": "Sua", "price": 31000}, {"name": "Banh mi", "price": 15500}]},
    {"user": "u3", "date": "2026-01-19", "items": [{"name": "Ca phe", "price": 48000}, {"name": "Banh mi", "price": 15500}]},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="JSON hóa đơn đã quét")
    ap.add_argument("--demo", action="store_true", help="dùng data mẫu")
    ap.add_argument("--min-sup", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=4)
    args = ap.parse_args()

    if args.demo:
        receipts = DEMO
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            receipts = json.load(f)
    else:
        print("Cần file JSON hoặc --demo. Xem docstring để biết format.")
        sys.exit(1)

    sequences, item_prices = build_sequences(receipts)
    print(f"{len(receipts)} hóa đơn → {len(sequences)} sequence (user), {len(item_prices)} item")
    for i, s in enumerate(sequences):
        print(f"  seq[{i}] = {s}")
    print(f"  giá TB: { {k: round(v) for k, v in item_prices.items()} }")

    engine = WCMSpade(min_sup=args.min_sup, alpha=args.alpha, max_pattern_length=args.max_len)
    res = engine.mine(sequences, item_prices)
    print(f"\nN={res.n_sequences} minSupAbs={res.min_sup_abs} "
          f"effectiveMinSup={res.effective_min_sup} MeanW={res.mean_w:.4f}")
    print("item weights:", {k: round(v, 3) for k, v in res.item_weights.items()})

    seq_pat = [p for p in res.patterns
               if len(p.itemsets) >= 2 or sum(len(it) for it in p.itemsets) >= 2]
    print(f"\nWeighted sequential patterns ({len(seq_pat)}):")
    for p in seq_pat[:15]:
        print(f"  {p.to_str():35s} rawSup={p.raw_sup} WSUP={p.wsup:.3f}")

    pred = NextBasketPredictor(res.patterns)
    print("\nNext-basket predict (ví dụ):")
    for hist in ([("Sua",)], [("Banh mi",)], [("Ca phe",)]):
        print(f"  sau {hist} → {[(i, round(w,2)) for i,w in pred.predict(hist, args.top_k)]}")


if __name__ == "__main__":
    main()
