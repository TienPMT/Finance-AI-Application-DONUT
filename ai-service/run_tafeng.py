"""
Chạy WCM-SPADE trên Ta-Feng (reproduce paper ICT-1).

Build sequence DB:
  - mỗi CUSTOMER_ID = 1 sequence
  - baskets gom theo TRANSACTION_DT, sort theo ngày
  - item = PRODUCT_SUBCLASS (mặc định) hoặc PRODUCT_ID
  - giá item = trung bình SALES_PRICE (cho value-weight)

Đánh giá next-basket: last-basket holdout (Recall@K / Precision@K) như paper.

Ví dụ:
  python run_tafeng.py --min-sup 0.01 --alpha 0.5 --top-k 5 --max-items 500
  python run_tafeng.py --item-col PRODUCT_ID --max-customers 5000
"""

import argparse
import csv
import os
import time
from collections import defaultdict
from datetime import datetime

from wcm_spade import WCMSpade, NextBasketPredictor

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ta_feng_all_months_merged.csv")


def to_ord(s: str):

    try:
        mm, dd, yy = s.split("/")
        return datetime(int(yy), int(mm), int(dd)).toordinal()
    except Exception:
        return None


def load_tafeng(path, item_col, max_customers=None, max_rows=None, return_cids=False):

    cust = defaultdict(lambda: defaultdict(set))
    price_sum = defaultdict(float)
    price_cnt = defaultdict(int)
    rows = 0
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name: i for i, name in enumerate(header)}
        i_cid, i_item = idx["CUSTOMER_ID"], idx[item_col]
        i_dt, i_price = idx["TRANSACTION_DT"], idx["SALES_PRICE"]
        for r in reader:
            rows += 1
            if max_rows and rows > max_rows:
                break
            cid, item = r[i_cid], r[i_item]
            if not cid or not item:
                continue
            if max_customers and cid not in cust and len(cust) >= max_customers:
                continue
            d = to_ord(r[i_dt])
            if d is None:
                continue
            cust[cid][d].add(item)
            try:
                price = float(r[i_price])
            except ValueError:
                price = 0.0
            price_sum[item] += price
            price_cnt[item] += 1
    print(f"  đọc {rows} dòng, {len(cust)} khách hàng, {len(price_sum)} item ({item_col})")

    sequences, cids = [], []
    for cid, days in cust.items():
        seq = [tuple(sorted(days[day])) for day in sorted(days.keys())]
        sequences.append(seq)
        cids.append(cid)

    item_prices = {it: price_sum[it] / price_cnt[it] for it in price_sum}
    if return_cids:
        return sequences, item_prices, cids
    return sequences, item_prices


def evaluate_next_basket(sequences, predictor, top_k, item_prices=None):
    item_prices = item_prices or {}
    pr = lambda it: float(item_prices.get(it, 0.0))
    recalls, precisions, vrecalls, hits = [], [], [], 0
    tot_truth_val = tot_hit_val = 0.0
    n = 0
    for seq in sequences:
        if len(seq) < 2:
            continue
        history, truth = seq[:-1], set(seq[-1])
        if not truth:
            continue
        preds = [item for item, _ in predictor.predict(history, top_k)]
        if not preds:
            continue
        pred_set = set(preds)
        inter = len(pred_set & truth)
        recalls.append(inter / len(truth))
        precisions.append(inter / len(preds))

        tv = sum(pr(t) for t in truth)
        hv = sum(pr(p) for p in pred_set & truth)
        vrecalls.append((hv / tv) if tv > 0 else 0.0)
        tot_truth_val += tv
        tot_hit_val += hv
        if inter > 0:
            hits += 1
        n += 1
    if n == 0:
        return {"recall": 0, "precision": 0, "value_recall": 0, "hit_ratio": 0,
                "value_coverage": 0, "n": 0}
    return {
        "recall": sum(recalls) / n,
        "precision": sum(precisions) / n,
        "value_recall": sum(vrecalls) / n,
        "value_coverage": (tot_hit_val / tot_truth_val) if tot_truth_val > 0 else 0.0,
        "hit_ratio": hits / n,
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV_PATH)
    ap.add_argument("--item-col", default="PRODUCT_SUBCLASS",
                    choices=["PRODUCT_SUBCLASS", "PRODUCT_ID"])
    ap.add_argument("--min-sup", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=4)
    ap.add_argument("--max-items", type=int, default=500)
    ap.add_argument("--max-customers", type=int, default=None)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--max-patterns", type=int, default=30000)
    args = ap.parse_args()

    print("=" * 60)
    print(f"WCM-SPADE on Ta-Feng | item={args.item_col} minSup={args.min_sup} "
          f"alpha={args.alpha} K={args.top_k}")
    print("=" * 60)

    t0 = time.time()
    sequences, item_prices = load_tafeng(args.csv, args.item_col, args.max_customers, args.max_rows)
    print(f"  load: {time.time()-t0:.1f}s | {len(sequences)} sequences")

    t1 = time.time()
    engine = WCMSpade(
        min_sup=args.min_sup,
        alpha=args.alpha,
        max_pattern_length=args.max_len,
        max_items=args.max_items,
        max_patterns=args.max_patterns,
    )
    result = engine.mine(sequences, item_prices)
    mine_t = time.time() - t1
    seq_pat = [p for p in result.patterns
               if len(p.itemsets) >= 2 or sum(len(it) for it in p.itemsets) >= 2]
    print(f"\n  MINE: {mine_t:.2f}s")
    print(f"  N={result.n_sequences} minSupAbs={result.min_sup_abs} "
          f"effectiveMinSup={result.effective_min_sup} MeanW={result.mean_w:.4f}")
    print(f"  patterns tổng={len(result.patterns)} | sequential(>=2)={len(seq_pat)}")

    print("\n  Top 10 weighted patterns:")
    for p in seq_pat[:10]:
        print(f"    {p.to_str():40s} rawSup={p.raw_sup:5d} WSUP={p.wsup:.2f}")

    t2 = time.time()
    predictor = NextBasketPredictor(result.patterns, result.item_weights)
    metrics = evaluate_next_basket(sequences, predictor, args.top_k, item_prices)
    print(f"\n  EVAL next-basket (last-basket holdout): {time.time()-t2:.1f}s")
    print(f"    Recall@{args.top_k}        = {metrics['recall']:.4f}")
    print(f"    Precision@{args.top_k}     = {metrics['precision']:.4f}")
    print(f"    Value-Recall@{args.top_k}  = {metrics['value_recall']:.4f}  (thiên giá trị)")
    print(f"    ValueCoverage@{args.top_k} = {metrics['value_coverage']:.4f}  (tổng VND bắt được/tổng VND truth)")
    print(f"    HitRatio@{args.top_k}      = {metrics['hit_ratio']:.4f}")
    print(f"    n(eval khách) = {metrics['n']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
