"""
Export toàn bộ kết quả KPDL ra CSV + JSON để dựng slide báo cáo.

Chạy:  python export_results.py --max-customers 20000 --alpha 0.3 --min-sup 0.01
Ghi vào thư mục  results/  :
  - metrics_k5.csv, metrics_k10.csv   (bảng thô + value mỗi thuật toán)
  - value_summary.csv                 (recall vs value-recall vs giá TB/gợi ý)
  - significance.csv                  (Wilcoxon WCM-SPADE vs baseline)
  - data_quality.csv                  (pipeline làm sạch + thống kê)
  - classification.csv                (Decision Tree / Naive Bayes / KNN)
  - all_results.json                  (gộp tất cả)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time

import tafeng_demo
import data_quality
import classification

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _w(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-customers", type=int, default=20000)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--min-sup", type=float, default=0.01)
    ap.add_argument("--item-col", default="PRODUCT_SUBCLASS")
    ap.add_argument("--classify-customers", type=int, default=8000)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    print(f"[1/4] run_experiment maxCust={args.max_customers} alpha={args.alpha} minSup={args.min_sup} ...")
    exp = tafeng_demo.run_experiment(
        min_sup=args.min_sup, alpha=args.alpha,
        max_customers=args.max_customers, item_col=args.item_col)
    if "error" in exp:
        print("ERROR:", exp["error"]); return

    metrics = exp["metrics"]
    for k in exp["params"]["ks"]:
        rows = [[a] + [exp["results"][a][str(k)][m] for m in metrics] for a in exp["algos"]]
        _w(os.path.join(OUT_DIR, f"metrics_k{k}.csv"), ["algo"] + metrics, rows)

    vs = exp["valueSummary"]
    _w(os.path.join(OUT_DIR, "value_summary.csv"),
       ["algo", "recall", "valueRecall", "valueCoverage", "recAvgPrice", "hitValue"],
       [[a, vs[a]["recall"], vs[a]["valueRecall"], vs[a]["valueCoverage"],
         vs[a]["recAvgPrice"], vs[a]["hitValue"]] for a in exp["algos"]])

    sig = exp.get("significance", {})
    srows = []
    for a, row in sig.get("vs", {}).items():
        for m, c in row.items():
            srows.append([a, m, c["meanDiff"], c["p"], c["sig"], c["winsRef"]])
    _w(os.path.join(OUT_DIR, "significance.csv"),
       ["vsBaseline", "metric", "meanDiff", "p", "significant", "spadeWins"], srows)

    print(f"[2/4] data_quality full CSV ...")
    dq = data_quality.analyze(item_col=args.item_col)
    r = dq["rows"]; c = dq["customers"]; it = dq["items"]; b = dq["baskets"]
    _w(os.path.join(OUT_DIR, "data_quality.csv"), ["metric", "value"], [
        ["totalRead", r["totalRead"]], ["dropMissingId", r["dropMissingId"]],
        ["dropBadDate", r["dropBadDate"]], ["dropBadPrice", r["dropBadPrice"]],
        ["dropBadAmount", r["dropBadAmount"]], ["kept", r["kept"]],
        ["keptPct", r["keptPct"]], ["duplicateLines", r["duplicateLines"]],
        ["duplicatePct", r["duplicatePct"]], ["nCustomers", c["nCustomers"]],
        ["nWithGe2Baskets", c["nWithGe2Baskets"]], ["ge2Pct", c["ge2Pct"]],
        ["spendMedian", c["spendMedian"]], ["spendP90", c["spendP90"]],
        ["spendMax", c["spendMax"]], ["nItems", it["nItems"]],
        ["priceMin", it["priceMin"]], ["priceMedian", it["priceMedian"]],
        ["priceP90", it["priceP90"]], ["priceMax", it["priceMax"]],
        ["avgItemsPerBasket", b["avgItemsPerBasket"]],
        ["avgBasketsPerCustomer", b["avgBasketsPerCustomer"]],
    ])

    print(f"[3/4] classification {args.classify_customers} khách ...")
    cl = classification.run(max_customers=args.classify_customers, item_col=args.item_col)
    if "error" not in cl:
        _w(os.path.join(OUT_DIR, "classification.csv"),
           ["model", "accuracy", "precision", "recall", "f1", "confusion"],
           [[name, m["accuracy"], m["precision"], m["recall"], m["f1"],
             json.dumps(m["confusion"])] for name, m in cl["models"].items()])

    print(f"[4/4] all_results.json ...")
    allj = {"experiment": exp, "dataQuality": dq, "classification": cl,
            "generatedSec": round(time.time() - t0, 1)}
    with open(os.path.join(OUT_DIR, "all_results.json"), "w", encoding="utf-8") as f:
        json.dump(allj, f, ensure_ascii=False, indent=2, default=str)
    print(f"  wrote all_results.json")
    print(f"DONE in {time.time()-t0:.1f}s → {OUT_DIR}")


if __name__ == "__main__":
    main()
