"""
Phân lớp khách hàng giá trị cao/thấp trên Ta-Feng (rubric Chương 3 — Phân lớp dữ liệu).

Bài toán: từ hành vi mua (RFM + đặc trưng giỏ) dự đoán khách có thuộc nhóm
CHI TIÊU CAO hay không. Nhãn = top 1/3 theo tổng chi (monetary tercile).

So sánh 3 bộ phân lớp trong đề cương: Decision Tree, Naïve Bayes, K-Nearest Neighbors.
Báo cáo Accuracy / Precision / Recall / F1 (macro) + confusion matrix, dùng
train/test split 70/30 stratify. KHÔNG dùng pandas.

Đặc trưng (behavioral — tránh leakage trực tiếp từ monetary):
  R = recency (ngày kể từ lần mua cuối tới mốc cuối dataset)
  F = frequency (số lần đi siêu thị / số ngày có giao dịch)
  nItems = số item phân biệt đã mua
  avgBasket = TB số item/lần
  avgPrice = giá TB item đã mua
  qty = tổng số lượng món

Chạy:  python classification.py 8000
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

from run_tafeng import CSV_PATH


def _parse_dt(s: str):
    try:
        mm, dd, yy = s.split("/")
        return datetime(int(yy), int(mm), int(dd)).toordinal()
    except Exception:
        return None


def build_features(path: str = CSV_PATH, max_customers: int | None = None,
                   item_col: str = "PRODUCT_SUBCLASS"):
    cust_days = defaultdict(set)
    cust_items = defaultdict(set)
    cust_qty = defaultdict(float)
    cust_spend = defaultdict(float)
    cust_lines = defaultdict(int)
    cust_last = defaultdict(int)
    max_day = 0

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {n.strip().strip('"'): i for i, n in enumerate(header)}
        i_cid, i_item = idx["CUSTOMER_ID"], idx[item_col]
        i_dt, i_price, i_amt = idx["TRANSACTION_DT"], idx["SALES_PRICE"], idx["AMOUNT"]
        for r in reader:
            cid, item = r[i_cid].strip(), r[i_item].strip()
            if not cid or not item:
                continue
            if max_customers and cid not in cust_days and len(cust_days) >= max_customers:
                continue
            d = _parse_dt(r[i_dt])
            if d is None:
                continue
            try:
                price = float(r[i_price]); amt = float(r[i_amt])
            except ValueError:
                continue
            if price <= 0 or amt <= 0:
                continue
            cust_days[cid].add(d)
            cust_items[cid].add(item)
            cust_qty[cid] += amt
            cust_spend[cid] += price * amt
            cust_lines[cid] += 1
            if d > cust_last[cid]:
                cust_last[cid] = d
            if d > max_day:
                max_day = d

    cids = list(cust_days.keys())
    X, spend = [], []
    for c in cids:
        nDays = len(cust_days[c])
        X.append([
            max_day - cust_last[c],
            nDays,
            len(cust_items[c]),
            cust_lines[c] / nDays,
            cust_spend[c] / cust_qty[c] if cust_qty[c] else 0.0,
            cust_qty[c],
        ])
        spend.append(cust_spend[c])
    X = np.array(X, dtype=np.float64)
    spend = np.array(spend)

    thr = np.quantile(spend, 2.0 / 3.0)
    y = (spend >= thr).astype(int)
    return X, y, cids


def run(max_customers: int | None = 8000, item_col: str = "PRODUCT_SUBCLASS",
        test_size: float = 0.3, seed: int = 42) -> dict:
    X, y, cids = build_features(CSV_PATH, max_customers, item_col)
    if len(X) < 50:
        return {"error": f"Quá ít khách ({len(X)})."}

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    models = {
        "Decision Tree": (DecisionTreeClassifier(max_depth=6, random_state=seed), False),
        "Naive Bayes": (GaussianNB(), True),
        "KNN (k=15)": (KNeighborsClassifier(n_neighbors=15), True),
    }
    out = {}
    for name, (clf, scaled) in models.items():
        a, b = (Xtr_s, Xte_s) if scaled else (Xtr, Xte)
        clf.fit(a, ytr)
        pred = clf.predict(b)
        cm = confusion_matrix(yte, pred).tolist()
        out[name] = {
            "accuracy": round(accuracy_score(yte, pred), 4),
            "precision": round(precision_score(yte, pred, average="macro", zero_division=0), 4),
            "recall": round(recall_score(yte, pred, average="macro", zero_division=0), 4),
            "f1": round(f1_score(yte, pred, average="macro", zero_division=0), 4),
            "confusion": cm,
        }
    return {
        "task": "Phân lớp khách CHI TIÊU CAO (top 1/3) vs còn lại",
        "features": ["recency", "frequency", "nItems", "avgBasket", "avgPrice", "qty"],
        "nSamples": int(len(X)),
        "nTrain": int(len(Xtr)),
        "nTest": int(len(Xte)),
        "posRate": round(float(y.mean()), 4),
        "models": out,
    }


if __name__ == "__main__":
    mc = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    rep = run(max_customers=mc)
    if "error" in rep:
        print(rep); raise SystemExit
    print("=" * 60)
    print(rep["task"])
    print(f"  features={rep['features']}")
    print(f"  N={rep['nSamples']} train={rep['nTrain']} test={rep['nTest']} "
          f"posRate={rep['posRate']}")
    print("=" * 60)
    print(f"  {'model':16s} {'acc':>7s} {'prec':>7s} {'recall':>7s} {'f1':>7s}")
    for name, m in rep["models"].items():
        print(f"  {name:16s} {m['accuracy']:7.3f} {m['precision']:7.3f} "
              f"{m['recall']:7.3f} {m['f1']:7.3f}  CM={m['confusion']}")
    print("=" * 60)
