"""
Phân lớp Bank Data.
"""

from __future__ import annotations

import json
from collections import defaultdict
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BANK_JSON_PATH = os.path.join(BASE_DIR, "Version3", "transactions_bank_clean.json")

def _to_ord(s):
    try:
        return datetime.fromisoformat(s).toordinal()
    except:
        return 0

def build_features(max_customers=10000):
    with open(BANK_JSON_PATH, 'r', encoding='utf-8') as f:
        txs = json.load(f)
    
    cust_days = defaultdict(set)
    cust_cats = defaultdict(set)
    cust_spend = defaultdict(float)
    cust_last = defaultdict(int)
    max_day = 0
    
    for t in txs:
        uid = t['user_id']
        if max_customers and uid not in cust_days and len(cust_days) >= max_customers:
            continue
        d = _to_ord(t['date'])
        amt = t['amount']
        cat = t['category']
        
        cust_days[uid].add(d)
        cust_cats[uid].add(cat)
        cust_spend[uid] += amt
        if d > cust_last[uid]: cust_last[uid] = d
        if d > max_day: max_day = d

    X, spend = [], []
    for uid in cust_days:
        nDays = len(cust_days[uid])
        X.append([
            max_day - cust_last[uid],
            nDays,
            len(cust_cats[uid]),
            cust_spend[uid] / nDays,
            cust_spend[uid]
        ])
        spend.append(cust_spend[uid])
    
    X = np.array(X)
    spend = np.array(spend)
    thr = np.quantile(spend, 2.0/3.0)
    y = (spend >= thr).astype(int)
    return X, y

def run(max_customers=10000):
    X, y = build_features(max_customers)
    if len(X) < 20:
        return {"error": "Không đủ dữ liệu khách hàng."}
        
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    models = {
        "Decision Tree": (DecisionTreeClassifier(max_depth=6, random_state=42), False),
        "Naive Bayes": (GaussianNB(), True),
        "KNN (k=5)": (KNeighborsClassifier(n_neighbors=5), True),
    }
    out = {}
    for name, (clf, scaled) in models.items():
        a, b = (Xtr_s, Xte_s) if scaled else (Xtr, Xte)
        clf.fit(a, ytr)
        pred = clf.predict(b)
        out[name] = {
            "accuracy": round(accuracy_score(yte, pred), 4),
            "precision": round(precision_score(yte, pred, average="macro", zero_division=0), 4),
            "recall": round(recall_score(yte, pred, average="macro", zero_division=0), 4),
            "f1": round(f1_score(yte, pred, average="macro", zero_division=0), 4),
        }
    return {
        "nSamples": len(X),
        "models": out
    }
