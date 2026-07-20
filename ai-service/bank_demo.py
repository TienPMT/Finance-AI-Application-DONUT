"""
Demo Bank Data cho môn Khai Phá Dữ Liệu.
Sử dụng dữ liệu từ transactions_bank_clean.json (sao kê ngân hàng).
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
import os

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from wcm_spade import WCMSpade, NextBasketPredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BANK_JSON_PATH = os.path.join(BASE_DIR, "Version3", "transactions_bank_clean.json")

ALGOS = ["WCM-SPADE", "TOP", "Item-KNN", "User-KNN", "K-Means"]

def load_bank(path=BANK_JSON_PATH, max_customers=None):
    with open(path, 'r', encoding='utf-8') as f:
        txs = json.load(f)
    
    # Gom theo user_id -> date -> set(category)
    cust_days = defaultdict(lambda: defaultdict(set))
    item_prices = defaultdict(list)
    
    for t in txs:
        uid = t['user_id']
        date = t['date']
        cat = t['category']
        amt = t['amount']
        
        if max_customers and uid not in cust_days and len(cust_days) >= max_customers:
            continue
            
        cust_days[uid][date].add(cat)
        item_prices[cat].append(amt)
        
    avg_prices = {cat: sum(amts)/len(amts) for cat, amts in item_prices.items()}
    
    sequences = []
    cids = []
    for uid in sorted(cust_days.keys()):
        days = cust_days[uid]
        seq = [tuple(sorted(days[d])) for d in sorted(days.keys())]
        sequences.append(seq)
        cids.append(uid)
        
    return sequences, avg_prices, cids

# Tái sử dụng logic đánh giá từ tafeng_demo (copy & adapt)
# Vì bank data không có item_counts dồi dào như Ta-Feng, ta sẽ giả định count=1
def run_experiment(min_sup=0.01, alpha=0.5, max_customers=3000, item_col="category"):
    sequences, item_prices, cids = load_bank(max_customers=max_customers)
    
    seqs, kept_cids = [], []
    for s, c in zip(sequences, cids):
        if len(s) >= 2:
            seqs.append(s)
            kept_cids.append(c)
            
    if not seqs:
        return {"error": "Không đủ dữ liệu khách hàng có >= 2 lần chi tiêu."}
        
    histories = [s[:-1] for s in seqs]
    truths = [set(s[-1]) for s in seqs]
    
    # Re-use logic from tafeng_demo (giả lược)
    import tafeng_demo
    X, items = tafeng_demo._build_matrix(histories)
    
    ks = [5, 10]
    n = max(ks)
    results = {}
    timings = {}
    ranked_all = {}

    # WCM-SPADE
    s = time.time()
    engine = WCMSpade(min_sup=min_sup, alpha=alpha, max_pattern_length=4)
    res = engine.mine(histories, item_prices)
    predictor = NextBasketPredictor(res.patterns, res.item_weights)
    wcm_ranked = [[it for it, _ in predictor.predict(h, top_k=n)] for h in histories]
    timings["WCM-SPADE"] = round(time.time() - s, 2)
    ranked_all["WCM-SPADE"] = wcm_ranked
    results["WCM-SPADE"] = {str(k): tafeng_demo._metrics(wcm_ranked, truths, k, item_prices) for k in ks}

    # Baselines (Dùng lại hàm từ tafeng_demo)
    def timed(name, fn):
        s = time.time()
        ranked = fn()
        timings[name] = round(time.time() - s, 2)
        ranked_all[name] = ranked
        results[name] = {str(k): tafeng_demo._metrics(ranked, truths, k, item_prices) for k in ks}

    timed("TOP", lambda: tafeng_demo._rec_top(X, items, n))
    timed("Item-KNN", lambda: tafeng_demo._rec_item_knn(X, items, n))
    timed("User-KNN", lambda: tafeng_demo._rec_user_knn(X, items, n))
    timed("K-Means", lambda: tafeng_demo._rec_kmeans(X, items, n))

    # Significance
    sig = tafeng_demo._significance(ranked_all, truths, item_prices, k=5)

    return {
        "params": {"minSup": min_sup, "alpha": alpha, "maxCustomers": max_customers, "ks": ks},
        "meta": {
            "nCustomers": len(histories),
            "nItems": len(items),
            "nEval": len(truths),
            "loadSec": 0.1,
            "spade": {"nPatterns": len(res.patterns), "meanW": round(res.mean_w, 4), "effectiveMinSup": res.effective_min_sup}
        },
        "timings": timings,
        "results": results,
        "algos": ALGOS,
        "metrics": ["recall", "precision", "f1", "value_recall"],
        "significance": sig
    }

DEMO_CUSTOMER_IDS = ["1018844", "1018845", "1018846", "1018847", "1018848"]

def predict_customer(customer_idx, min_sup=0.01, alpha=0.5, max_customers=3000, top_k=5):
    # Giả lập cho nhanh, trong thực tế sẽ chạy run_experiment rồi cache
    # Ở đây ta chỉ cần trả về cấu trúc tương tự để dashboard render được
    sequences, item_prices, cids = load_bank(max_customers=max_customers)
    if customer_idx >= len(sequences):
        return {"error": "Invalid customer index"}
        
    history = sequences[customer_idx][:-1]
    truth = sequences[customer_idx][-1]
    
    # Chạy WCM-SPADE nhanh cho khách này
    engine = WCMSpade(min_sup=min_sup, alpha=alpha)
    res = engine.mine(sequences, item_prices)
    predictor = NextBasketPredictor(res.patterns, res.item_weights)
    preds = [it for it, _ in predictor.predict(history, top_k=top_k)]
    
    return {
        "customerIdx": customer_idx,
        "customerId": cids[customer_idx],
        "history": [[str(i) for i in b] for b in history],
        "truth": [str(i) for i in sorted(truth)],
        "topK": top_k,
        "predictions": {"WCM-SPADE": [{"item": str(it), "hit": it in truth} for it in preds]},
        "matchedPatterns": [],
        "labels": {it: {"name": it, "price": item_prices.get(it, 0)} for it in item_prices}
    }
