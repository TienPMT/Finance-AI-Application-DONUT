"""
Linear Regression cho Bank Data.
"""

from __future__ import annotations

import json
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BANK_JSON_PATH = os.path.join(BASE_DIR, "Version3", "transactions_bank_clean.json")

def build_regression_data(max_customers=10000):
    with open(BANK_JSON_PATH, 'r', encoding='utf-8') as f:
        txs = json.load(f)
    
    # user -> month -> total_spend
    cust_monthly = defaultdict(lambda: defaultdict(float))
    for t in txs:
        uid = t['user_id']
        date_obj = t['date'] # YYYY-MM-DD
        year_month = date_obj[:7] # YYYY-MM
        cust_monthly[uid][year_month] += t['amount']
    
    # Giả định dữ liệu có vài tháng, ta lấy tháng cuối làm target
    all_months = sorted({m for uid in cust_monthly for m in cust_monthly[uid]})
    if len(all_months) < 2:
        return np.array([]), np.array([]), []
        
    target_month = all_months[-1]
    feature_months = all_months[:-1]
    
    X, y = [], []
    for uid, spends in cust_monthly.items():
        row = [spends.get(m, 0.0) for m in feature_months]
        X.append(row)
        y.append(spends.get(target_month, 0.0))
        
    return np.array(X), np.array(y), feature_months

def run_regression(max_customers=10000):
    X, y, feature_months = build_regression_data(max_customers)
    if len(X) < 10:
        return {"error": "Không đủ dữ liệu cho hồi quy."}
        
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(Xtr, ytr)
    pred = model.predict(Xte)
    
    return {
        "mae": round(mean_absolute_error(yte, pred), 2),
        "r2": round(r2_score(yte, pred), 4),
        "coef": model.coef_.tolist(),
        "intercept": round(float(model.intercept_), 4),
        "n_train": len(Xtr),
        "n_test": len(Xte),
        "feature_months": feature_months
    }
