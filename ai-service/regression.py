"""
Phân tích hồi quy dự báo mức chi tiêu (Syllabus Chương 5).

Bài toán: Dự báo tổng số tiền khách hàng sẽ chi tiêu trong tháng kế tiếp
dựa trên lịch sử chi tiêu các tháng trước.

Sử dụng: scikit-learn LinearRegression.
Dữ liệu: Ta-Feng (customer, month, total_spend).
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from run_tafeng import CSV_PATH


def _parse_month(s: str):
    try:
        # Ta-Feng format: MM/DD/YYYY
        mm, dd, yy = s.split("/")
        return int(yy) * 12 + int(mm)
    except Exception:
        return None


def build_regression_data(path: str = CSV_PATH, max_customers: int | None = 5000):
    """
    Xây dựng bảng dữ liệu: Mỗi hàng là 1 khách hàng.
    Features: Chi tiêu tháng 11, 12, 1 (Ta-Feng thường có data 4 tháng).
    Target: Chi tiêu tháng 2.
    """
    cust_monthly_spend = defaultdict(lambda: defaultdict(float))
    
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {n.strip(): i for i, n in enumerate(header)}
        i_cid = idx["CUSTOMER_ID"]
        i_dt = idx["TRANSACTION_DT"]
        i_price = idx["SALES_PRICE"]
        i_amt = idx["AMOUNT"]
        
        for r in reader:
            cid = r[i_cid].strip()
            if not cid: continue
            
            month_abs = _parse_month(r[i_dt])
            if month_abs is None: continue
            
            try:
                val = float(r[i_price]) * float(r[i_amt])
            except ValueError:
                continue
            
            cust_monthly_spend[cid][month_abs] += val
            if max_customers and len(cust_monthly_spend) >= max_customers and cid not in cust_monthly_spend:
                continue

    # Ta-Feng data 2000-11 -> 2001-02
    # m1=24011 (Nov), m2=24012 (Dec), m3=24013 (Jan), target=24014 (Feb)
    months = [24011, 24012, 24013]
    target_month = 24014
    
    X, y = [], []
    for cid, spends in cust_monthly_spend.items():
        # Chỉ lấy khách có xuất hiện trong ít nhất 2 tháng để có dữ liệu học
        if len(spends) < 2: continue
        
        row = [spends.get(m, 0.0) for m in months]
        target = spends.get(target_month, 0.0)
        
        X.append(row)
        y.append(target)
        
    return np.array(X), np.array(y)


def run_regression():
    print("[*] Đang chuẩn bị dữ liệu hồi quy từ Ta-Feng...")
    X, y = build_regression_data(max_customers=10000)
    
    if len(X) < 100:
        print(f"[!] Quá ít dữ liệu ({len(X)} mẫu).")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    # Đánh giá
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("-" * 60)
    print("KẾT QUẢ HỒI QUY TUYẾN TÍNH (Linear Regression)")
    print("-" * 60)
    print(f"Số lượng mẫu huấn luyện : {len(X_train)}")
    print(f"Số lượng mẫu kiểm thử   : {len(X_test)}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} (VND)")
    print(f"R-squared Score (R2)    : {r2:.4f}")
    print("-" * 60)
    print("Hệ số hồi quy (Coefficients):")
    labels = ["Tháng 11", "Tháng 12", "Tháng 01"]
    for label, coef in zip(labels, model.coef_):
        print(f"  {label:10s}: {coef:8.4f}")
    print(f"  Hệ số chặn (Intercept): {model.intercept_:.4f}")
    print("-" * 60)
    
    # Demo 5 khách hàng đầu tiên trong tập test
    print("Dự báo thực tế (5 khách hàng đầu):")
    print(f"{'Thực tế':>12s} | {'Dự báo':>12s} | {'Chênh lệch':>12s}")
    for i in range(min(5, len(y_test))):
        diff = y_pred[i] - y_test[i]
        print(f"{y_test[i]:12.1f} | {y_pred[i]:12.1f} | {diff:12.1f}")


if __name__ == "__main__":
    run_regression()
