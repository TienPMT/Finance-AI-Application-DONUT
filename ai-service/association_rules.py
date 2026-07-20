"""
Khai thác luật kết hợp với thuật toán Apriori (Syllabus Chương 2).

Bài toán: Tìm các tập sản phẩm thường xuyên mua cùng nhau trong một giỏ hàng (basket).
Output: Frequent Itemsets & Association Rules (Support, Confidence, Lift).

Sử dụng dữ liệu Ta-Feng, không dùng thư viện ngoài (chỉ Python thuần).
"""

from __future__ import annotations

import csv
import sys
import time
from collections import defaultdict
from itertools import combinations

from run_tafeng import CSV_PATH


class Apriori:
    def __init__(self, min_support: float = 0.01, min_confidence: float = 0.5):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.n_transactions = 0
        self.frequent_itemsets = {}  # {length: {itemset: count}}

    def _get_frequent_1_itemsets(self, transactions: list[list[str]]):
        counts = defaultdict(int)
        for basket in transactions:
            for item in basket:
                counts[frozenset([item])] += 1
        
        min_abs_sup = self.min_support * self.n_transactions
        f1 = {itemset: count for itemset, count in counts.items() if count >= min_abs_sup}
        return f1

    def _generate_candidates(self, prev_frequent: list[frozenset], k: int):
        candidates = set()
        prev_list = list(prev_frequent)
        for i in range(len(prev_list)):
            for j in range(i + 1, len(prev_list)):
                # Join step: ghép 2 tập k-1 có k-2 phần tử chung
                l1, l2 = sorted(list(prev_list[i])), sorted(list(prev_list[j]))
                if l1[:-1] == l2[:-1]:
                    candidates.add(prev_list[i] | prev_list[j])
        return candidates

    def run(self, transactions: list[list[str]]):
        self.n_transactions = len(transactions)
        if self.n_transactions == 0: return []

        # Bước 1: Tìm tập phổ biến 1-item
        current_frequent = self._get_frequent_1_itemsets(transactions)
        k = 1
        while current_frequent:
            self.frequent_itemsets[k] = current_frequent
            k += 1
            # Bước 2: Tạo ứng viên k-item
            candidates = self._generate_candidates(list(current_frequent.keys()), k)
            
            # Bước 3: Đếm ứng viên
            counts = defaultdict(int)
            for basket in transactions:
                # Chỉ kiểm tra nếu giỏ có đủ k item
                if len(basket) < k: continue
                basket_set = set(basket)
                for cand in candidates:
                    if cand.issubset(basket_set):
                        counts[cand] += 1
            
            # Bước 4: Lọc theo min_support
            min_abs_sup = self.min_support * self.n_transactions
            current_frequent = {itemset: count for itemset, count in counts.items() if count >= min_abs_sup}
            if k > 5: break # Chặn bùng nổ tổ hợp

        # Bước 5: Sinh luật kết hợp
        rules = []
        for k, itemsets in self.frequent_itemsets.items():
            if k < 2: continue
            for itemset, count in itemsets.items():
                # Sinh mọi luật con A -> B
                for i in range(1, k):
                    for ante in combinations(itemset, i):
                        ante = frozenset(ante)
                        cons = itemset - ante
                        
                        # confidence = sup(A U B) / sup(A)
                        sup_a = self._get_count(ante)
                        conf = count / sup_a if sup_a > 0 else 0
                        
                        if conf >= self.min_confidence:
                            sup_b = self._get_count(cons)
                            lift = conf / (sup_b / self.n_transactions) if sup_b > 0 else 0
                            rules.append({
                                "ante": sorted(list(ante)),
                                "cons": sorted(list(cons)),
                                "support": round(count / self.n_transactions, 4),
                                "confidence": round(conf, 4),
                                "lift": round(lift, 4)
                            })
        return sorted(rules, key=lambda r: r["lift"], reverse=True)

    def _get_count(self, itemset: frozenset):
        k = len(itemset)
        return self.frequent_itemsets.get(k, {}).get(itemset, 0)


def load_baskets(path=CSV_PATH, max_rows=50000, item_col="PRODUCT_SUBCLASS"):
    """Biến dữ liệu Ta-Feng thành danh sách giỏ hàng (không xét sequence)."""
    baskets = defaultdict(set)
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {n.strip(): i for i, n in enumerate(header)}
        i_cid, i_dt, i_item = idx["CUSTOMER_ID"], idx["TRANSACTION_DT"], idx[item_col]
        
        count = 0
        for r in reader:
            cid, dt, item = r[i_cid].strip(), r[i_dt].strip(), r[i_item].strip()
            if not cid or not item: continue
            baskets[(cid, dt)].add(item)
            count += 1
            if count >= max_rows: break
    return [list(b) for b in baskets.values()]


if __name__ == "__main__":
    print("[*] Đang tải dữ liệu Ta-Feng...")
    data = load_baskets(max_rows=100000)
    print(f"[*] Đã tải {len(data)} giỏ hàng. Bắt đầu Apriori (min_sup=0.01)...")
    
    start = time.time()
    model = Apriori(min_support=0.01, min_confidence=0.3)
    rules = model.run(data)
    end = time.time()
    
    print(f"[*] Tìm thấy {len(rules)} luật kết hợp trong {end-start:.2f}s")
    print("-" * 60)
    print(f"{'Luật (Ante -> Cons)':40s} | {'Sup':>6s} | {'Conf':>6s} | {'Lift':>6s}")
    for r in rules[:15]:
        ante_str = "{" + ",".join(r["ante"]) + "}"
        cons_str = "{" + ",".join(r["cons"]) + "}"
        print(f"{ante_str + ' -> ' + cons_str:40s} | {r['support']:6.4f} | {r['confidence']:6.4f} | {r['lift']:6.2f}")
