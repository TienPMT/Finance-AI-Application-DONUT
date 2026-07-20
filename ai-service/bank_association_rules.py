"""
Apriori cho Bank Data.
"""

from __future__ import annotations

import json
from collections import defaultdict
from association_rules import Apriori
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BANK_JSON_PATH = os.path.join(BASE_DIR, "Version3", "transactions_bank_clean.json")

def load_bank_baskets(path=BANK_JSON_PATH, max_rows=50000):
    with open(path, 'r', encoding='utf-8') as f:
        txs = json.load(f)
    
    baskets = defaultdict(set)
    for t in txs[:max_rows]:
        baskets[(t['user_id'], t['date'])].add(t['category'])
    return [list(b) for b in baskets.values()]

def run_apriori(min_sup=0.01, min_conf=0.3, max_rows=50000):
    data = load_bank_baskets(max_rows=max_rows)
    model = Apriori(min_support=min_sup, min_confidence=min_conf)
    rules = model.run(data)
    return rules, len(data)
