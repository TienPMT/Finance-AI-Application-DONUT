"""
Demo TaFeng cho môn Khai Phá Dữ Liệu: chạy WCM-SPADE (paper ICT-1) trên bộ
Ta-Feng và so sánh hiệu năng next-basket với các baseline TOP, Item-KNN,
User-KNN (kiểu TIFU-KNN) và K-Means recommender.

Giao thức đánh giá: last-basket holdout (giống paper).
  - mỗi khách >= 2 basket: history = các basket trừ basket cuối, truth = basket cuối.
  - mọi recommender fit trên history của TẤT CẢ khách, rồi dự đoán basket cuối.

Metric @K (K=5,10): Recall, Precision, F1, PHR (hit ratio), MAP, MRR, NDCG.

Kết quả được cache theo (min_sup, alpha, max_customers, item_col) để bấm lại nhanh.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from run_tafeng import load_tafeng, CSV_PATH
from wcm_spade import WCMSpade, NextBasketPredictor


ALGOS = ["WCM-SPADE", "TOP", "Item-KNN", "User-KNN", "K-Means"]

_CACHE: dict = {}
_DETAIL: dict = {}


def _key(min_sup, alpha, max_customers, item_col, ks):
    return (round(min_sup, 4), round(alpha, 3), int(max_customers), item_col, tuple(ks))





def _prepare(item_col: str, max_customers: int):
    sequences, item_prices, cids = load_tafeng(
        CSV_PATH, item_col, max_customers=max_customers, return_cids=True)
    seqs, kept_cids = [], []
    for s, c in zip(sequences, cids):
        if len(s) >= 2:
            seqs.append(s)
            kept_cids.append(c)
    histories = [s[:-1] for s in seqs]
    truths = [set(s[-1]) for s in seqs]
    return histories, truths, item_prices, kept_cids


def _build_matrix(histories):
    """User-item count matrix (CSR) từ history. Trả về (X, items)."""
    vocab = sorted({it for h in histories for basket in h for it in basket})
    idx = {it: i for i, it in enumerate(vocab)}
    rows, cols, data = [], [], []
    for u, h in enumerate(histories):
        cnt = defaultdict(int)
        for basket in h:
            for it in basket:
                cnt[idx[it]] += 1
        for c, v in cnt.items():
            rows.append(u)
            cols.append(c)
            data.append(v)
    X = csr_matrix((data, (rows, cols)),
                   shape=(len(histories), len(vocab)), dtype=np.float64)
    return X, vocab


def _topn_from_scores(scores: np.ndarray, items, n: int):
    """Lấy top-n item (theo nhãn) cho mỗi user từ ma trận điểm dense (U x I)."""
    U, I = scores.shape
    n = min(n, I)
    part = np.argpartition(-scores, n - 1, axis=1)[:, :n]
    out = []
    for u in range(U):
        cand = part[u]
        cand = cand[np.argsort(-scores[u, cand])]
        out.append([items[i] for i in cand])
    return out





def _rec_top(X, items, n):
    pop = np.asarray(X.sum(axis=0)).ravel()
    order = [items[i] for i in np.argsort(-pop)[:n]]
    return [order for _ in range(X.shape[0])]


def _rec_item_knn(X, items, n):
    sim = cosine_similarity(X.T)
    np.fill_diagonal(sim, 0.0)
    scores = X @ sim
    return _topn_from_scores(np.asarray(scores), items, n)


def _rec_user_knn(X, items, n, k_neighbors=100):
    Xn = normalize(X)
    k = min(k_neighbors + 1, X.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(Xn)
    dist, ind = nn.kneighbors(Xn)
    Xd = X.toarray()
    scores = np.zeros_like(Xd)
    for u in range(X.shape[0]):
        for j in range(1, k):
            w = 1.0 - dist[u, j]
            scores[u] += w * Xd[ind[u, j]]
    return _topn_from_scores(scores, items, n)


def _rec_kmeans(X, items, n, n_clusters=50):
    c = min(n_clusters, max(2, X.shape[0] // 20))
    km = KMeans(n_clusters=c, n_init=4, random_state=42)
    labels = km.fit_predict(normalize(X))
    Xd = X.toarray()
    cluster_pop = np.zeros((c, X.shape[1]))
    for cl in range(c):
        mask = labels == cl
        if mask.any():
            cluster_pop[cl] = Xd[mask].sum(axis=0)
    scores = cluster_pop[labels]
    return _topn_from_scores(scores, items, n)


def _rec_wcm_spade(histories, item_prices, min_sup, alpha, n):
    engine = WCMSpade(
        min_sup=min_sup,
        alpha=alpha,
        max_pattern_length=4,
        max_items=500,
        max_patterns=30000,
    )
    result = engine.mine(histories, item_prices)
    predictor = NextBasketPredictor(result.patterns, result.item_weights)
    ranked = []
    for h in histories:
        ranked.append([item for item, _ in predictor.predict(h, top_k=n)])
    meta = {
        "nPatterns": len(result.patterns),
        "meanW": round(result.mean_w, 4),
        "effectiveMinSup": result.effective_min_sup,
        "minSupAbs": result.min_sup_abs,
    }
    return ranked, meta, predictor





RAW_METRICS = ["recall", "precision", "f1", "phr", "map", "mrr", "ndcg"]
VALUE_METRICS = ["value_recall", "value_precision"]
ALL_METRICS = RAW_METRICS + VALUE_METRICS


def _metrics(ranked_per_user, truths, k, item_prices=None):
    """
    Tính metric @k. Ngoài metric "thô" (đếm số món trúng), bổ sung metric
    "thiên giá trị" (Value-Aware) dùng giá item làm trọng số:

      Value-Recall@k    = (tổng GIÁ TRỊ các món dự đoán trúng) / (tổng giá trị giỏ truth)
      Value-Precision@k = (tổng giá trị các món trúng) / (tổng giá trị các món được gợi ý)

    Ý nghĩa: TOP/KNN hay trúng các món rẻ phổ biến (kẹo, tăm bông) → Recall thô cao
    nhưng giá trị bắt được thấp. WCM-SPADE ưu tiên món giá trị lớn (thịt cá, đồ gia
    dụng) → Value-Recall cao hơn. Kèm theo:
      _totals.valueCoverage = tổng VND bắt được / tổng VND của mọi giỏ truth (toàn cục)
      _totals.recAvgPrice   = giá TB của 1 món được gợi ý (TOP thấp, WCM-SPADE cao)
    """
    item_prices = item_prices or {}
    pr = lambda it: float(item_prices.get(it, 0.0))

    agg = defaultdict(float)
    n = 0
    tot_truth_val = tot_hit_val = tot_rec_val = 0.0
    tot_rec_items = 0
    for ranked, truth in zip(ranked_per_user, truths):
        if not truth:
            continue
        preds = ranked[:k]
        if not preds:
            continue
        n += 1
        hit = sum(1 for p in preds if p in truth)
        rec = hit / len(truth)
        prec = hit / k
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        num_hits, ap, rr, dcg = 0, 0.0, 0.0, 0.0
        for rank, item in enumerate(preds, start=1):
            if item in truth:
                num_hits += 1
                ap += num_hits / rank
                if rr == 0.0:
                    rr = 1.0 / rank
                dcg += 1.0 / math.log2(rank + 1)
        denom = min(len(truth), k)
        ap = ap / denom if num_hits > 0 else 0.0
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, denom + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0


        truth_val = sum(pr(t) for t in truth)
        hit_val = sum(pr(p) for p in preds if p in truth)
        rec_val = sum(pr(p) for p in preds)
        agg["value_recall"] += (hit_val / truth_val) if truth_val > 0 else 0.0
        agg["value_precision"] += (hit_val / rec_val) if rec_val > 0 else 0.0
        tot_truth_val += truth_val
        tot_hit_val += hit_val
        tot_rec_val += rec_val
        tot_rec_items += len(preds)

        agg["recall"] += rec
        agg["precision"] += prec
        agg["f1"] += f1
        agg["phr"] += 1.0 if hit > 0 else 0.0
        agg["map"] += ap
        agg["mrr"] += rr
        agg["ndcg"] += ndcg

    if n == 0:
        out = {m: 0.0 for m in ALL_METRICS}
        out["_totals"] = {"truthValue": 0.0, "hitValue": 0.0,
                          "valueCoverage": 0.0, "recAvgPrice": 0.0}
        return out

    out = {m: round(agg[m] / n, 4) for m in ALL_METRICS}
    out["_totals"] = {
        "truthValue": round(tot_truth_val, 1),
        "hitValue": round(tot_hit_val, 1),
        "valueCoverage": round(tot_hit_val / tot_truth_val, 4) if tot_truth_val > 0 else 0.0,
        "recAvgPrice": round(tot_rec_val / tot_rec_items, 2) if tot_rec_items else 0.0,
    }
    return out





def _per_user_vectors(ranked_per_user, truths, item_prices, k):
    """Trả (recall[], value_recall[], avg_price[]) per-user để kiểm định cặp."""
    pr = lambda it: float((item_prices or {}).get(it, 0.0))
    rec, vrec, aprice = [], [], []
    for ranked, truth in zip(ranked_per_user, truths):
        if not truth:
            continue
        preds = ranked[:k]
        if not preds:
            continue
        hit = sum(1 for p in preds if p in truth)
        rec.append(hit / len(truth))
        tv = sum(pr(t) for t in truth)
        hv = sum(pr(p) for p in preds if p in truth)
        vrec.append((hv / tv) if tv > 0 else 0.0)
        aprice.append(sum(pr(p) for p in preds) / len(preds))
    return np.array(rec), np.array(vrec), np.array(aprice)


def _significance(ranked_all, truths, item_prices, k, ref="WCM-SPADE"):
    """Wilcoxon signed-rank: WCM-SPADE vs từng baseline trên recall / value_recall /
    avg_price (paired theo từng khách). Trả p-value + dấu chênh lệch trung bình."""
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return {}
    
    vecs = {a: _per_user_vectors(ranked_all[a], truths, item_prices, k) for a in ranked_all}
    rrec, rvrec, raprice = vecs[ref]
    
    # Nếu bộ tham chiếu (WCM-SPADE) trống, không thể so sánh
    if len(rrec) == 0:
        return {}
        
    out = {}
    for a, (rec, vrec, aprice) in vecs.items():
        if a == ref:
            continue
        
        # Nếu baseline trống hoặc không khớp kích thước, bỏ qua
        if len(rec) == 0 or len(rec) != len(rrec):
            continue
            
        row = {}
        for name, ours, theirs in [("recall", rrec, rec),
                                    ("valueRecall", rvrec, vrec),
                                    ("avgPrice", raprice, aprice)]:
            try:
                diff = float(np.mean(ours) - np.mean(theirs))
                p = float(wilcoxon(ours, theirs, zero_method="zsplit").pvalue)
            except (ValueError, ZeroDivisionError, TypeError):
                diff = 0.0
                p = 1.0
            row[name] = {"meanDiff": round(diff, 3), "p": round(p, 5),
                         "sig": p < 0.05, "winsRef": diff > 0}
        if row:
            out[a] = row
            
    if not out:
        return {}
    return {"ref": ref, "k": k, "alpha": 0.05, "vs": out}





def run_experiment(min_sup=0.01, alpha=0.5, max_customers=3000,
                   item_col="PRODUCT_SUBCLASS", ks=(5, 10)):
    key = _key(min_sup, alpha, max_customers, item_col, ks)
    if key in _CACHE:
        return _CACHE[key]

    t0 = time.time()
    histories, truths, item_prices, cids = _prepare(item_col, max_customers)
    if len(histories) < 10:
        return {"error": f"Quá ít khách đủ điều kiện ({len(histories)}). Tăng max_customers."}
    X, items = _build_matrix(histories)
    load_t = time.time() - t0

    n = max(ks)
    results, timings, spade_meta = {}, {}, {}
    ranked_all = {}

    def timed(name, fn):
        s = time.time()
        ranked = fn()
        timings[name] = round(time.time() - s, 2)
        ranked_all[name] = ranked
        results[name] = {str(k): _metrics(ranked, truths, k, item_prices) for k in ks}

    s = time.time()
    wcm_ranked, spade_meta, predictor = _rec_wcm_spade(histories, item_prices, min_sup, alpha, n)
    timings["WCM-SPADE"] = round(time.time() - s, 2)
    ranked_all["WCM-SPADE"] = wcm_ranked
    results["WCM-SPADE"] = {str(k): _metrics(wcm_ranked, truths, k, item_prices) for k in ks}

    timed("TOP", lambda: _rec_top(X, items, n))
    timed("Item-KNN", lambda: _rec_item_knn(X, items, n))
    timed("User-KNN", lambda: _rec_user_knn(X, items, n))
    timed("K-Means", lambda: _rec_kmeans(X, items, n))

    significance = _significance(ranked_all, truths, item_prices, k=min(ks))

    counts_arr = np.asarray(X.sum(axis=0)).ravel()
    item_counts = {items[j]: int(counts_arr[j]) for j in range(len(items))}
    _DETAIL[key] = {
        "histories": histories,
        "truths": truths,
        "predictor": predictor,
        "ranked": ranked_all,
        "item_prices": item_prices,
        "item_counts": item_counts,
        "cids": cids,
        "cid_to_idx": {c: i for i, c in enumerate(cids)},
    }


    k0 = str(min(ks))
    value_summary = {}
    for a in ALGOS:
        tot = results[a][k0].get("_totals", {})
        value_summary[a] = {
            "recall": results[a][k0]["recall"],
            "valueRecall": results[a][k0]["value_recall"],
            "valueCoverage": tot.get("valueCoverage", 0.0),
            "recAvgPrice": tot.get("recAvgPrice", 0.0),
            "hitValue": tot.get("hitValue", 0.0),
        }

    out = {
        "params": {"minSup": min_sup, "alpha": alpha, "maxCustomers": max_customers,
                   "itemCol": item_col, "ks": list(ks)},
        "meta": {
            "nCustomers": len(histories),
            "nItems": len(items),
            "nEval": len(truths),
            "loadSec": round(load_t, 1),
            "spade": spade_meta,
        },
        "timings": timings,
        "results": results,
        "algos": ALGOS,
        "metrics": ALL_METRICS,
        "valueSummary": value_summary,
        "valueK": int(min(ks)),
        "significance": significance,
    }
    _CACHE[key] = out
    return out


def predict_customer(customer_idx, min_sup=0.01, alpha=0.5, max_customers=3000,
                     item_col="PRODUCT_SUBCLASS", top_k=10, ks=(5, 10)):
    """Demo tương tác: Top-K dự đoán của mọi model cho 1 khách + luật WCM-SPADE đã khớp."""
    exp = run_experiment(min_sup, alpha, max_customers, item_col, ks)
    if "error" in exp:
        return exp
    key = _key(min_sup, alpha, max_customers, item_col, ks)
    d = _DETAIL[key]
    n_cust = len(d["histories"])
    if customer_idx < 0 or customer_idx >= n_cust:
        return {"error": f"customer_idx ngoài khoảng [0, {n_cust - 1}]"}

    top_k = min(top_k, max(ks))
    history = d["histories"][customer_idx]
    truth = d["truths"][customer_idx]

    item_prices = d.get("item_prices") or {}
    pr = lambda it: float(item_prices.get(it, 0.0))
    truth_value = round(sum(pr(t) for t in truth), 1)


    predictions = {}
    value_capture = {}
    for algo in ALGOS:
        preds = d["ranked"][algo][customer_idx][:top_k]
        predictions[algo] = [{"item": str(it), "hit": it in truth} for it in preds]
        hit_val = round(sum(pr(p) for p in preds if p in truth), 1)
        rec_val = sum(pr(p) for p in preds)
        value_capture[algo] = {
            "hitValue": hit_val,
            "valueRecall": round(hit_val / truth_value, 4) if truth_value > 0 else 0.0,
            "recAvgPrice": round(rec_val / len(preds), 1) if preds else 0.0,
        }


    predictor = d["predictor"]
    matched, seen = [], set()
    for basket in reversed(history):
        bset = frozenset(basket)
        for last_ante, entries in predictor.index.items():
            if last_ante.issubset(bset):
                for consequent, wsup in entries:
                    sig = (tuple(sorted(map(str, last_ante))), tuple(map(str, consequent)))
                    if sig in seen:
                        continue
                    seen.add(sig)
                    matched.append({
                        "ante": [str(i) for i in sorted(last_ante)],
                        "consequent": [str(i) for i in consequent],
                        "wsup": round(wsup, 3),
                    })
    matched.sort(key=lambda m: m["wsup"], reverse=True)
    matched = matched[:20]


    import item_labels
    ref = set()
    for basket in history:
        ref.update(str(i) for i in basket)
    ref.update(str(i) for i in truth)
    for preds in predictions.values():
        ref.update(p["item"] for p in preds)
    for m in matched:
        ref.update(m["ante"])
        ref.update(m["consequent"])
    labels = item_labels.build_labels(ref, d.get("item_prices"), d.get("item_counts"))

    return {
        "customerIdx": customer_idx,
        "customerId": d["cids"][customer_idx] if customer_idx < len(d.get("cids", [])) else None,
        "nCustomers": n_cust,
        "history": [[str(i) for i in basket] for basket in history],
        "truth": [str(i) for i in sorted(truth)],
        "topK": top_k,
        "predictions": predictions,
        "truthValue": truth_value,
        "valueCapture": value_capture,
        "matchedPatterns": matched,
        "nMatched": len(matched),
        "labels": labels,
    }








DEMO_CUSTOMER_IDS: list[str] = [
    "01327205", "02112596", "02040677", "02032849", "00368568", "02006246",
]


_DEMO_PARAMS = dict(min_sup=0.01, alpha=0.5, max_customers=3000,
                    item_col="PRODUCT_SUBCLASS", ks=(5, 10))


def spade_predict(customer_id, top_k=8):
    """Tra cứu theo CUSTOMER_ID thật → trả full payload predict_customer + customerId.

    Dùng cho POST /spade/predict {"customer_id": "..."} của tab 'Trợ lý Chi tiêu'.
    """
    p = _DEMO_PARAMS
    exp = run_experiment(p["min_sup"], p["alpha"], p["max_customers"],
                         p["item_col"], p["ks"])
    if "error" in exp:
        return exp
    key = _key(p["min_sup"], p["alpha"], p["max_customers"], p["item_col"], p["ks"])
    d = _DETAIL[key]
    cid = str(customer_id)
    idx = d["cid_to_idx"].get(cid)
    if idx is None:
        return {"error": f"CUSTOMER_ID {cid} không có trong Data Lake (tập phân tích).",
                "demoCustomerIds": DEMO_CUSTOMER_IDS}
    out = predict_customer(idx, p["min_sup"], p["alpha"], p["max_customers"],
                           p["item_col"], top_k, p["ks"])
    out["customerId"] = cid
    out["savingsTips"] = _savings_tips(out)
    return out


def _savings_tips(out):
    """Sinh gợi ý tiết kiệm từ luật thói quen + giỏ dự báo (giá thật từ CSV).

    Mọi con số đều tính từ dữ liệu (wsup, giá TB SALES_PRICE quy đổi VND
    theo item_labels.TWD_TO_VND) — không bịa.
    """
    labels = out.get("labels", {})

    def name(code):
        l = labels.get(str(code))
        return l["name"] if l and l.get("name") else str(code)

    def price(code):
        l = labels.get(str(code))
        return float(l["price"]) if l and l.get("price") is not None else 0.0

    def vnd(x):
        return f"{x:,.0f}".replace(",", ".") + "đ"

    tips = []
    matched = out.get("matchedPatterns", [])
    preds = [p["item"] for p in out.get("predictions", {}).get("WCM-SPADE", [])]
    n_baskets = len(out.get("history", []))

    if matched:
        top = matched[0]
        ante = ", ".join(name(c) for c in top["ante"])
        cons = ", ".join(name(c) for c in top["consequent"])
        tips.append({
            "icon": "habit",
            "text": (f"Bạn có thói quen mua {cons} sau khi đã mua {ante} "
                     f"(trọng số {top['wsup']}). Lên danh sách trước khi đi siêu thị "
                     f"để mua một lần, tránh đi nhiều chuyến phát sinh mua linh tinh."),
        })

    if preds:
        spend = round(sum(price(c) for c in preds), 1)
        if spend > 0:
            tips.append({
                "icon": "budget",
                "text": (f"Theo thói quen cũ, lần mua tới bạn sẽ tốn khoảng {vnd(spend)} "
                         f"cho {len(preds)} món quen thuộc. Đặt trần ngân sách bằng đúng "
                         f"con số này — vượt là đang mua ngoài kế hoạch."),
            })
        priced = [(price(c), c) for c in preds if price(c) > 0]
        if priced:
            pmax, cmax = max(priced)
            if spend > 0 and pmax / spend >= 0.3:
                tips.append({
                    "icon": "deal",
                    "text": (f"{name(cmax)} (≈{vnd(pmax)}) chiếm phần lớn chi phí sắp tới. "
                             f"Canh khuyến mãi hoặc so giá món này trước — giảm được nó "
                             f"là giảm mạnh cả giỏ."),
                })

    if n_baskets >= 4 and matched:
        tips.append({
            "icon": "plan",
            "text": (f"Lịch sử có {n_baskets} chuyến mua sắm. Các món hay đi cặp ở trên "
                     f"nên gộp chung một chuyến — mỗi chuyến bớt đi là bớt một lần "
                     f"chi tiêu ngoài dự kiến."),
        })
    return tips
