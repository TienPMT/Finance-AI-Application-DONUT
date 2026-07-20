"""
Chọn 5-6 CUSTOMER_ID THẬT "ngon nhất" làm thẻ thành viên demo cho tab
'Trợ lý Chi tiêu'. Tiêu chí (demo phải nhìn thuyết phục):

  - WCM-SPADE dự báo giỏ kế tiếp TRÚNG truth (hit@top_k > 0)
  - có LUẬT thói quen khớp lịch sử (matchedPatterns > 0) → kể được "A → B"
  - lịch sử đủ giàu (>= 3 basket) → câu chuyện cá nhân hóa rõ
  - bắt được GIÁ TRỊ kinh tế cao (valueRecall WCM-SPADE cao)

Chạy:  .venv/bin/python pick_demo_customers.py
In ra top khách + dòng DEMO_CUSTOMER_IDS để dán vào tafeng_demo.py.
"""

import tafeng_demo as T

P = T._DEMO_PARAMS


def main():
    exp = T.run_experiment(P["min_sup"], P["alpha"], P["max_customers"],
                           P["item_col"], P["ks"])
    if "error" in exp:
        print("LỖI:", exp["error"])
        return
    key = T._key(P["min_sup"], P["alpha"], P["max_customers"], P["item_col"], P["ks"])
    d = T._DETAIL[key]
    cids = d["cids"]
    histories = d["histories"]
    truths = d["truths"]
    ranked = d["ranked"]["WCM-SPADE"]
    predictor = d["predictor"]
    ip = d["item_prices"]
    pr = lambda it: float(ip.get(it, 0.0))
    top_k = 8

    scored = []
    for i, cid in enumerate(cids):
        hist = histories[i]
        truth = truths[i]
        if len(hist) < 2 or not truth:
            continue
        preds = ranked[i][:top_k]
        hit = sum(1 for p in preds if p in truth)
        if hit == 0:
            continue


        n_matched = 0
        seen = set()
        for basket in hist:
            bset = frozenset(basket)
            for last_ante, entries in predictor.index.items():
                if last_ante.issubset(bset):
                    for consequent, _w in entries:
                        sig = (tuple(sorted(map(str, last_ante))), tuple(map(str, consequent)))
                        if sig not in seen:
                            seen.add(sig)
                            n_matched += 1
        if n_matched == 0:
            continue

        truth_val = sum(pr(t) for t in truth)
        hit_val = sum(pr(p) for p in preds if p in truth)
        vrec = (hit_val / truth_val) if truth_val > 0 else 0.0
        rec_val = sum(pr(p) for p in preds)


        score = (hit * 2.0) + min(n_matched, 10) * 0.3 + vrec * 3.0 + min(len(hist), 6) * 0.2
        scored.append({
            "cid": cid, "idx": i, "nBaskets": len(hist) + 1,
            "hit": hit, "nMatched": n_matched, "vrec": round(vrec, 3),
            "spend": round(rec_val, 1), "truthVal": round(truth_val, 1),
            "score": round(score, 3),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    print(f"\nĐủ điều kiện: {len(scored)} khách. Top 15:")
    print(f"{'CUSTOMER_ID':>12} {'baskets':>7} {'hit':>4} {'rules':>6} "
          f"{'vRec':>6} {'spend':>9} {'score':>7}")
    for s in scored[:15]:
        print(f"{s['cid']:>12} {s['nBaskets']:>7} {s['hit']:>4} {s['nMatched']:>6} "
              f"{s['vrec']:>6} {s['spend']:>9} {s['score']:>7}")

    pick = [s["cid"] for s in scored[:6]]
    print("\n>>> Dán vào tafeng_demo.py:")
    print(f"DEMO_CUSTOMER_IDS = {pick}")


if __name__ == "__main__":
    main()
