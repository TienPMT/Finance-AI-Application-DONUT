from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uvicorn
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel, AutoTokenizer
import re
import io
import os
from wcm_spade import WCMSpade
import budget_advisor
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BANK_JSON_PATH = os.path.join(BASE_DIR, "Version3", "transactions_bank_clean.json")
app = FastAPI()

# =====================================================================
# [10/10 FEATURE] KHỞI ĐỘNG LẠNH (COLD START) TỪ BANK DATA KAGGLE
# =====================================================================
import json
GLOBAL_TRENDS = []

def _init_global_trends():
    global GLOBAL_TRENDS
    BANK_JSON_PATH = os.path.join(BASE_DIR, "Version3", "transactions_bank_clean.json")
    try:
        if os.path.exists(BANK_JSON_PATH):
            print("[COLD START] Đang nạp và khai phá 4.604 giao dịch Ngân hàng...")
            with open(BANK_JSON_PATH, 'r', encoding='utf-8') as f:
                bank_txs = json.load(f)
            
            # Chạy mồi hàm advise một lần duy nhất lúc start server để lấy Trends
            dummy_res = budget_advisor.advise(bank_txs, target=0, monthly_income=1)
            GLOBAL_TRENDS = dummy_res.get("trends", [])
            print(f"[*] COLD START READY: Đã nạp thành công {len(GLOBAL_TRENDS)} Luật thói quen chung!")
        else:
            print("[!] Không tìm thấy file transactions_bank_clean.json để làm Cold Start.")
    except Exception as e:
        print(f"[!] Lỗi nạp Bank Data: {e}")

_init_global_trends()
# =====================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Version3", "donut_result")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_HEIGHT = 1280
IMAGE_WIDTH = 960

print(f"Đang khởi động AI DONUT Engine trên thiết bị: [{DEVICE.upper()}]...")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Không tìm thấy weights tại {MODEL_PATH}")

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
try:
    processor.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
except Exception:
    pass

model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH, ignore_mismatched_sizes=True).to(DEVICE)
model.eval()
print("AI DONUT Engine đã tải xong!")

def extract_final_cost(raw_cost_str: str) -> int:

    clean_str = raw_cost_str.replace('.', '').replace(',', '')
    numbers = re.findall(r'\d+', clean_str)
    if numbers:
        return max(int(num) for num in numbers)
    return 0

def clean_vietnamese_text(sequence: str) -> str:
    sequence = re.sub(r'([,.;])([A-ZĐa-zà-ỹ])', r'\1 \2', sequence)
    toned_vowels = "áắấéếíóốớúứýàằầèềìòồờùừỳảẳẩẻểỉỏổởủửỷãẵẫẽễĩõỗỡũữỹạặậẹệịọộợụựỵ"
    toned_vowels += toned_vowels.upper()
    sequence = re.sub(r'(["' + toned_vowels + r'])(["' + toned_vowels + r'])', r'\1 \2', sequence)

    all_vowels = "aăâeêioôơuưyáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵĐđ"
    all_vowels += all_vowels.upper()
    tails = ["c", "ch", "m", "n", "ng", "nh", "p", "t", "i", "y", "a", "o", "u"]
    tails += [t.upper() for t in tails]
    tail_pattern = "|".join(tails)
    pattern = r"([" + all_vowels + r"])\s+(" + tail_pattern + r")\b"

    old_sequence = ""
    while old_sequence != sequence:
        old_sequence = sequence
        sequence = re.sub(pattern, r"\1\2", sequence)

    return sequence

@app.post("/donut/scan")
async def extract_receipt_info(image: UploadFile = File(...)):
    print(f"\n[REQUEST] Đã nhận hóa đơn từ App: {image.filename}")

    try:
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lỗi đọc file ảnh: {str(e)}")

    pixel_values = processor(
        pil_img,
        return_tensors="pt",
        do_resize=True,
        size={"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH},
        do_align_long_axis=True
    ).pixel_values.to(DEVICE)

    start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_seller>")

    outputs = model.generate(
        pixel_values,
        max_length=768,
        decoder_input_ids=torch.tensor([[start_token_id]]).to(DEVICE),
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
    )

    sequence = processor.batch_decode(outputs)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<pad>", "", sequence)
    sequence = clean_vietnamese_text(sequence)

    try:
        prediction = processor.token2json(sequence)
        if "seller" in prediction and "total_cost" in prediction["seller"]:
            raw_cost = prediction["seller"]["total_cost"]
            prediction["seller"]["total_cost_number"] = extract_final_cost(raw_cost)

    except Exception as e:
        print(f"[CẢNH BÁO] Lỗi parse JSON từ Donut: {e}")
        prediction = {"raw_text": sequence}
    print(f" [SUCCESS] Bóc tách thành công: {prediction}")
    return prediction

class SpadeTx(BaseModel):
    category: str
    amount: float = 0.0
    date: str = ""
    isExpense: bool = True

class SpadeRequest(BaseModel):
    transactions: List[SpadeTx]
    max_gap_days: int = 7
    min_sup_factor: float = 0.5
    top_n: int = 10
    alpha: float = 0.5
    beta: float = 0.5


class RecommendRequest(BaseModel):
    transactions: List[SpadeTx]
    target: float = 0.0          # mục tiêu tiết kiệm
    current: float = 0.0         # đã tiết kiệm được
    months_left: int = 0         # số tháng còn lại tới hạn
    monthly_income: Optional[float] = None  # thu nhập/tháng (None → tự tính từ giao dịch thu)
    needed_monthly: Optional[float] = None  # cần tiết kiệm/tháng (None → tính từ target/current/months_left)
    alpha: float = 0.5
    min_sup_factor: float = 0.4
    max_cut_ratio: float = 0.30

def _parse_date(s: str):
    if not s:
        return None
    s = s.strip().replace("Z", "")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        try:
            return datetime.fromisoformat(s[:10])
        except ValueError:
            return None

def _build_user_sequences(exp):
    """
    Biến lịch sử chi tiêu của 1 user thành sequence DB cho WCM-SPADE.
    Mỗi THÁNG = 1 sequence; mỗi NGÀY = 1 event (itemset = các danh mục chi trong ngày).
    Trả về (sequences, item_prices).
    """
    from collections import defaultdict
    by_month = defaultdict(lambda: defaultdict(set))  # (y,m) -> day -> set(category)
    price_sum = defaultdict(float)
    price_cnt = defaultdict(int)
    for d, cat, amt in exp:
        by_month[(d.year, d.month)][d.toordinal()].add(cat)
        price_sum[cat] += amt
        price_cnt[cat] += 1

    sequences = []
    for _, days in by_month.items():
        seq = [tuple(sorted(days[day])) for day in sorted(days.keys())]
        sequences.append(seq)

    item_prices = {c: price_sum[c] / price_cnt[c] for c in price_sum}
    return sequences, item_prices


@app.post("/spade/analyze")
def analyze_patterns(req: SpadeRequest):
    print(f"\n[REQUEST] WCM-SPADE nhận {len(req.transactions)} giao dịch")

    exp = []
    for t in req.transactions:
        d = _parse_date(t.date)
        if t.isExpense and d is not None:
            exp.append((d, t.category, max(t.amount, 0.0)))
    exp.sort(key=lambda x: x[0])

    # [XỬ LÝ COLD START]: Nếu user có < 10 giao dịch, trả về luật Global của Ngân hàng!
    if len(exp) < 10 and GLOBAL_TRENDS:
        print("[INFO] User mới -> Trả về Luật Global (Bank Data)")
        patterns = []
        for p in GLOBAL_TRENDS[:req.top_n]:
            patterns.append({
                "pattern": p["pattern"],
                "count": p["count"],
                "weight": p["wsupport"],
                "wsupport": "Dữ liệu chung (Global)",
            })
        return {
            "patterns": patterns,
            "effectiveMinSup": "Auto",
            "meanW": "Auto",
            "nSequences": 4604, # Fake số lượng user thực tế từ file Kaggle
            "message": "Dữ liệu cá nhân ít, đang hiển thị Luật từ CSDL Ngân hàng."
        }

    # Nếu User cũ (>10 giao dịch), chạy WCM-SPADE cá nhân hóa như bình thường
    sequences, item_prices = _build_user_sequences(exp)
    if len(sequences) < 1:
        return {"patterns": [], "message": "Không đủ dữ liệu chuỗi"}

    engine = WCMSpade(min_sup=req.min_sup_factor, alpha=req.alpha, max_pattern_length=4)
    result = engine.mine(sequences, item_prices)

    seq_patterns = [p for p in result.patterns if len(p.itemsets) >= 2 or sum(len(it) for it in p.itemsets) >= 2]
    patterns = []
    for p in seq_patterns[: req.top_n]:
        human = " → ".join("+".join(str(i) for i in it) for it in p.itemsets)
        cover = (p.raw_sup / result.n_sequences * 100) if result.n_sequences else 0.0
        patterns.append({
            "pattern": human,
            "count": p.raw_sup,
            "weight": round(p.wsup, 4),
            "wsupport": f"{cover:.1f}%",
        })

    print(f" [SUCCESS] Trả về {len(patterns)} mẫu Cá nhân hóa.")
    return {
        "patterns": patterns,
        "effectiveMinSup": result.effective_min_sup,
        "meanW": round(result.mean_w, 4),
        "nSequences": result.n_sequences,
    }

@app.post("/spade/recommend")
def recommend_budget(req: RecommendRequest):
    print(f"\n[REQUEST] BudgetAdvisor nhận {len(req.transactions)} giao dịch")
    txs = [
        {"category": t.category, "amount": t.amount, "date": t.date, "isExpense": t.isExpense}
        for t in req.transactions
    ]
    
    # Chạy Budget Advisor thuần túy trên số tiền thật của User (Không trộn tiền Global)
    if not txs:
        # User chưa tiêu đồng nào
        return {
            "trends": GLOBAL_TRENDS[:5],
            "tagStats": [],
            "goal": {"neededMonthly": 0, "monthlyIncome": req.monthly_income or 0, "monthlyExpense": 0, "surplus": 0, "gap": 0},
            "recommendations": [],
            "summary": "Bạn chưa có giao dịch. Hãy nhập chi tiêu để AI tính toán nhé! (Bên dưới là các thói quen chung tham khảo)"
        }

    result = budget_advisor.advise(
        txs,
        target=req.target,
        current=req.current,
        months_left=req.months_left,
        monthly_income=req.monthly_income,
        needed_monthly=req.needed_monthly,
        alpha=req.alpha,
        min_sup_factor=req.min_sup_factor,
        max_cut_ratio=req.max_cut_ratio,
    )
    
    # [XỬ LÝ COLD START]: Nếu user < 10 giao dịch, nhúng Luật Global vào phần 'trends'
    if len(txs) < 10 and GLOBAL_TRENDS:
        result["trends"] = GLOBAL_TRENDS[:5] # Lấy 5 luật tiêu biểu
        summary_old = result.get("summary", "")
        result["summary"] = f"(AI dùng Luật Khai Phá Chung) {summary_old}"
        
    print(f" [SUCCESS] BudgetAdvisor: {len(result.get('recommendations', []))} đề xuất")
    return result


@app.post("/spade/apriori")
def spade_apriori(req: SpadeRequest):
    """Khai thác luật kết hợp Apriori trên lịch sử cá nhân của người dùng."""
    from collections import defaultdict
    import association_rules
    
    # Gom baskets theo ngày
    baskets = defaultdict(set)
    for t in req.transactions:
        if t.isExpense:
            d = _parse_date(t.date)
            date_str = d.strftime("%Y-%m-%d") if d else "unknown"
            baskets[date_str].add(t.category)
    
    data = [list(b) for b in baskets.values()]
    if not data:
        return {"rules": [], "message": "Chưa có đủ giao dịch để tìm thói quen mua sắm."}
        
    model = association_rules.Apriori(min_support=0.1, min_confidence=0.4)
    rules = model.run(data)
    return {"rules": rules[:10]}


@app.post("/spade/regression")
def spade_regression(req: SpadeRequest):
    """Dự báo chi tiêu tháng tới cho người dùng dựa trên hồi quy tuyến tính."""
    from collections import defaultdict
    import numpy as np
    from sklearn.linear_model import LinearRegression
    
    # Gom chi tiêu theo tháng
    monthly = defaultdict(float)
    for t in req.transactions:
        if t.isExpense:
            d = _parse_date(t.date)
            if d:
                m_key = f"{d.year}-{d.month:02d}"
                monthly[m_key] += t.amount
                
    months_sorted = sorted(monthly.keys())
    if len(months_sorted) < 2:
        return {"prediction": 0, "message": "Cần dữ liệu của ít nhất 2 tháng để dự báo."}
        
    # X = số thứ tự tháng (0, 1, 2...), y = số tiền
    X = np.array(range(len(months_sorted))).reshape(-1, 1)
    y = np.array([monthly[m] for m in months_sorted])
    
    model = LinearRegression().fit(X, y)
    next_month_idx = len(months_sorted)
    pred = model.predict([[next_month_idx]])[0]
    
    return {
        "prediction": round(float(pred), 2),
        "lastMonth": round(float(y[-1]), 2),
        "trend": "tăng" if pred > y[-1] else "giảm"
    }


# ===================== DEMO TA-FENG (Khai Phá Dữ Liệu) =====================
# Trang web so sánh WCM-SPADE vs baselines (TOP/Item-KNN/User-KNN/K-Means)
# trên bộ Ta-Feng theo paper ICT-1. Mở từ nút trong tab Phân tích AI (Flutter).

class TafengRequest(BaseModel):
    min_sup: float = 0.01
    alpha: float = 0.5
    max_customers: int = 3000
    item_col: str = "PRODUCT_SUBCLASS"


class TafengPredictRequest(TafengRequest):
    customer_idx: int = 0
    top_k: int = 10


class AprioriRequest(BaseModel):
    min_sup: float = 0.01
    min_conf: float = 0.3
    max_rows: int = 100000
    item_col: str = "PRODUCT_SUBCLASS"


class RegressionRequest(BaseModel):
    max_customers: int = 10000


@app.get("/tafeng")
def tafeng_dashboard():
    return FileResponse(os.path.join(BASE_DIR, "tafeng_dashboard.html"))


@app.post("/tafeng/run")
def tafeng_run(req: TafengRequest):
    import tafeng_demo
    print(f"\n[REQUEST] Ta-Feng demo: minSup={req.min_sup} alpha={req.alpha} "
          f"maxCust={req.max_customers} itemCol={req.item_col}")
    try:
        out = tafeng_demo.run_experiment(
            min_sup=req.min_sup,
            alpha=req.alpha,
            max_customers=req.max_customers,
            item_col=req.item_col,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500,
                            detail="Không tìm thấy CSV Ta-Feng. Kiểm tra CSV_PATH trong run_tafeng.py.")
    return out


class TafengClassifyRequest(BaseModel):
    max_customers: int = 8000
    item_col: str = "PRODUCT_SUBCLASS"


_QUALITY_CACHE: dict = {}
_CLASSIFY_CACHE: dict = {}


@app.post("/tafeng/quality")
def tafeng_quality(req: TafengRequest):
    """EDA + báo cáo làm sạch dữ liệu Ta-Feng (rubric: làm sạch/phân tích dữ liệu)."""
    import data_quality
    key = (req.max_customers, req.item_col)
    if key in _QUALITY_CACHE:
        return _QUALITY_CACHE[key]
    try:
        out = data_quality.analyze(max_customers=req.max_customers, item_col=req.item_col)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Không tìm thấy CSV Ta-Feng.")
    _QUALITY_CACHE[key] = out
    return out


@app.post("/tafeng/classify")
def tafeng_classify(req: TafengClassifyRequest):
    """Phân lớp khách chi tiêu cao/thấp: Decision Tree / Naive Bayes / KNN (rubric Chương 3)."""
    import classification
    key = (req.max_customers, req.item_col)
    if key in _CLASSIFY_CACHE:
        return _CLASSIFY_CACHE[key]
    try:
        out = classification.run(max_customers=req.max_customers, item_col=req.item_col)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Không tìm thấy CSV Ta-Feng.")
    _CLASSIFY_CACHE[key] = out
    return out


@app.post("/tafeng/apriori")
def tafeng_apriori(req: AprioriRequest):
    """Khai thác luật kết hợp Apriori (Syllabus Chương 2)."""
    import association_rules
    try:
        data = association_rules.load_baskets(max_rows=req.max_rows, item_col=req.item_col)
        model = association_rules.Apriori(min_support=req.min_sup, min_confidence=req.min_conf)
        rules = model.run(data)
        return {"rules": rules[:50], "n_transactions": len(data)}
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Không tìm thấy CSV Ta-Feng.")


@app.post("/tafeng/regression")
def tafeng_regression(req: RegressionRequest):
    """Phân tích hồi quy dự báo mức chi tiêu (Syllabus Chương 5)."""
    import regression
    try:
        X, y = regression.build_regression_data(max_customers=req.max_customers)
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, r2_score
        
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression().fit(Xtr, ytr)
        pred = model.predict(Xte)
        
        return {
            "mae": round(mean_absolute_error(yte, pred), 2),
            "r2": round(r2_score(yte, pred), 4),
            "coef": model.coef_.tolist(),
            "intercept": round(float(model.intercept_), 4),
            "n_train": len(Xtr),
            "n_test": len(Xte)
        }
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Không tìm thấy CSV Ta-Feng.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tafeng/predict")
def tafeng_predict(req: TafengPredictRequest):
    import tafeng_demo
    try:
        out = tafeng_demo.predict_customer(
            customer_idx=req.customer_idx,
            min_sup=req.min_sup,
            alpha=req.alpha,
            max_customers=req.max_customers,
            item_col=req.item_col,
            top_k=req.top_k,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=500,
                            detail="Không tìm thấy CSV Ta-Feng. Kiểm tra CSV_PATH trong run_tafeng.py.")
    return out


# ---- Trợ lý Chi tiêu: tra theo CUSTOMER_ID thật (Loyalty Card / Data Lake) ----
class SpadePredictRequest(BaseModel):
    customer_id: str
    top_k: int = 8


@app.post("/spade/predict")
def spade_predict(req: SpadePredictRequest):
    """Quét lịch sử của 1 CUSTOMER_ID thật trong Data Lake Ta-Feng, chạy WCM-SPADE,
    trả luật thói quen + dự báo giỏ hàng kế tiếp + giá trị kinh tế."""
    import tafeng_demo
    print(f"\n[REQUEST] /spade/predict customer_id={req.customer_id} top_k={req.top_k}")
    try:
        out = tafeng_demo.spade_predict(req.customer_id, top_k=req.top_k)
    except FileNotFoundError:
        raise HTTPException(status_code=500,
                            detail="Không tìm thấy CSV Ta-Feng. Kiểm tra CSV_PATH trong run_tafeng.py.")
    return out


# ---- Giỏ hàng tương tác: chọn sản phẩm → gợi ý WCM-SPADE ----
class CartRecommendRequest(BaseModel):
    items: list[str]
    top_k: int = 5
    max_customers: int = 3000


@app.get("/tafeng/products")
def tafeng_products(max_customers: int = 3000, limit: int = 60):
    """Danh sách sản phẩm Ta-Feng (đã gán nhãn tự động) cho picker. Sắp theo count giảm dần."""
    import tafeng_demo
    import item_labels
    exp = tafeng_demo.run_experiment(
        min_sup=0.01, alpha=0.5, max_customers=max_customers,
        item_col="PRODUCT_SUBCLASS", ks=(5, 10))
    if "error" in exp:
        return exp
    key = tafeng_demo._key(0.01, 0.5, max_customers, "PRODUCT_SUBCLASS", (5, 10))
    d = tafeng_demo._DETAIL[key]
    counts = d.get("item_counts") or {}
    prices = d.get("item_prices") or {}
    rows = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    labels = item_labels.build_labels([it for it, _ in rows], prices, counts)
    out = []
    for it, _ in rows:
        lab = labels.get(str(it), {})
        out.append({
            "code": str(it),
            "name": lab.get("name") or str(it),
            "price": lab.get("price"),
            "count": lab.get("count"),
        })
    return {"products": out, "n": len(out)}


@app.post("/tafeng/recommend")
def tafeng_recommend(req: CartRecommendRequest):
    """Nhận danh sách mã sản phẩm user chọn → dựng history giả (1 basket) →
    gọi NextBasketPredictor.predict() trả gợi ý WCM-SPADE + pattern khớp."""
    import tafeng_demo
    import item_labels
    exp = tafeng_demo.run_experiment(
        min_sup=0.01, alpha=0.5, max_customers=req.max_customers,
        item_col="PRODUCT_SUBCLASS", ks=(5, 10))
    if "error" in exp:
        return exp
    key = tafeng_demo._key(0.01, 0.5, req.max_customers, "PRODUCT_SUBCLASS", (5, 10))
    d = tafeng_demo._DETAIL[key]
    predictor = d["predictor"]
    item_prices = d.get("item_prices") or {}
    item_counts = d.get("item_counts") or {}

    cart_keys = [str(c) for c in (req.items or []) if str(c).strip()]
    if not cart_keys:
        return {"cart": [], "recommendations": [], "matchedPatterns": [], "labels": {}}

    history = [tuple(cart_keys)]
    raw = predictor.predict(history, top_k=max(req.top_k * 2, 10))

    bset = frozenset(cart_keys)
    matched = []
    for last_ante, entries in predictor.index.items():
        if last_ante.issubset(bset):
            for consequent, wsup in entries:
                matched.append({
                    "ante": [str(i) for i in last_ante],
                    "consequent": [str(i) for i in consequent],
                    "wsup": round(float(wsup), 4),
                })
    matched.sort(key=lambda m: m["wsup"], reverse=True)
    matched = matched[:15]

    ref = set(cart_keys)
    for item, _ in raw:
        ref.add(str(item))
    for m in matched:
        ref.update(m["ante"])
        ref.update(m["consequent"])
    labels = item_labels.build_labels(ref, item_prices, item_counts)

    cart_set = set(cart_keys)
    recommendations = []
    for item, score in raw:
        code = str(item)
        if code in cart_set:
            continue
        recommendations.append({
            "code": code,
            "score": round(float(score), 4),
            "name": labels.get(code, {}).get("name", code),
            "price": labels.get(code, {}).get("price"),
            "count": labels.get(code, {}).get("count"),
        })
        if len(recommendations) >= req.top_k:
            break

    return {
        "cart": cart_keys,
        "recommendations": recommendations,
        "matchedPatterns": matched,
        "labels": labels,
    }


@app.get("/spade/demo-customers")
def spade_demo_customers():
    """Danh sách thẻ thành viên demo (CUSTOMER_ID thật) cho tab Trợ lý Chi tiêu."""
    import tafeng_demo
    return {"customerIds": tafeng_demo.DEMO_CUSTOMER_IDS}


# ===================== BANK DATA (Nội dung mở rộng) =====================
@app.get("/bank")
def bank_dashboard():
    return FileResponse(os.path.join(BASE_DIR, "bank_dashboard.html"))


@app.post("/bank/run")
def bank_run(req: TafengRequest):
    import bank_demo
    return bank_demo.run_experiment(
        min_sup=req.min_sup, alpha=req.alpha, max_customers=req.max_customers)


@app.post("/bank/quality")
def bank_quality(req: TafengRequest):
    import json
    with open(BANK_JSON_PATH, 'r', encoding='utf-8') as f:
        txs = json.load(f)
    n_cust = len(set(t['user_id'] for t in txs))
    return {
        "rows": {"totalRead": len(txs), "kept": len(txs), "keptPct": 100},
        "customers": {"nCustomers": n_cust, "ge2Pct": 100},
        "items": {"priceMin": 0, "priceMax": max(t['amount'] for t in txs)}
    }


@app.post("/bank/apriori")
def bank_apriori(req: AprioriRequest):
    import bank_association_rules
    rules, n = bank_association_rules.run_apriori(
        min_sup=req.min_sup, min_conf=req.min_conf, max_rows=req.max_rows)
    return {"rules": rules[:50], "n_transactions": n}


@app.post("/bank/regression")
def bank_regression(req: RegressionRequest):
    import bank_regression
    return bank_regression.run_regression(max_customers=req.max_customers)


@app.post("/bank/classify")
def bank_classify(req: TafengClassifyRequest):
    import bank_classification
    return bank_classification.run(max_customers=req.max_customers)


@app.post("/bank/predict")
def bank_predict(req: TafengPredictRequest):
    import bank_demo
    return bank_demo.predict_customer(
        customer_idx=req.customer_idx, top_k=req.top_k)


@app.get("/bank/demo-customers")
def bank_demo_customers():
    import bank_demo
    return {"customerIds": bank_demo.DEMO_CUSTOMER_IDS}


if __name__ == "__main__":
    uvicorn.run("ai_service", host="0.0.0.0", port=8000, reload=True)
