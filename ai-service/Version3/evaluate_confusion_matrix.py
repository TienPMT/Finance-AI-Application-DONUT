import os
import json
import torch
import re
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel
from difflib import SequenceMatcher
from tqdm import tqdm
from model_config import MODEL_PATH, DEVICE, VAL_METADATA, VAL_IMG_DIR

RELAXED_THRESHOLD = 0.80

def compute_similarity(gt, pred):
    """Tính độ tương đồng giữa 2 chuỗi (từ 0.0 đến 1.0)"""
    gt_str = str(gt).strip().lower()
    pred_str = str(pred).strip().lower()
    if not gt_str and not pred_str:
        return 1.0
    return SequenceMatcher(None, gt_str, pred_str).ratio()

def extract_money_amount(text):
    """Trích xuất giá trị số từ chuỗi tiền tệ"""
    matches = re.findall(r'[\d.,]+', str(text))
    if not matches:
        return str(text).strip()
    max_val = -1
    best_str = str(text).strip()
    for m in matches:
        m = m.rstrip('.,')
        if m.endswith('.00') or m.endswith(',00'):
            m = m[:-3]
        clean_num = re.sub(r'[.,]', '', m)
        if clean_num.isdigit():
            val = int(clean_num)
            if val > max_val:
                max_val = val
                best_str = clean_num
    return best_str

def run_confusion_matrix_eval():
    print(f"\n{'='*70}")
    print("🚀 ĐANG TÍNH TOÁN MA TRẬN NHẦM LẪN (TP, FP, FN) & F1-SCORE...")
    print(f"{'='*70}")

    if not os.path.exists(MODEL_PATH):
        print(f"[LỖI] Chưa có model tại {MODEL_PATH}.")
        return

    processor = DonutProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    with open(VAL_METADATA, "r", encoding="utf-8") as f:
        val_samples = [json.loads(line) for line in f]

    # Khởi tạo ma trận đếm TP, FP, FN cho từng trường
    confusion_matrix = {
        "seller": {"TP": 0, "FP": 0, "FN": 0},
        "address": {"TP": 0, "FP": 0, "FN": 0},
        "timestamp": {"TP": 0, "FP": 0, "FN": 0},
        "total_cost": {"TP": 0, "FP": 0, "FN": 0}
    }

    start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_seller>")

    for sample in tqdm(val_samples, desc="Đang phân tích hóa đơn"):
        img_path = os.path.join(VAL_IMG_DIR, sample["file_name"])
        if not os.path.exists(img_path): continue

        image = Image.open(img_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt", do_resize=True,
                                 size={"height": 1280, "width": 960}, # Điều chỉnh nếu cần
                                 do_align_long_axis=False).pixel_values.to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values, max_length=768,
                decoder_input_ids=torch.tensor([[start_token_id]]).to(DEVICE),
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True
            )

        sequence = processor.batch_decode(outputs)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = re.sub(r"<pad>", "", sequence)

        try:
            pred_dict = processor.token2json(sequence)
            if "seller" not in pred_dict: pred_dict = {"seller": pred_dict}
        except Exception:
            pred_dict = {"seller": {}}

        preds = pred_dict.get("seller", {})
        if not isinstance(preds, dict): preds = {}

        gt_data = json.loads(sample["ground_truth"]).get("gt_parse", {})

        field_mapping = [
            ("seller", ["SELLER", "seller", "seller_name"]),
            ("address", ["ADDRESS", "address"]),
            ("timestamp", ["TIMESTAMP", "timestamp"]),
            ("total_cost", ["TOTAL_COST", "total_cost"])
        ]

        for metric_key, gt_keys in field_mapping:
            gt_val, pred_val = "", ""
            
            # Lấy Ground Truth
            for k in gt_keys:
                if k in gt_data:
                    gt_val = str(gt_data[k]).strip()
                    break
            
            # Lấy Prediction
            for k in gt_keys:
                if k in preds:
                    pred_val = str(preds[k]).strip()
                    break

            has_gt = bool(gt_val)
            has_pred = bool(pred_val)

            # TIẾN HÀNH CHẤM ĐIỂM (TP, FP, FN)
            if has_gt and has_pred:
                # Xử lý riêng trường tiền tệ
                if metric_key == "total_cost":
                    gt_val_clean = extract_money_amount(gt_val)
                    pred_val_clean = extract_money_amount(pred_val)
                else:
                    gt_val_clean = gt_val
                    pred_val_clean = pred_val

                sim_score = compute_similarity(gt_val_clean, pred_val_clean)
                
                # Trúng đích (Đoán CÓ và ĐÚNG)
                if sim_score >= RELAXED_THRESHOLD:
                    confusion_matrix[metric_key]["TP"] += 1
                else:
                    # Đoán sai (Đoán CÓ nhưng SAI) -> Vừa tạo ra 1 rác (FP), vừa lỡ mất 1 gốc (FN)
                    confusion_matrix[metric_key]["FP"] += 1
                    confusion_matrix[metric_key]["FN"] += 1
                    
            elif has_gt and not has_pred:
                # Bỏ sót (Gốc CÓ nhưng AI KHÔNG thèm đọc)
                confusion_matrix[metric_key]["FN"] += 1
                
            elif not has_gt and has_pred:
                # Bịa đặt (Gốc KHÔNG CÓ nhưng AI tự vẽ ra)
                confusion_matrix[metric_key]["FP"] += 1


    # ==========================================
    # IN BÁO CÁO TỔNG KẾT
    # ==========================================
    print("\n\n" + "="*85)
    print(f"{'BẢNG ĐÁNH GIÁ MA TRẬN NHẦM LẪN (CONFUSION MATRIX - ENTITY LEVEL)':^85}")
    print("="*85)
    print(f"{'Trường Dữ Liệu':<15} | {'TP':<5} | {'FP':<5} | {'FN':<5} || {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 85)

    sum_tp, sum_fp, sum_fn = 0, 0, 0

    for key, matrix in confusion_matrix.items():
        tp = matrix["TP"]
        fp = matrix["FP"]
        fn = matrix["FN"]

        sum_tp += tp
        sum_fp += fp
        sum_fn += fn

        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{key.upper():<15} | {tp:<5} | {fp:<5} | {fn:<5} || {precision:>9.2f}% | {recall:>9.2f}% | {f1_score:>9.2f}%")

    print("-" * 85)
    
    overall_precision = (sum_tp / (sum_tp + sum_fp)) * 100 if (sum_tp + sum_fp) > 0 else 0
    overall_recall = (sum_tp / (sum_tp + sum_fn)) * 100 if (sum_tp + sum_fn) > 0 else 0
    overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    print(f"{'TỔNG CỘNG':<15} | {sum_tp:<5} | {sum_fp:<5} | {sum_fn:<5} || {overall_precision:>9.2f}% | {overall_recall:>9.2f}% | {overall_f1:>9.2f}%")
    print("="*85)

if __name__ == "__main__":
    run_confusion_matrix_eval()