import os
import json
import torch
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel, AutoTokenizer
import re

from model_config import MODEL_PATH, DATA_DIR, DEVICE, VAL_METADATA, VAL_IMG_DIR, BASE_DIR, IMAGE_HEIGHT, IMAGE_WIDTH

IMAGE_DIR = os.path.join(DATA_DIR, "real2")

IMAGE_NAME = "z7879189394720_f82181f04a4d8b952a26d66f1a0a331a.jpg"

def run_prediction(model, processor, image):
    """Hàm xử lý ảnh và bóc tách thông tin qua mô hình Donut"""
    pixel_values = processor(
        image,
        return_tensors="pt",
        do_resize=True,
        size={"height": IMAGE_HEIGHT, "width": IMAGE_WIDTH},
        do_align_long_axis=False
    ).pixel_values.to(DEVICE)

    start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_seller>")

    with torch.no_grad():
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

    vowels = "áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵĐđ"
    vowels += vowels.upper()
    tails = ["c", "ch", "m", "n", "ng", "nh", "p", "t", "i", "y", "a", "o", "u"]
    tails += [t.upper() for t in tails]
    tail_pattern = "|".join(tails)

    pattern = r"([" + vowels + r"])\s+(" + tail_pattern + r")\b"

    old_sequence = ""
    while old_sequence != sequence:
        old_sequence = sequence
        sequence = re.sub(pattern, r"\1\2", sequence)
    try:
        prediction_json = processor.token2json(sequence)
        if "seller" not in prediction_json:
            prediction_json = {"seller": prediction_json}
    except Exception:
        prediction_json = {"seller": {}}

    return prediction_json, sequence

def main():
    print("\n" + "="*60)
    print(" TRÍCH XUẤT THÔNG TIN HÓA ĐƠN ĐƠN LẺ (DONUT V4)")
    print("="*60)

    image_path = os.path.join(IMAGE_DIR, IMAGE_NAME)
    if not os.path.exists(image_path):
        print(f"\n[LỖI] Không tìm thấy ảnh: {image_path}")
        print("Vui lòng kiểm tra lại biến IMAGE_DIR và IMAGE_NAME.")
        return

    print(f"\n Đang tải mô hình từ: {MODEL_PATH}...")
    try:
        processor = DonutProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    except OSError:
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
        try:
            processor.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        except OSError:
            pass

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    print(f" Đã tải mô hình xong (Sử dụng {DEVICE.upper()}).")
    print(f"\n Đang phân tích ảnh: {IMAGE_NAME}...")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[LỖI] Không thể đọc được file ảnh: {e}")
        return

    print("\n" + "="*60)
    print("KẾT QUẢ TRÍCH XUẤT (JSON)")
    print("="*60)

    print(json.dumps(prediction_json, indent=4, ensure_ascii=False))

    print("\n" + "="*60)
    print("CHUỖI VĂN BẢN THÔ MÔ HÌNH SINH RA (RAW SEQUENCE)")
    print("="*60)
    print(raw_sequence)
    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()
