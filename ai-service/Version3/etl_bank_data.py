import pandas as pd
import json
import re
import time

# ==========================================
# 1. CẤU HÌNH FILE ĐẦU VÀO / ĐẦU RA
# ==========================================
INPUT_EXCEL = 'bank.xlsx'
OUTPUT_JSON = 'transactions_bank_clean.json'

def categorize_transaction(detail):
    """
    Hệ chuyên gia (Rule-based) quét text thô để gắn Tag phân loại chi tiêu.
    Bộ tag khớp 100% với danh sách ESSENTIAL và DISCRETIONARY trong budget_advisor.py
    """
    detail = str(detail).upper()
    
    # Bỏ qua các giao dịch chuyển tiền nội bộ, rút tiền mặt, trả lãi (Nhiễu)
    if re.search(r'INTERNAL FUND|TRF|NEFT|RTGS|ATM|CASH WITHDRAWAL|INTEREST|FDRL', detail):
        return 'IGNORE'
        
    # Nhóm ESSENTIAL (Thiết yếu)
    if re.search(r'ZOMATO|SWIGGY|MCDONALDS|KFC|FOOD|RESTAURANT|DOMINOS|BAKERY', detail):
        return 'an uong'
    elif re.search(r'UBER|OLA|IRCTC|TICKET|RAILWAY|METRO|AIRLINES|TRANSPORT', detail):
        return 'di lai'
    elif re.search(r'HOSPITAL|PHARMACY|CLINIC|HEALTH|MEDICAL|DRUG', detail):
        return 'y te'
    elif re.search(r'BILL|RECHARGE|ELECTRICITY|TELECOM|AIRTEL|VODAFONE|WATER', detail):
        return 'hoa don'
    elif re.search(r'SCHOOL|COLLEGE|UNIVERSITY|FEE|EDUCATION', detail):
        return 'giao duc'
        
    # Nhóm DISCRETIONARY (Tùy ý)
    elif re.search(r'AMAZON|FLIPKART|SHOP|POS|RETAIL|MART|STORE|MALL', detail):
        return 'mua sam'
    elif re.search(r'MOVIE|CINEMA|NETFLIX|PVR|ENTERTAINMENT|GAME', detail):
        return 'giai tri'
    elif re.search(r'HOTEL|RESORT|MAKEMYTRIP|OYO|TRAVEL', detail):
        return 'du lich'
    elif re.search(r'COFFEE|STARBUCKS|CAFE', detail):
        return 'ca phe'
        
    # Nếu không nhận diện được rõ ràng nhưng quẹt thẻ POS thì gán mặc định là mua sắm
    elif 'POS' in detail:
        return 'mua sam'
        
    # Những cái còn lại không phân loại được thì gán là Khác
    return 'khac'

def run_etl_pipeline():
    print(" BẮT ĐẦU CHẠY PIPELINE ETL (TIỀN XỬ LÝ DỮ LIỆU)...")
    start_time = time.time()
    
    # 1. ĐỌC DỮ LIỆU TỪ EXCEL
    print(f"[*] Đang nạp dữ liệu từ {INPUT_EXCEL} (Sẽ mất khoảng 5-10 giây)...")
    try:
        df = pd.read_excel(INPUT_EXCEL)
    except Exception as e:
        print(f"[!] Lỗi đọc file: {e}")
        return
        
    total_raw = len(df)
    print(f"[*] Đã đọc thành công {total_raw:,} dòng dữ liệu thô.")

    # 2. LÀM SẠCH VÀ CHUẨN HÓA (TRANSFORM)
    print("[*] Đang tiến hành làm sạch và phân loại bằng Regex...")
    
    # Giữ lại các cột quan trọng theo tên cột trong hình em chụp
    # Account No, DATE, TRANSACTION DETAILS, WITHDRAWAL AMT
    df = df[['Account No', 'DATE', 'TRANSACTION DETAILS', 'WITHDRAWAL AMT']]
    
    # Lọc bỏ các dòng không có số tiền rút (Nghĩa là không phải chi tiêu)
    df = df.dropna(subset=['WITHDRAWAL AMT'])
    df = df[df['WITHDRAWAL AMT'] > 0]
    
    # Chuẩn hóa ngày tháng
    df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%Y-%m-%d')
    
    # Áp dụng hàm phân loại gắn Tag
    df['category'] = df['TRANSACTION DETAILS'].apply(categorize_transaction)
    
    # Lọc bỏ các dòng bị đánh dấu IGNORE (Giao dịch nhiễu) hoặc KHÁC
    df = df[~df['category'].isin(['IGNORE', 'khac'])]
    
    # 3. CHUYỂN ĐỔI SANG JSON FORMAT
    print("[*] Đang định dạng dữ liệu cho AI Cố Vấn (WCM-SPADE)...")
    transactions_list = []
    
    # Tỷ giá Rupee Ấn (INR) sang VNĐ (Nhân với 300 cho số tiền hợp lý với VN)
    INR_TO_VND = 300 
    
    for _, row in df.iterrows():
        transaction = {
            "user_id": str(row['Account No']).strip().replace("'", ""),
            "date": row['DATE'],
            "amount": float(row['WITHDRAWAL AMT']) * INR_TO_VND,
            "category": row['category'],
            "isExpense": True
        }
        transactions_list.append(transaction)

    # 4. LƯU KẾT QUẢ (LOAD)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(transactions_list, f, ensure_ascii=False, indent=2)

    end_time = time.time()
    print("\n" + "="*60)
    print(" HOÀN TẤT QUÁ TRÌNH ETL")
    print("="*60)
    print(f"Tổng số giao dịch thô: {total_raw:,} dòng")
    print(f"Số giao dịch tiêu dùng HỢP LỆ giữ lại: {len(transactions_list):,} dòng")
    print(f"File đầu ra: {OUTPUT_JSON}")
    print(f"Thời gian chạy: {round(end_time - start_time, 2)} giây")
    print("="*60)
    print("-> Data đã sạch sẽ 100%, sẵn sàng nạp vào WCM-SPADE để khai phá!")

if __name__ == "__main__":
    run_etl_pipeline()