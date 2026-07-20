import json
from budget_advisor import advise

def test_ai_advisor():
    print("\n" + "="*70)
    print(" KHỞI CHẠY HỆ THỐNG AI CỐ VẤN TÀI CHÍNH (WCM-SPADE)")
    print("="*70)

    # 1. Nạp dữ liệu đã làm sạch
    print("[*] Đang nạp 4.604 giao dịch từ tập dữ liệu Ngân hàng...")
    try:
        with open('transactions_bank_clean.json', 'r', encoding='utf-8') as f:
            transactions = json.load(f)
    except FileNotFoundError:
        print("[!] Không tìm thấy file transactions_bank_clean.json")
        return

    # 2. Cấu hình bài toán giả định cho AI
    muc_tieu_tiet_kiem = 5000000  # Muốn tiết kiệm 5 triệu
    thu_nhap_thang = 25000000     # Lương 25 triệu
    
    print(f"[*] Mục tiêu: Tiết kiệm {muc_tieu_tiet_kiem:,.0f} VNĐ/tháng")
    print(f"[*] Thu nhập: {thu_nhap_thang:,.0f} VNĐ/tháng")
    print("[*] Thuật toán WCM-SPADE đang quét chuỗi thời gian... Vui lòng đợi!\n")

    # Gọi hàm advise từ code của nhóm em
    result = advise(
        transactions=transactions,
        target=muc_tieu_tiet_kiem,
        current=0,
        months_left=1,
        monthly_income=thu_nhap_thang
    )

    if "error" in result:
        print(f"Lỗi từ AI: {result['error']}")
        return

    # ==========================================
    # 3. IN BÁO CÁO KẾT QUẢ ĐẸP MẮT
    # ==========================================
    print("="*70)
    print("🌟 1. CÁC QUY LUẬT CHI TIÊU KHAI PHÁ ĐƯỢC (GLOBAL RULES)")
    print("="*70)
    if result['trends']:
        for i, trend in enumerate(result['trends'], 1):
            print(f"Luật {i}: {trend['pattern']}")
            print(f"   -> Số lần lặp (Support): {trend['count']} | Trọng số (W-Sup): {trend['wsupport']}\n")
    else:
        print("Không tìm thấy quy luật nào đủ mạnh với min_sup hiện tại.")

    print("="*70)
    print(" 2. ĐÁNH GIÁ NGÂN SÁCH & LỜI KHUYÊN CẮT GIẢM (PRESCRIPTIVE AI)")
    print("="*70)
    print(f" TỔNG QUAN: {result['summary']}\n")
    
    for rec in result['recommendations']:
        print(f"🔹 DANH MỤC: {rec['tag'].upper()}")
        print(f"   - Hành động: {rec['action'].upper()} {abs(rec['delta']):,.0f} VNĐ")
        print(f"   - Lý do AI đưa ra: {rec['reason']}")
        print(f"   - Hạn mức đề xuất mới: {rec['suggested']:,.0f} VNĐ\n")
        
    print("="*70)
    print(" Đã chạy xong! Sẵn sàng tích hợp API cho Flutter.")
    print("="*70)

if __name__ == "__main__":
    test_ai_advisor()