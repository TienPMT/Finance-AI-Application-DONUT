"""
Gán nhãn dễ đọc cho mã item Ta-Feng.

LƯU Ý: CSV Ta-Feng KHÔNG chứa tên sản phẩm (chỉ có mã PRODUCT_SUBCLASS / PRODUCT_ID
là số / mã vạch EAN). Vì vậy:
  - `name`  = ALIAS tự sinh, CỐ ĐỊNH theo mã (deterministic) — chỉ để demo dễ đọc,
              KHÔNG phải tên thật của sản phẩm.
  - `price` = giá TB SALES_PRICE thật từ CSV (Đài tệ TWD) × tỉ giá cố định
              TWD_TO_VND để hiển thị theo VND. Gốc vẫn là số liệu thật.
  - `count` = số lượt mua thật từ CSV.

Alias = danh từ siêu thị + mô tả, chọn theo hash của mã → cùng mã luôn ra cùng tên.
Mã gốc luôn đi kèm để không gây hiểu nhầm.
"""

import hashlib
TWD_TO_VND = 800

_GROUPS = [

    (["Táo", "Chuối", "Cam", "Nho", "Dưa hấu", "Xoài", "Lê", "Dâu", "Cà chua", "Khoai tây",
      "Cà rốt", "Hành tây", "Bắp cải", "Rau cải", "Nấm", "Gừng", "Tỏi", "Ớt", "Dưa leo", "Bí đỏ"],
     ["", "loại 1", "tươi", "nhập khẩu", "khuyến mãi"]),

    (["Gạo", "Mì gói", "Bún khô", "Phở khô", "Bột mì", "Đường", "Muối", "Nước mắm", "Nước tương", "Dầu ăn"],
     ["", "loại 1", "túi lớn", "khuyến mãi", "gia đình"]),

    (["Trứng gà", "Sữa tươi", "Sữa chua", "Phô mai", "Bơ", "Kem", "Sữa đặc", "Sữa bột", "Đậu hũ", "Đậu phộng"],
     ["", "hộp", "loại 1", "nhập khẩu", "khuyến mãi"]),

    (["Thịt heo", "Thịt bò", "Thịt gà", "Cá thu", "Cá hồi", "Tôm", "Mực", "Xúc xích", "Giò chả", "Cá viên"],
     ["", "tươi", "đông lạnh", "loại 1", "khuyến mãi"]),

    (["Bánh mì", "Bánh quy", "Bánh gạo", "Snack khoai tây", "Sô cô la", "Kẹo", "Mứt", "Ngũ cốc", "Yến mạch", "Hạt điều"],
     ["", "hộp", "túi lớn", "nhập khẩu", "khuyến mãi"]),

    (["Nước suối", "Nước ngọt", "Trà xanh", "Cà phê", "Nước cam", "Bia", "Rượu vang", "Nước tăng lực", "Sữa đậu nành", "Nước dừa"],
     ["", "nhập khẩu", "khuyến mãi"]),

    (["Giấy vệ sinh", "Khăn giấy", "Nước rửa chén", "Bột giặt", "Nước xả vải", "Dầu gội", "Sữa tắm", "Kem đánh răng", "Bàn chải", "Xà phòng"],
     ["", "túi lớn", "gia đình", "khuyến mãi", "cao cấp"]),

    (["Tã em bé", "Khăn ướt", "Băng vệ sinh", "Pin", "Bóng đèn", "Túi rác", "Màng bọc", "Nước lau sàn", "Nước rửa tay", "Bình xịt côn trùng"],
     ["", "cỡ lớn", "khuyến mãi", "gia đình"]),
]


_FLAT = [(noun, descs) for nouns, descs in _GROUPS for noun in nouns]


def _h(item_id) -> int:
    return int(hashlib.md5(str(item_id).encode()).hexdigest(), 16)


def alias(item_id) -> str:
    h = _h(item_id)
    noun, descs = _FLAT[h % len(_FLAT)]
    desc = descs[(h // len(_FLAT)) % len(descs)]
    return f"{noun} {desc}".strip()


def build_labels(item_ids, item_prices=None, item_counts=None) -> dict:
    """Trả map str(id) -> {name(alias), code, price(VND quy đổi), count(thật)}."""
    out = {}
    for it in item_ids:
        key = str(it)
        out[key] = {
            "name": alias(it),
            "code": key,
            "price": round(item_prices.get(it, 0.0) * TWD_TO_VND) if item_prices else None,
            "count": int(item_counts.get(it, 0)) if item_counts else None,
        }
    return out
