import os
import json
import random
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont

NUM_IMAGES_TO_GENERATE = 4500
OUTPUT_DIR = "data1/synthetic_train"
OUTPUT_JSONL = "data1/synthetic_metadata.jsonl"

os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_VAULT = {
   "CAFE_MILKTEA": {
        "sellers": ["Highlands Coffee", "Phúc Long Tea", "The Coffee House", "Katinat", "GongCha", "Trung Nguyên Legend", "Cheese Coffee", "Cộng Cà Phê", "Tocotoco", "Trà Sữa KOI Thé"],
        "items": ["Cà phê sữa đá", "Trà đào cam sả", "Trà sữa trân châu", "Trà sen vàng", "Bạc xỉu", "Cà phê đen", "Trà đen Macchiato", "Sữa tươi trân châu đường đen", "Trà vải nhiệt đới", "Bánh flan"],
        "price_range": (25000, 85000)
    },
    "FAST_FOOD": {
        "sellers": ["KFC Vietnam", "Lotteria", "Texas Chicken", "McDonald's", "Jollibee", "Domino's Pizza", "Pizza Hut"],
        "items": ["Gà rán sốt cay", "Combo Gà Rán", "Gà giòn không xương", "Khoai tây chiên (L)", "Burger Bò", "Burger Tôm", "Cơm gà quay tiêu", "Pizza Hải Sản (M)", "Pizza phô mai (S)", "Nước ngọt Coca Cola"],
        "price_range": (35000, 250000)
    },
    "RESTAURANT_VN": {
        "sellers": ["Cơm Tấm Sà Bì Chưởng", "Phở 24", "Bún Bò Huế 32", "Bò Né 3 Ngon", "Nhà hàng Quán Bụi", "Cơm Tấm Cali", "Bún chả Hà Nội Oanh"],
        "items": ["Cơm sườn bì chả", "Bánh mì thịt nướng", "Phở bò tái nạm", "Phở gà đặc biệt", "Cơm chiên dương châu", "Bò bít tết trứng", "Bún đậu mắm nôm", "Bún chả Hà Nội", "Bún bò huế", "Trà đá"],
        "price_range": (35000, 120000)
    },
    "SUPERMARKET": {
        "sellers": ["BÁCH HÓA XANH", "CoopMart", "Siêu thị WinMart", "Cửa hàng tiện lợi GS25", "GO! Mall", "Lotte Mart", "Aeon Mall", "Emart", "MM Mega Market"],
        "items": [
            "snack khoai tây lay's wavy vị thăn bò nướng texas gói 56g", "snack poca/lay's khoai tây tự nhiên classic gói 52g",
            "đậu phộng oishi vị mực cay 85g", "Sữa TH True Milk ít đường 180ml lốc 4", "Mì Hảo Hảo chua cay",
            "Thịt heo xay 500g", "Bắp cải thảo 1kg", "Nước giặt OMO Matic 3.2kg", "Dầu ăn Tường An 1L",
            "Nước mắm Chinsu cá hồi 500ml", "Tương ớt Nam Dương 250ml", "Đường tinh luyện Biên Hòa 1kg",
            "Muối i ốt Bạc Liêu 500g", "Bột ngọt Ajinomoto 400g", "Hạt nêm Knorr 400g", "Nước rửa chén Sunlight chanh 1.5kg",
            "Sữa tươi Vinamilk có đường 1L", "Sữa đặc Ông Thọ đỏ", "Trứng gà Ba Huân vỉ 10", "Xúc xích Vissan 5 cây",
            "Thịt bò Úc nhập khẩu 300g", "Cá hồi phi lê 200g", "Táo Mỹ nhập khẩu 1kg", "Nho xanh không hạt 500g",
            "Chuối già Nam Mỹ 1kg", "Cam sành Vĩnh Long 1kg", "Xoài keo 1kg", "Rau muống Đà Lạt 500g",
            "Cà chua VietGAP 500g", "Hành tây 500g", "Tỏi khô 200g", "Hành tím 200g", "Gạo dẻo ST25 5kg",
            "Gạo thơm Thái 5kg", "Mì Omachi sườn hầm ngũ quả", "Phở bò Cung Đình", "Miến Phú Hương",
            "Bún gạo Vifon", "Nước khoáng Aquafina 1.5L", "Nước ngọt Sprite 1.5L", "Bia Tiger lon 330ml",
            "Bia Heineken lon 330ml", "Nước ép cam Vfresh 1L", "Trà xanh Không Độ 500ml", "Nước tăng lực Redbull",
            "Bánh quy Cosy", "Bánh Choco-pie 12 cái", "Bánh gạo One One", "Kẹo Alpenliebe", "Sữa chua Vinamilk nha đam lốc 4",
            "Kem Wall's dâu", "Phô mai con bò cười", "Bơ Tường An 200g", "Sữa rửa mặt Ponds 100g",
            "Dầu gội Clear thảo dược 650g", "Sữa tắm Lifebuoy 850g", "Kem đánh răng P/S trà xanh 240g",
            "Bàn chải đánh răng Colgate", "Khăn giấy Tempo 3 lớp", "Giấy vệ sinh Bless You lốc 10",
            "Băng vệ sinh Diana siêu thấm", "Bột giặt Surf 5kg", "Nước xả vải Downy hương nắng mai 1.5L",
            "Nước lau sàn Vim 1L", "Nước tẩy bồn cầu Duck 900ml", "Túi đựng rác tự hủy sinh học",
            "Màng bọc thực phẩm", "Khăn lau bếp", "Bọt biển rửa chén", "Găng tay cao su", "Nước súc miệng Listerine 500ml",
            "Xà bông cục Lifebuoy", "Lăn khử mùi X-men", "Bọt cạo râu Nivea", "Dao cạo râu Gillette", "Bông tẩy trang Silcot 82 miếng",
            "Nước tẩy trang L'Oreal 400ml", "Sữa rửa mặt Senka", "Mặt nạ giấy Vedette", "Kem chống nắng Biore",
            "Kem dưỡng da Vaseline", "Sữa dưỡng thể Hada Labo", "Dầu xả Pantene 320g", "Sữa tắm nam Romano",
            "Nước hoa hồng Simple", "Khăn ướt Baby", "Sữa bột Friso", "Sữa bột GrowPlus", "Bột ăn dặm Ridielac",
            "Tã quần Bobby", "Tã dán Huggies", "Phấn rôm Johnson Baby", "Sữa tắm gội Pigeon", "Kem chống hăm Bubchen",
            "Giấy bạc bọc thực phẩm", "Hộp nhựa đựng thực phẩm", "Móc phơi quần áo", "Kẹp quần áo", "Nhang muỗi Jumbo"
        ],
        "price_range": (5000, 500000)
    },
    "PHARMACY_1": {
        "sellers": ["Pharmacity", "NHÀ THUỐC LONG CHÂU", "NHÀ THUỐC AN KHANG", "Watsons", "Guardian"],
        "items": [
            "JOMI BÔNG TẨY TRANG 120M", "Túi 2", "Khẩu trang y tế 4 lớp Niva", "Băng cá nhân Urgo trong suốt",
            "Vitamin C sủi MyVita", "Viên sủi Plusssz cam", "Nước súc miệng Listerine", "Kem đánh răng Sensodyne",
            "Bao cao su Durex Performa", "Gel bôi trơn Durex Play", "Băng vệ sinh Laurier", "Sữa rửa mặt Cetaphil 125ml",
            "Nước muối sinh lý Vĩnh Phúc 500ml", "Cồn i-ốt 90 độ 500ml", "Oxy già 60ml", "Thuốc nhỏ mắt V.Rohto Vitamin",
            "Viên ngậm trắng da Neocell", "Viên uống Omega 3 Blackmores", "Viên dầu cá Fish Oil 1000mg",
            "Vitamin E 400 IU Kirkland", "Canxi D3 K2 OSTELIN", "Kẹo ngậm ho Bảo Thanh", "Kẹo ngậm Strepsils",
            "Cao xoa bóp Bạch Hổ", "Dầu gió xanh Thiên Ân", "Dầu tràm Thái Đan", "Miếng dán hạ sốt Salonpas",
            "Nhiệt kế điện tử Omron", "Bình xịt mũi Xisat", "Siro ho Astex 90ml", "Men vi sinh BioGaia",
            "Trà thảo mộc Boganic", "Thực phẩm chức năng Hoạt Huyết Dưỡng Não", "Sâm Alipas", "Viên uống trắng da DHC",
            "Nước rửa tay diệt khuẩn", "Khăn ướt cồn", "Găng tay y tế Vglove", "Bơm tiêm nhựa 5ml", "Gạc y tế tiệt trùng",
            "Băng keo cá nhân cuộn", "Băng thun y tế", "Sữa Ensure Gold 400g", "Sữa Glucerna 850g", "Sữa Prosure",
            "Trà Atiso túi lọc", "Trà hoa cúc mật ong", "Kẹo dẻo Vitamin C", "Kẹo ngậm trị ho Prospan", "Dung dịch vệ sinh phụ nữ Dạ Hương",
            "Kem bôi muỗi đốt Remos", "Xịt chống muỗi Soffell", "Dầu gội trị gàu Selsun", "Sữa tắm Cetaphil 500ml"
        ],
        "price_range": (10000, 800000)
    },
    "PHARMACY_2": {
        "sellers": ["NHÀ THUỐC NGỌC CHÂU 2", "NHÀ THUỐC MINH CHÂU", "PHÒNG KHÁM ĐA KHOA KỲ HÒA", "Nhà thuốc Phước Thiện", "Nhà thuốc số 1"],
        "items": [
            "Refresh Tears C/15ml (Allergan)", "Agicardi 2.5mg H, 30v (Agimex)",
            "Men tiêu hóa Enterogermina", "Siro trị ho Prospan 100ml", "Thuốc Panadol Extra đỏ",
            "Panadol Cảm Cúm 10 viên", "Decolgen Forte vỉ 10", "Tiffy vỉ 4 viên", "Berberin 100mg lọ 100v",
            "Thuốc ho Bổ Phế Nam Hà", "Smecta ống uống", "Oresol gói bù nước", "Paracetamol 500mg vỉ 10",
            "Efferax sủi 500mg", "Amoxicillin 500mg vỉ 10", "Augmentin 500mg", "Cephalexin 500mg",
            "Thuốc dạ dày Phosphalugel", "Gaviscon gói", "Omeprazol 20mg vỉ", "Loratadin 500mg vỉ 10",
            "Cetirizin 10mg vỉ 10", "Cinarizin 25mg vỉ", "Diazepam 25mg vỉ", "Aspirin 500mg vỉ",
            "Diclofenac 50mg vỉ", "Meloxicam 50mg vỉ", "Vitamin B1 B6 B12 vỉ", "Magnesium B6 vỉ",
            "Kẽm Corbiere ống", "Sắt Zinc gluconat", "Viên uống bổ gan", "Thuốc nhỏ mắt Osla",
            "Nước biển sâu xịt mũi", "Betadine sát trùng", "Cồn 70 độ 60ml", "Oxy già 3%",
            "Povidine 10%", "Thuốc bôi ngoài da Gentri-s", "Thuốc mỡ Fucidin", "Thuốc bôi trị bỏng Panthenol",
            "Thuốc trị mụn Derma", "Kem trị sẹo Acnes", "Thuốc bôi trĩ", "Thuốc tẩy giun Fugacar",
            "Thuốc tẩy giun Zentel", "Viên sủi Halixol", "Thuốc ho Mucosolvan", "Thuốc giãn phế quản Ventolin"
        ],
        "price_range": (5000, 300000)
    },
    "CIRCLE_K": {
        "sellers": ["CIRCLE K"],
        "items": [
            "PhaPhin Sua Da M/1 Ly", "Banh Bao Xa Xiu & Trung 190g", "Mi Hao Hao Chua Cay 75g",
            "Nuoc Suoi Aquafina 500ml", "Redbull 250ml", "Banh Mi Thit Nuong", "Snack Lays Wavy",
            "Keo Singum Coolair", "Xuc Xich Ponnie Quay", "Xuc Xich CP 5 cay", "Bia Tiger Lon",
            "Bia Heineken Lon", "Sua tuoi Vinamilk", "Tra Xanh Khong Do", "Sting Dau", "Coca Cola Lon",
            "Kem Merino", "Kem Celano", "Com nam Kimbap", "Sandwich thit nguoi", "Banh bao nhan thit",
            "Hotdog pho mai", "Ca phe den da M", "Tra sua Thai Lan L", "Tra dao cam sa L", "Nuoc ep trai cay",
            "Sua chua uong Probi", "Keo deo Chupa Chups", "Socola Snickers", "Keo Mentos", "Mi ly Omachi"
        ],
        "price_range": (8000, 80000)
    },
    "MINISO": {
        "sellers": ["MINISO", "MUMUSO", "DAISO"],
        "items": [
            "Bong trang diem Barbie Collectio", "Tui Dung Hang MINISO (Nho)",
            "Khan lau Pineapple(Blue)", "Gau bong hinh chim canh cut", "Ly su uong nuoc co nap",
            "Mockhoa hinh Marvel", "Hop dung viet chi", "But bi nuoc xanh", "Guong cam tay Mini",
            "Luoc chai dau go", "Sua tam Miniso rose", "Nuoc hoa xit phong", "Tui xach vai Canvas",
            "Binh dung nuoc thuy tinh", "Lo hoa de ban", "Dong ho bao thuc de thuong", "Hop dung my pham",
            "Cot toc hoa tiet", "Bang do rua mat", "Khan tam soi cotton", "Tham lau chan cute",
            "Goi ke co chu U", "Mat na ngu ban dem", "Gel rua tay kho", "Khuyen tai nu de thuong",
            "Bop dung tien xu", "Day sac dien thoai", "Tai nghe nhet tai", "Bao da dung ho chieu"
        ],
        "price_range": (20000, 350000)
    }
}

ADDRESSES = [
    "270 Lý Thái Tổ, Phường Bản Cờ, Thành phố Hồ Chí Minh",
    "177 NGUYỄN THIỆN THUẬT, Phường 1, Quận 3, TP.HCM",
    "123 Nguyễn Đình Chiểu, Phường 6, Quận 3, TP.HCM",
    "45 Lê Lợi, Phường Bến Nghé, Quận 1, TP.HCM",
    "89 Võ Văn Ngân, Phường Linh Chiểu, TP Thủ Đức",
    "12 Phan Văn Trị, Phường 7, Quận Gò Vấp, HCM",
    "Tầng 1, Vincom Thảo Điền, Quận 2, TP Thủ Đức",
    "Số 5 Nguyễn Huệ, Phường Bến Nghé, Quận 1",
    "124 Tôn Đức Thắng, Quận Đống Đa, Hà Nội",
    "35 Mạc Đĩnh Chi, Phường Bến Nghé, Quận 1",
    "56 Hai Bà Trưng, Phường Bến Nghé, Quận 1, TP.HCM",
    "789 Điện Biên Phủ, Phường 2, Quận 3, TP.HCM",
    "234 Trần Hưng Đạo, Phường Nguyễn Cư Trinh, Quận 1, TP.HCM",
    "111 Pasteur, Phường 6, Quận 3, TP.HCM",
    "222 Lê Duẩn, Phường Bến Nghé, Quận 1, TP.HCM",
    "333 Nam Kỳ Khởi Nghĩa, Phường 7, Quận 3, TP.HCM",
    "444 Nguyễn Thị Minh Khai, Phường 5, Quận 3, TP.HCM",
    "555 Cách Mạng Tháng Tám, Phường 11, Quận 3, TP.HCM",
    "666 Võ Văn Tần, Phường 5, Quận 3, TP.HCM",
    "777 Bà Huyện Thanh Quan, Phường 7, Quận 3, TP.HCM",
    "888 Trương Định, Phường 6, Quận 3, TP.HCM",
    "999 Tú Xương, Phường 7, Quận 3, TP.HCM",
    "101 Nguyễn Trãi, Phường Bến Thành, Quận 1, TP.HCM",
    "202 Lê Lai, Phường Bến Thành, Quận 1, TP.HCM",
    "303 Lê Thánh Tôn, Phường Bến Nghé, Quận 1, TP.HCM",
    "404 Lý Tự Trọng, Phường Bến Thành, Quận 1, TP.HCM",
    "505 Đồng Khởi, Phường Bến Nghé, Quận 1, TP.HCM",
    "606 Nguyễn Du, Phường Bến Nghé, Quận 1, TP.HCM",
    "707 Tôn Thất Thiệp, Phường Bến Nghé, Quận 1, TP.HCM",
    "808 Huỳnh Thúc Kháng, Phường Bến Nghé, Quận 1, TP.HCM",
    "12A Quang Trung, Phường 10, Quận Gò Vấp, TP.HCM",
    "34B Phan Huy Ích, Phường 5, Quận Gò Vấp, TP.HCM",
    "56C Nguyễn Oanh, Phường 17, Quận Gò Vấp, TP.HCM",
    "78D Lê Văn Thọ, Phường 11, Quận Gò Vấp, TP.HCM",
    "90E Thống Nhất, Phường 16, Quận Gò Vấp, TP.HCM",
    "12F Phạm Văn Chiêu, Phường 14, Quận Gò Vấp, TP.HCM",
    "34G Nguyễn Kiệm, Phường 3, Quận Gò Vấp, TP.HCM",
    "56H Nguyễn Thái Sơn, Phường 3, Quận Gò Vấp, TP.HCM",
    "78J Hoàng Minh Giám, Phường 3, Quận Gò Vấp, TP.HCM",
    "90K Bạch Đằng, Phường 2, Quận Tân Bình, TP.HCM",
    "12L Trường Sơn, Phường 2, Quận Tân Bình, TP.HCM",
    "34M Cộng Hòa, Phường 4, Quận Tân Bình, TP.HCM",
    "56N Hoàng Văn Thụ, Phường 4, Quận Tân Bình, TP.HCM",
    "78P Út Tịch, Phường 4, Quận Tân Bình, TP.HCM",
    "90Q Nguyễn Thái Bình, Phường 4, Quận Tân Bình, TP.HCM",
    "12R Âu Cơ, Phường 14, Quận Tân Bình, TP.HCM",
    "34S Lạc Long Quân, Phường 9, Quận Tân Bình, TP.HCM",
    "56T Lý Thường Kiệt, Phường 7, Quận Tân Bình, TP.HCM",
    "78U Hồng Bàng, Phường 12, Quận 5, TP.HCM",
    "90V An Dương Vương, Phường 8, Quận 5, TP.HCM",
    "12W Nguyễn Tri Phương, Phường 8, Quận 5, TP.HCM",
    "34X Châu Văn Liêm, Phường 14, Quận 5, TP.HCM",
    "56Y Hải Thượng Lãn Ông, Phường 10, Quận 5, TP.HCM",
    "78Z Trần Bình Trọng, Phường 1, Quận 5, TP.HCM",
    "15 Cầu Giấy, Phường Quan Hoa, Quận Cầu Giấy, Hà Nội",
    "28 Xuân Thủy, Phường Dịch Vọng Hậu, Quận Cầu Giấy, Hà Nội",
    "42 Trần Thái Tông, Phường Dịch Vọng Hậu, Quận Cầu Giấy, Hà Nội",
    "99 Nguyễn Phong Sắc, Phường Dịch Vọng Hậu, Quận Cầu Giấy, Hà Nội",
    "105 Tôn Thất Thuyết, Phường Dịch Vọng Hậu, Quận Cầu Giấy, Hà Nội",
    "201 Trung Kính, Phường Yên Hòa, Quận Cầu Giấy, Hà Nội",
    "305 Vũ Phạm Hàm, Phường Yên Hòa, Quận Cầu Giấy, Hà Nội",
    "401 Trần Duy Hưng, Phường Trung Hòa, Quận Cầu Giấy, Hà Nội",
    "502 Nguyễn Chí Thanh, Phường Láng Thượng, Quận Đống Đa, Hà Nội",
    "603 Láng Hạ, Phường Láng Thượng, Quận Đống Đa, Hà Nội",
    "704 Huỳnh Thúc Kháng, Phường Láng Hạ, Quận Đống Đa, Hà Nội",
    "805 Thái Hà, Phường Thành Công, Quận Ba Đình, Hà Nội",
    "906 Giảng Võ, Phường Thành Công, Quận Ba Đình, Hà Nội",
    "11 Láng, Phường Ngã Tư Sở, Quận Đống Đa, Hà Nội",
    "22 Tây Sơn, Phường Trung Liệt, Quận Đống Đa, Hà Nội",
    "33 Chùa Bộc, Phường Trung Liệt, Quận Đống Đa, Hà Nội",
    "44 Phạm Ngọc Thạch, Phường Láng Hạ, Quận Đống Đa, Hà Nội",
    "55 Xã Đàn, Phường Phương Liên, Quận Đống Đa, Hà Nội",
    "66 Tôn Đức Thắng, Phường Cát Linh, Quận Đống Đa, Hà Nội",
    "77 Nguyễn Lương Bằng, Phường Ô Chợ Dừa, Quận Đống Đa, Hà Nội",
    "88 Khâm Thiên, Phường Thổ Quan, Quận Đống Đa, Hà Nội",
    "99 Hai Bà Trưng, Phường Trần Hưng Đạo, Quận Hoàn Kiếm, Hà Nội",
    "110 Lý Thường Kiệt, Phường Trần Hưng Đạo, Quận Hoàn Kiếm, Hà Nội",
    "220 Trần Hưng Đạo, Phường Phan Chu Trinh, Quận Hoàn Kiếm, Hà Nội",
    "330 Bà Triệu, Phường Trần Hưng Đạo, Quận Hoàn Kiếm, Hà Nội",
    "440 Phố Huế, Phường Ngô Thì Nhậm, Quận Hai Bà Trưng, Hà Nội",
    "12 Bạch Đằng, Phường Thạch Thang, Hải Châu, Đà Nẵng",
    "34 Trần Phú, Phường Hải Châu 1, Hải Châu, Đà Nẵng",
    "56 Lê Duẩn, Phường Hải Châu 1, Hải Châu, Đà Nẵng",
    "78 Hùng Vương, Phường Hải Châu 1, Hải Châu, Đà Nẵng",
    "90 Nguyễn Văn Linh, Phường Nam Dương, Hải Châu, Đà Nẵng",
    "12 Hàm Nghi, Phường Vĩnh Trung, Thanh Khê, Đà Nẵng",
    "34 Lý Thái Tổ, Phường Thạc Gián, Thanh Khê, Đà Nẵng",
    "56 Điện Biên Phủ, Phường Chính Gián, Thanh Khê, Đà Nẵng",
    "78 Võ Văn Kiệt, Phường Vĩnh Trung, Thanh Khê, Đà Nẵng",
    "90 Nguyễn Tri Phương, Phường Thạc Gián, Thanh Khê, Đà Nẵng",
    "12 Đường 30/4, Phường Hưng Lợi, Ninh Kiều, Cần Thơ",
    "34 Mậu Thân, Phường Xuân Khánh, Ninh Kiều, Cần Thơ",
    "56 Trần Việt Châu, Phường An Hòa, Ninh Kiều, Cần Thơ",
    "78 Nguyễn Văn Cừ, Phường An Hòa, Ninh Kiều, Cần Thơ",
    "90 Cách Mạng Tháng 8, Phường Cái Khế, Ninh Kiều, Cần Thơ"
]

CASHIERS = [
    "Ngan Le", "Khách Lẻ", "Nguyễn Thị Ngọc", "Trần Hữu Phát", "Lê Thanh Trà",
    "Phạm Trọng", "Vương Mẫn", "Đinh Hằng", "NV: Lê Minh", "Thu Ngân 1",
    "Thu Ngân 2", "Quản Lý Cửa Hàng", "Nguyễn Thị Mai", "Trần Lê Tuyền", "Lê Văn Vĩ",
    "Hoàng Thị Lan", "Phạm Văn Cường", "Đặng Thị Mai", "Bùi Văn Nam", "Đỗ Thị Thu",
    "Ngô Văn Quang", "Hồ Thị Hương", "Vũ Văn Long", "Dương Thị Trang", "Lý Văn Dũng",
    "Đào Thị Phương", "Đoàn Văn Hùng", "Vương Thị Thanh", "Trịnh Văn Bình", "Lâm Thị Tâm",
    "Phùng Văn Sơn", "Mai Thị Thúy", "Tô Văn Hải", "Địch Thị Yến", "Khổng Văn Tuấn",
    "Thạch Thị Bích", "Bao Văn Đức", "Chử Thị Kim", "Cấn Văn Thái", "Sầm Thị Nga",
    "NV: Nguyễn Hùng", "NV: Trần Quyên", "NV: Lê Nga", "CASHIER 01", "CASHIER 02",
    "Nguyen Thi Thu", "Tran Van Tuan", "Le Thi Hanh", "Pham Van Hoang", "Admin"
]

def generate_receipt(index):
    category = random.choice(list(DATA_VAULT.keys()))
    seller = random.choice(DATA_VAULT[category]["sellers"])
    address = random.choice(ADDRESSES)

    start_date = datetime(2024, 1, 1)
    random_days = random.randint(0, 800)
    random_time = timedelta(hours=random.randint(7, 22), minutes=random.randint(0, 59))
    timestamp_obj = start_date + timedelta(days=random_days) + random_time

    time_formats = ["%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d-%m-%Y %H:%M"]
    timestamp = timestamp_obj.strftime(random.choice(time_formats))

    receipt_no = f"HD{random.randint(100000, 999999)}"
    cashier = random.choice(CASHIERS)

    num_items = random.randint(2, 6)
    items_bought = []
    sub_total = 0

    for _ in range(num_items):
        item_name = random.choice(DATA_VAULT[category]["items"])
        qty = random.randint(1, 3)
        price = random.randint(DATA_VAULT[category]["price_range"][0]//1000, DATA_VAULT[category]["price_range"][1]//1000) * 1000
        line_total = qty * price
        sub_total += line_total
        items_bought.append({"name": item_name, "qty": qty, "price": price, "total": line_total})

    grand_total = sub_total
    cash_given = ((grand_total // 50000) + 1) * 50000
    change_due = cash_given - grand_total

    img_width = 480
    img_height = 500 + (len(items_bought) * 70)

    bg_color = random.choice([(250, 250, 250), (240, 240, 240), (235, 235, 235)])
    image = Image.new('RGB', (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(image)

    try:
        font_title = ImageFont.truetype("consolab.ttf", 26)
        font_text = ImageFont.truetype("consola.ttf", 16)
        font_bold = ImageFont.truetype("consolab.ttf", 18)
        font_total = ImageFont.truetype("consolab.ttf", 22)
    except:
        font_title = font_text = font_bold = font_total = ImageFont.load_default()

    y = 20
    text_color = (15, 15, 15)
    divider = random.choice(["-" * 50, "." * 60, "=" * 45])

    def draw_center(text, y_pos, font):
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        draw.text(((img_width - w) / 2, y_pos), text, font=font, fill=text_color)

    def draw_lr(text_left, text_right, y_pos, font):
        draw.text((20, y_pos), text_left, font=font, fill=text_color)
        bbox = draw.textbbox((0, 0), text_right, font=font)
        w = bbox[2] - bbox[0]
        draw.text((img_width - 20 - w, y_pos), text_right, font=font, fill=text_color)

    draw_center(seller, y, font_title); y += 35

    if len(address) > 40:
        draw_center(address[:40], y, font_text); y += 25
        draw_center(address[40:], y, font_text); y += 25
    else:
        draw_center(address, y, font_text); y += 25

    if category == "SUPERMARKET":
        draw_center(divider, y, font_text); y += 20
        draw_center(f"PHIẾU THANH TOÁN {seller}", y, font_bold); y += 30
        draw.text((20, y), f"Số CT: {receipt_no} - Ngày: {timestamp}", font=font_text, fill=text_color); y += 30
        draw_center(divider, y, font_text); y += 20
        draw_lr("SL      Giá bán (có VAT)", "Thành tiền", y, font_bold); y += 30

        for item in items_bought:
            words = item["name"].split()
            lines = [" ".join(words[i:i+5]) for i in range(0, len(words), 5)]
            for line in lines:
                draw.text((20, y), line, font=font_text, fill=text_color); y += 25

            qty_price = f"{item['qty']}       {item['price']:,}".replace(",", ".")
            if random.choice([True, False]): qty_price += " (VAT 8%)"
            draw_lr(qty_price, f"{item['total']:,}".replace(",", "."), y, font_text); y += 30

        draw_center(divider, y, font_text); y += 20
        draw_lr("Phải thanh toán:", f"{grand_total:,}".replace(",", "."), y, font_bold); y += 30
    elif category == "CIRCLE_K":
        draw_center("CONG TY TNHH VONG TRON DO", y, font_text); y += 20
        draw.text((20, y), "MST/TAX CODE: 0306182043", font=font_text, fill=text_color); y += 25
        draw.text((20, y), "PHIEU TINH TIEN/RECEIPT", font=font_bold, fill=text_color); y += 25
        draw.text((20, y), "Store: SG0155", font=font_text, fill=text_color); y += 20
        draw.text((20, y), address[:40], font=font_text, fill=text_color); y += 25
        draw.text((20, y), f"Receipt: {receipt_no}", font=font_text, fill=text_color); y += 25

        circle_k_time = timestamp_obj.strftime("%a %d %b %Y %H:%M:%S")
        draw.text((20, y), f"Date: {circle_k_time}", font=font_text, fill=text_color); y += 25
        draw.text((20, y), "Terminal: 03", font=font_text, fill=text_color); y += 20
        draw.text((20, y), f"CashierName: {cashier}", font=font_text, fill=text_color); y += 30

        draw_lr("ITEM  UnitPrice  Qty", "Amount", y, font_bold); y += 25

        total_qty = 0
        for item in items_bought:
            draw.text((20, y), item['name'][:35], font=font_text, fill=text_color); y += 25
            qty_price = f"      {item['price']:,}      {item['qty']}".replace(",", ".")
            draw_lr(qty_price, f"{item['total']:,}".replace(",", "."), y, font_text); y += 25
            total_qty += item['qty']

        draw_center(divider, y, font_text); y += 20
        draw_lr("Total Item(s) Qty:", str(total_qty), y, font_text); y += 25
        draw_lr("Subtotal:", f"{grand_total:,} VND".replace(",", "."), y, font_text); y += 25
        draw_lr("Total(+VAT):", f"{grand_total:,} VND".replace(",", "."), y, font_total); y += 35

        draw_lr("Cash:", f"{cash_given:,} VND".replace(",", "."), y, font_text); y += 25
        draw_lr("CHANGE DUE:", f"{change_due:,} VND".replace(",", "."), y, font_bold); y += 30

    elif category == "MINISO":
        draw.text((20, y), f"Miniso {address[:20]}", font=font_text, fill=text_color); y += 25
        draw.text((20, y), "Serial No.:8      POS NO.:VN84a", font=font_text, fill=text_color); y += 25
        draw.text((20, y), f"Cashier:VN8403[{cashier[:6]}]", font=font_text, fill=text_color); y += 25
        draw_center("--------Sales--------", y, font_text); y += 20
        draw_lr("Ma Sp       DG   SL", "TT", y, font_bold); y += 25

        total_qty = 0
        for item in items_bought:
            draw.text((20, y), item['name'][:35], font=font_text, fill=text_color); y += 25
            fake_barcode = f"{random.randint(1000000000, 9999999999)}"
            qty_price = f"{fake_barcode}  {item['price']:,}  {item['qty']}".replace(",", ".")
            draw_lr(qty_price, f"{item['total']:,}".replace(",", "."), y, font_text); y += 25
            total_qty += item['qty']

        draw_center(divider, y, font_text); y += 20
        draw.text((20, y), "Tong gia tri hoa don", font=font_text, fill=text_color); y += 25
        draw_lr(f"     {total_qty}Piece", f"{grand_total:,}VND".replace(",", "."), y, font_text); y += 25
        draw_lr("Giam gia", "0VND", y, font_text); y += 25
        draw.text((20, y), "Tong tien thanh toan", font=font_text, fill=text_color); y += 25
        draw_lr("", f"{grand_total:,}VND".replace(",", "."), y, font_total); y += 35

        draw_lr("Points Received", f"{int(grand_total/100)}", y, font_text); y += 25
        draw_lr("Tien mat tra lai khach", f"{change_due:,}VND".replace(",", "."), y, font_text); y += 25
        miniso_time = timestamp_obj.strftime("%Y-%m-%d %H:%M:%S")
        draw.text((20, y), f"Date:{miniso_time}", font=font_text, fill=text_color); y += 25

    elif category == "PHARMACY_2":
        draw_center("HÓA ĐƠN THUỐC", y, font_bold); y += 30
        draw_center(f"admin {timestamp} Số CT: {receipt_no}", y, font_text); y += 30
        draw.text((20, y), "Kho xuất: BÁN LẺ", font=font_text, fill=text_color); y += 25
        draw.text((20, y), f"NV: {cashier} - Tên KH: Khách lẻ", font=font_text, fill=text_color); y += 30
        draw_center(divider, y, font_text); y += 20

        for idx, item in enumerate(items_bought):
            draw.text((20, y), f"{idx+1}. {item['name']}", font=font_bold, fill=text_color); y += 25
            detail = f"ĐVT: chai ĐG: {item['price']:,} SL: {item['qty']} TT: {item['total']:,}".replace(",", ".")
            draw.text((20, y), detail, font=font_text, fill=text_color); y += 30

        draw_center(divider, y, font_text); y += 20
        draw_lr("TỔNG TIỀN THANH TOÁN", f"{grand_total:,}".replace(",", "."), y, font_total); y += 40

    elif category == "PHARMACY_1":
            draw_center(divider, y, font_text); y += 20
            draw_lr("Hạng thành viên:", "Gold", y, font_text); y += 25
            draw_lr("Tổng P-xu Đồng hiện tại:", f"{random.randint(100, 50000):,}", y, font_text); y += 30
            draw_lr(f"Thu ngân: {cashier}", timestamp, y, font_text); y += 30
            draw_lr("Tên       SL   Đơn giá", "Thành Tiền", y, font_bold); y += 30
            draw_center(divider, y, font_text); y += 20

            for item in items_bought:
                draw.text((20, y), item['name'][:35], font=font_text, fill=text_color); y += 25
                detail = f"          {item['qty']}    {item['price']:,}".replace(",", ".")
                draw_lr(detail, f"{item['total']:,}".replace(",", "."), y, font_text); y += 30

            draw_center(divider, y, font_text); y += 20
            draw_lr(f"Tổng {num_items} sản phẩm", f"{grand_total:,}".replace(",", "."), y, font_text); y += 25
            draw_lr("Tổng Tiền", f"{grand_total:,}".replace(",", "."), y, font_total); y += 40
            draw_lr("Phương thức thanh toán:", "Tiền mặt", y, font_text); y += 25
            draw_lr("Tiền khách đưa", f"{cash_given:,}".replace(",", "."), y, font_text); y += 25
            draw_lr("Trả lại", f"{change_due:,}".replace(",", "."), y, font_text); y += 30

    else:
        draw_center(divider, y, font_text); y += 20
        draw.text((20, y), f"Số CT: {receipt_no}", font=font_text, fill=text_color)
        draw_lr("", f"Ngày: {timestamp}", y, font_text); y += 25
        draw.text((20, y), f"Thu ngân: {cashier}", font=font_text, fill=text_color); y += 25

        draw_center(divider, y, font_text); y += 20
        draw_lr("Tên món", "Thành tiền", y, font_bold); y += 30
        draw_center(divider, y, font_text); y += 20

        for item in items_bought:
            name = item["name"][:25] + "..." if len(item["name"]) > 25 else item["name"]
            draw.text((20, y), name, font=font_bold, fill=text_color); y += 25
            qty_price = f"  {item['qty']} x {item['price']:,}".replace(",", ".")
            draw_lr(qty_price, f"{item['total']:,}".replace(",", "."), y, font_text); y += 25

        draw_center(divider, y, font_text); y += 20
        draw_lr("Cộng tiền hàng:", f"{sub_total:,}".replace(",", "."), y, font_text); y += 25

        y += 10
        grand_total_str = f"{grand_total:,}".replace(",", ".")
        draw_lr("TỔNG THANH TOÁN:", grand_total_str, y, font_total); y += 40

        draw_lr("Tiền khách đưa:", f"{cash_given:,}".replace(",", "."), y, font_text); y += 25
        draw_lr("Tiền thối lại:", f"{change_due:,}".replace(",", "."), y, font_text); y += 30

        draw_center(divider, y, font_text); y += 20
        draw_center("CẢM ƠN QUÝ KHÁCH & HẸN GẶP LẠI!", y, font_bold); y += 25

    y += 20
    for i in range(20, img_width-20, 4):
        if random.choice([True, False]):
            draw.line([(i, y), (i, y+40)], fill=(0,0,0), width=random.randint(1,3))

    file_name = f"syn_bill_{index:05d}.jpg"
    image_path = os.path.join(OUTPUT_DIR, file_name)
    image.save(image_path, "JPEG", quality=random.randint(50, 90))

    grand_total_str = f"{grand_total:,}".replace(",", ".")
    ground_truth = {
        "gt_parse": {
            "seller": seller,
            "address": address,
            "timestamp": timestamp,
            "total_cost": grand_total_str
        }
    }

    return {
        "file_name": file_name,
        "ground_truth": json.dumps(ground_truth, ensure_ascii=False)
    }

print(" KHỞI ĐỘNG HỆ THỐNG SINH DỮ LIỆU ...")
with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
    for i in range(NUM_IMAGES_TO_GENERATE):
        metadata = generate_receipt(i)
        f.write(json.dumps(metadata, ensure_ascii=False) + '\n')
        if (i+1) % 500 == 0:
            print(f"Đã render {i+1}/{NUM_IMAGES_TO_GENERATE} hóa đơn...")

print(f" HOÀN TẤT  {NUM_IMAGES_TO_GENERATE} HÓA ĐƠN!")
