# Smart Expense Tracker & Insight

Hệ thống quản lý chi tiêu kết hợp phân tích hành vi mua sắm và OCR hóa đơn bằng mô hình Deep Learning - DONUT.

---

## Tổng Quan Dự Án

Dự án **Finance_AI_App** là một monorepo bao gồm 4 thành phần chính được thiết kế để hoạt động cùng nhau:

1. **`LTDD_Finance` (Frontend):** Ứng dụng di động được viết bằng **Flutter (Dart)**.
2. **`core-backend` (Backend chính):** API Server được phát triển bằng **Spring Boot (Java 21)** kết hợp hệ quản trị cơ sở dữ liệu **PostgreSQL**.
3. **`ai-service` (AI Service):** Dịch vụ Deep Learning và khai thác dữ liệu được viết bằng **FastAPI (Python)**, hỗ trợ OCR hóa đơn (Donut model) và gợi ý tài chính (WCM-SPADE).
4. **`DuDoanHanhViMuaHang` (Analytics Demo):** Demo phân tích hành vi mua sắm sản phẩm và đề xuất sản phẩm bó (WCM-SPADE).

---

## Quản Lý Bảo Mật & Khóa 

> [!IMPORTANT]
> **Quy tắc vàng:** Không bao giờ đẩy các tệp chứa thông tin nhạy cảm (Keys, Passwords, API Tokens, Environment Variables) lên GitHub.

### 1. Cách `.gitignore` Bảo Vệ Khóa Của Bạn
Tệp [.gitignore](file:///d:/data/HUIT/Nam3/HK2/AAA_Finance/Finance_AI_App/Final_Finance/.gitignore) đã được cấu hình chặt chẽ để tự động loại bỏ các tệp nhạy cảm sau khỏi Git:
- **Biến môi trường:** Các tệp `.env`, `*.env`, `.env.*` chứa thông tin cấu hình môi trường.
- **Chứng chỉ/Khóa:** Các tệp khóa bảo mật như `*.pem`, `*.key`, `*.keystore`, `*.jks`.
- **Cấu hình Spring Boot:** Bỏ qua toàn bộ các tệp cấu hình thực tế `**/application*.properties`, `application.yml`, v.v.
- **Google Services:** Các tệp cấu hình Firebase/Google SDK như `google-services.json` và `GoogleService-Info.plist`.
- **Thư mục chứa Key:** Bỏ qua mọi thư mục có tên `**/keys/`.

### 2. Cách Thiết Lập Khóa Khi Clone Dự Án Về
Để dự án có thể chạy được sau khi clone từ GitHub, bạn cần tạo lại các tệp cấu hình cục bộ:

#### Cấu hình cho Core Backend (Spring Boot)
1. Truy cập thư mục [core-backend/src/main/resources](file:///d:/data/HUIT/Nam3/HK2/AAA_Finance/Finance_AI_App/Final_Finance/core-backend/src/main/resources).
2. Copy tệp [application.properties.template](file:///d:/data/HUIT/Nam3/HK2/AAA_Finance/Finance_AI_App/Final_Finance/core-backend/src/main/resources/application.properties.template) và đổi tên thành `application.properties`.
3. Mở tệp `application.properties` mới tạo và điền các khóa cá nhân của bạn:
   - Thông tin kết nối PostgreSQL (`spring.datasource.username`, `spring.datasource.password`).
   - Khóa Cloudinary (`cloudinary.cloud-name`, `cloudinary.api-key`, `cloudinary.api-secret`).
   - Khóa cổng thanh toán thử nghiệm (VNPay, ZaloPay).
   - Mã Client ID Google OAuth (`app.google.web-client-id`).
   - Tài khoản gửi Mail (`spring.mail.username`, `spring.mail.password` - mật khẩu ứng dụng Gmail).

---

## Yêu Cầu Hệ Thống

Để chạy toàn bộ hệ thống, máy tính của bạn cần cài đặt sẵn:
- **Flutter SDK** (phiên bản `>= 3.19` hoặc mới hơn)
- **Java Development Kit (JDK 21)**
- **Python** (khuyên dùng phiên bản `3.10` - `3.12`, không dùng `3.14` do thư viện PyTorch chưa hỗ trợ chính thức)
- **PostgreSQL** (phiên bản `15` hoặc `16`)
- **Maven** (được tích hợp sẵn qua `./mvnw`)
- Công cụ **uv** (tùy chọn - giúp cài đặt thư viện Python nhanh hơn)

---

## Hướng Dẫn Chạy Ứng Dụng

Bạn có thể khởi động toàn bộ ứng dụng bằng một script tự động hoặc khởi chạy thủ công từng dịch vụ.

### Cách 1: Chạy Tự Động Bằng Script (Khuyên Dùng)

#### Trên Windows:
Double click vào tệp [run_system.bat](file:///d:/data/HUIT/Nam3/HK2/AAA_Finance/Finance_AI_App/Final_Finance/run_system.bat) hoặc chạy lệnh sau trong Terminal (CMD/PowerShell):
```bash
.\run_system.bat
```

#### Trên Linux / macOS:
Cấp quyền thực thi và chạy tệp [run_system.sh](file:///d:/data/HUIT/Nam3/HK2/AAA_Finance/Finance_AI_App/Final_Finance/run_system.sh):
```bash
chmod +x run_system.sh
./run_system.sh
```

---

### Cách 2: Khởi Chạy Thủ Công Từng Dịch Vụ

#### Bước 1: Khởi động AI Service (FastAPI)
1. Di chuyển vào thư mục dịch vụ AI:
   ```bash
   cd ai-service
   ```
2. Tạo môi trường ảo Python 3.12 và cài đặt thư viện:
   - *Sử dụng công cụ `uv` (nhanh nhất):*
     ```bash
     uv venv --python 3.12
     uv pip install -r requirements.txt
     ```
   - *Sử dụng `venv` truyền thống:*
     ```bash
     python -m venv .venv
     # Windows:
     .venv\Scripts\activate
     # Linux/macOS:
     source .venv/bin/activate
     pip install -r requirements.txt
     ```
3. Khởi chạy FastAPI Server:
   ```bash
   uvicorn ai_service:app --host 0.0.0.0 --port 8000 --reload
   ```
   *Swagger UI tài liệu API sẽ hoạt động tại:* `http://localhost:8000/docs`

#### Bước 2: Khởi động Core Backend (Spring Boot)
1. Đảm bảo bạn đã tạo cơ sở dữ liệu có tên `finance_db` trong PostgreSQL và cập nhật thông tin trong tệp `application.properties`.
2. Di chuyển vào thư mục backend:
   ```bash
   cd core-backend
   ```
3. Khởi chạy dự án Spring Boot:
   - *Trên Windows:*
     ```bash
     mvnw.cmd spring-boot:run
     ```
   - *Trên Linux/macOS:*
     ```bash
     ./mvnw spring-boot:run
     ```
   *Backend API sẽ chạy trên cổng:* `8081`

#### Bước 3: Khởi động Mobile App (Flutter)
1. Di chuyển vào thư mục ứng dụng di động:
   ```bash
   cd LTDD_Finance
   ```
2. Tải các thư viện phụ thuộc:
   ```bash
   flutter pub get
   ```
3. Chạy ứng dụng trên thiết bị mục tiêu:
   - *Chạy trên trình duyệt Web Chrome:*
     ```bash
     flutter run -d chrome
     ```
   - *Chạy trên máy ảo Android:*
     ```bash
     flutter run -d emulator --dart-define=API_BASE=http://10.0.2.2:8081 --dart-define=ML_BASE=http://10.0.2.2:8000
     ```
   - *Chạy trên thiết bị thật (Qua mạng LAN - thay thế IP tương ứng với máy tính của bạn):*
     ```bash
     flutter run -d <DEVICE_ID> --dart-define=API_BASE=http://192.168.100.4:8081 --dart-define=ML_BASE=http://192.168.100.4:8000
     ```

#### Bước 4: Khởi động Retail Analytics Demo (DuDoanHanhViMuaHang) - Tùy chọn
1. Di chuyển vào thư mục:
   ```bash
   cd DuDoanHanhViMuaHang
   ```
2. Cài đặt môi trường ảo và dependencies tương tự:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r pyproject.toml
   ```
3. Chạy ứng dụng:
   ```bash
   uvicorn app.main:app --port 8002 --reload
   ```

---

## Sơ Đồ Cổng Dịch Vụ & Địa Chỉ Truy Cập

| Thành phần | Cổng mặc định | URL tài liệu / Giao diện | Mô tả |
| :--- | :--- | :--- | :--- |
| **FastAPI (AI Service)** | `8000` | `http://localhost:8000/docs` | OCR hóa đơn (`/donut/scan`), Phân tích SPADE (`/spade/analyze`) |
| **Spring Boot Backend** | `8081` | `http://localhost:8081` | Quản lý người dùng, giao dịch, ví, kết nối CSDL và xử lý logic nghiệp vụ |
| **Flutter Web/App** | Tự động | Giao diện di động / Trình duyệt | Giao diện chính của DonutPay |
| **Retail Analytics Web**| `8002` | `http://localhost:8002` | Dashboard demo phân tích và đề xuất hành vi mua hàng |

---

## Khắc Phục Lỗi Thường Gặp

1. **Lỗi không tìm thấy lệnh `uvicorn` khi chạy AI Service:**
   - *Nguyên nhân:* Do `uvicorn` được cài đặt trong môi trường ảo (.venv) chứ không phải cài đặt toàn cục.
   - *Khắc phục:* Bạn phải kích hoạt môi trường ảo (`source .venv/bin/activate`) trước khi chạy lệnh uvicorn, hoặc chạy bằng `uv run uvicorn...`.

2. **Lỗi kết nối cơ sở dữ liệu Spring Boot:**
   - *Khắc phục:* Hãy kiểm tra PostgreSQL đã chạy chưa, kiểm tra cổng mặc định có phải là `5432` không, và đảm bảo Database tên `finance_db` đã được tạo trước khi khởi chạy Backend.

3. **Lỗi PyTorch không tương thích:**
   - *Khắc phục:* Không sử dụng Python 3.14 vì chưa được hỗ trợ PyTorch ổn định. Hãy cài đặt môi trường ảo chạy Python 3.12 hoặc 3.11.
