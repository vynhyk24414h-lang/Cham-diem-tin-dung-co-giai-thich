# 💳 Chấm Điểm Tín Dụng Có Giải Thích (Explainable Credit Scoring)

Đề án xây dựng và đánh giá hệ thống ước lượng Xác suất vỡ nợ (PD) trên bộ dữ liệu **UCI Default of Credit Card Clients (30.000 khách hàng)**, so sánh giữa mô hình **Logistic Regression Scorecard (WoE/IV)** và các mô hình học máy (**Random Forest, XGBoost**), kết hợp giải thích mô hình bằng **SHAP**, kiểm định tính công bằng (**Fairness Audit**) và xác định ngưỡng phê duyệt tối ưu.

---

## 🚀 Hướng dẫn Chạy Nhanh (Quick Start)

Dự án đã được tích hợp toàn bộ pipeline vào một file điều phối duy nhất (`run_all.py`). Bạn chỉ cần mở Terminal tại thư mục dự án và thực hiện:

**Bước 1: Cài đặt thư viện**
```bash
pip install -r requirements.txt

**Bước 2: Khởi chạy điều phối
python run_all.py

## Nhấn phím 1 (Khuyến nghị): Chạy kiểm tra nhanh toàn bộ pipeline (~20 giây sử dụng model đã lưu) và tự động mở Dashboard HTML báo cáo trực quan trên trình duyệt. 
##  Nhấn phím 2: Huấn luyện lại toàn bộ mô hình từ đầu bằng 5-Fold Stratified CV (khoảng 5–10 phút).   
## Nhấn phím 3: Xem nhanh bản tóm tắt các bảng chỉ số ngay trên Terminal[cite: 19].
## Nhấn phím 4: Mở ngay Dashboard tương tác (ưu tiên bản HTML chính dashboard.html, tự động kích hoạt Streamlit app.py nếu cần dự phòng)