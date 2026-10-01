# 💳 Chấm Điểm Tín Dụng Có Giải Thích (Explainable Credit Scoring)

Đề án xây dựng và đánh giá hệ thống ước lượng Xác suất vỡ nợ (PD) trên bộ dữ liệu **UCI Default of Credit Card Clients (30.000 khách hàng)**, so sánh giữa mô hình **Logistic Regression Scorecard (WoE/IV)** và các mô hình học máy (**Random Forest, XGBoost**), kết hợp giải thích mô hình bằng **SHAP**, kiểm định tính công bằng (**Fairness Audit**) và xác định ngưỡng phê duyệt tối ưu.

---

## 🚀 Hướng dẫn chạy dự án (Quick Start)

Dự án đã được tích hợp sẵn vào một file điều phối duy nhất (`run_all.py`). Bạn chỉ cần mở Terminal tại thư mục gốc của dự án và chạy 2 lệnh sau:

**Bước 1: Cài đặt các thư viện cần thiết**
```bash
pip install -r requirements.txt
**Bước 2: Khởi chạy toàn bộ quy trình (Pipeline)
python run_all.py
** Bước 3: Chạy dashboard của dự án
python -m streamlit run app.py
