import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve

# --- 1. Đọc dữ liệu từ file dự đoán của XGBoost ---
pred_path = os.path.join("reports", "pred_xgb.csv")

if not os.path.exists(pred_path):
    print(
        f"❌ Không tìm thấy file '{pred_path}'. Hãy đảm bảo bạn đã chạy train_xgb.py trước!"
    )
    exit()

df_pred = pd.read_csv(pred_path)
y_true = df_pred["y_true"].values
prob_default = df_pred["prob_default"].values

# --- 2. Tính đường cong ROC và tìm điểm max KS ---
fpr, tpr, thresholds = roc_curve(y_true, prob_default)
ks_values = tpr - fpr
best_idx = np.argmax(ks_values)

best_ks = ks_values[best_idx]
best_thresh = thresholds[best_idx]

# --- 3. Phân loại theo ngưỡng best_thresh ---
y_pred = (prob_default >= best_thresh).astype(int)

# --- 4. Tính ma trận nhầm lẫn (Confusion Matrix) ---
# 0 = Không vỡ nợ (Duyệt), 1 = Vỡ nợ (Từ chối)
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

approved_count = tn + fn
rejected_count = fp + tp
total_count = len(df_pred)

approval_rate = (approved_count / total_count) * 100
rejection_rate = (rejected_count / total_count) * 100
bad_rate_approved = (fn / approved_count) * 100

# --- 5. In kết quả báo cáo ---
print("\n" + "=" * 60)
print(" KẾT QUẢ TÍNH TOÁN NGƯỠNG PHÊ DUYỆT TÍN DỤNG (TỪ PRED_XGB.CSV)")
print("=" * 60)
print(f"• Tổng số mẫu kiểm tra (Test Set): {total_count:,} quan sát")
print(f"• Chỉ số KS lớn nhất (max KS)    : {best_ks:.6f}")
print(f"• Ngưỡng cắt tối ưu theo KS      : {best_thresh:.6f}")
print("-" * 60)
print("MA TRẬN PHÂN LOẠI (CONFUSION MATRIX):")
print(
    f"  - TN (Không vỡ nợ - Dự báo đúng) : {tn:,}  (Khách tốt, được duyệt)"
)
print(
    f"  - FP (Không vỡ nợ - Dự báo sai)  : {fp:,}  (Khách tốt, bị từ chối nhầm)"
)
print(f"  - FN (Vỡ nợ - Dự báo sai)        : {fn:,}  (Khách xấu, bị duyệt nhầm)")
print(f"  - TP (Vỡ nợ - Dự báo đúng)       : {tp:,}  (Khách xấu, bị từ chối)")
print("-" * 60)
print("TÁC ĐỘNG ĐẾN QUYẾT ĐỊNH TÍN DỤNG:")
print(f"  - Tỷ lệ phê duyệt (Approval Rate): {approval_rate:.2f}% ({approved_count:,} hồ sơ)")
print(f"  - Tỷ lệ từ chối (Reject Rate)    : {rejection_rate:.2f}% ({rejected_count:,} hồ sơ)")
print(f"  - Tỷ lệ vỡ nợ trong tập duyệt    : {bad_rate_approved:.2f}% ({fn:,}/{approved_count:,})")
print("=" * 60 + "\n")