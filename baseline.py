import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


# =========================
# 1. Đọc dữ liệu test
# =========================
test = pd.read_csv("data/test.csv")

# Biến đầu vào của baseline
X = test["LIMIT_BAL"]

# Biến mục tiêu
y = test["default"]


# =========================
# 2. Tạo điểm dự báo
# =========================
# LIMIT_BAL càng thấp -> rủi ro default càng cao
# Vì vậy dùng -LIMIT_BAL làm score rủi ro
score = -X


# =========================
# 3. Tính AUC
# =========================
auc = roc_auc_score(y, score)


# =========================
# 4. Tính Gini
# =========================
gini = 2 * auc - 1


# =========================
# 5. Tính KS
# =========================
fpr, tpr, thresholds = roc_curve(y, score)
ks = max(tpr - fpr)


# =========================
# 6. In kết quả
# =========================
print("===== BASELINE: LIMIT_BAL =====")
print(f"AUC  : {auc:.6f}")
print(f"Gini : {gini:.6f}")
print(f"KS   : {ks:.6f}")