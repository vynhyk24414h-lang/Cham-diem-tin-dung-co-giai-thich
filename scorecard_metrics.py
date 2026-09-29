import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


# =========================
# 1. Đọc dữ liệu và model
# =========================
test = pd.read_csv("data/test.csv")

with open("models/logit_scorecard.pkl", "rb") as f:
    model = __import__("pickle").load(f)


# =========================
# 2. Lấy thông tin Scorecard
# =========================
features = model["features"]
params = model["params"]

# Hệ số của mô hình
coef = pd.Series(params)

# Ma trận biến đầu vào
X = test[features]

# Biến mục tiêu
y = test["default"]


# =========================
# 3. Tính logit / risk score
# =========================
logit = coef["const"]

for feature in features:
    logit += coef[feature] * X[feature]

# Logit càng lớn -> xác suất default càng lớn
risk_score = logit


# =========================
# 4. Tính AUC
# =========================
auc = roc_auc_score(y, risk_score)


# =========================
# 5. Tính Gini
# =========================
gini = 2 * auc - 1


# =========================
# 6. Tính KS
# =========================
fpr, tpr, thresholds = roc_curve(y, risk_score)
ks = np.max(tpr - fpr)


# =========================
# 7. In kết quả
# =========================
print("===== SCORECARD / LOGISTIC =====")
print(f"AUC  : {auc:.6f}")
print(f"Gini : {gini:.6f}")
print(f"KS   : {ks:.6f}")


# =========================
# 8. Lưu kết quả
# =========================
metrics = {
    "model_name": "Scorecard / Logistic",
    "test_metrics": {
        "auc": float(auc),
        "gini": float(gini),
        "ks": float(ks)
    }
}

with open(
    "reports/metrics/scorecard_metrics.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=4)

print("\nĐã lưu: reports/metrics/scorecard_metrics.json")