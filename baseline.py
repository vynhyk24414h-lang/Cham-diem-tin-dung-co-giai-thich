import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


# =========================
# 1. Đọc dữ liệu test
# =========================
test = pd.read_csv("data/test.csv")

X = test["LIMIT_BAL"]
y = test["default"]


# =========================
# 2. Tạo điểm rủi ro
# =========================
# LIMIT_BAL càng thấp -> rủi ro default càng cao
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
ks = np.max(tpr - fpr)


# =========================
# 6. In kết quả
# =========================
print("===== BASELINE: LIMIT_BAL =====")
print(f"AUC  : {auc:.6f}")
print(f"Gini : {gini:.6f}")
print(f"KS   : {ks:.6f}")


# =========================
# 7. Lưu kết quả ra JSON
# =========================
metrics = {
    "model_name": "Baseline - LIMIT_BAL",
    "test_metrics": {
        "auc": float(auc),
        "gini": float(gini),
        "ks": float(ks)
    }
}

output_path = "reports/metrics/baseline_metrics.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=4)

print(f"\nĐã lưu kết quả: {output_path}")