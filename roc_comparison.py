import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from xgboost import XGBClassifier


# ==========================================
# 1. Đọc test data
# ==========================================

test = pd.read_csv("data/test.csv")
woe_test = pd.read_csv("data/woe_test.csv")

y = test["default"]


# ==========================================
# 2. Baseline - LIMIT_BAL
# ==========================================

baseline_score = -test["LIMIT_BAL"]

fpr_base, tpr_base, _ = roc_curve(y, baseline_score)
auc_base = auc(fpr_base, tpr_base)


# ==========================================
# 3. Scorecard / Logistic
# ==========================================

with open("models/logit_scorecard.pkl", "rb") as f:
    scorecard = pickle.load(f)

features = scorecard["features"]
params = scorecard["params"]

logit_score = np.full(len(test), params["const"], dtype=float)

for feature in features:
    logit_score += params[feature] * woe_test[feature]

fpr_logit, tpr_logit, _ = roc_curve(y, logit_score)
auc_logit = auc(fpr_logit, tpr_logit)


# ==========================================
# 4. Random Forest
# ==========================================

with open("models/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

X_rf = test.drop(columns=["default"])

rf_prob = rf_model.predict_proba(X_rf)[:, 1]

fpr_rf, tpr_rf, _ = roc_curve(y, rf_prob)
auc_rf = auc(fpr_rf, tpr_rf)


# ==========================================
# 5. XGBoost
# ==========================================

with open("models/xgb_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)

X_xgb = test.drop(columns=["default"])

xgb_prob = xgb_model.predict_proba(X_xgb)[:, 1]

fpr_xgb, tpr_xgb, _ = roc_curve(y, xgb_prob)
auc_xgb = auc(fpr_xgb, tpr_xgb)


# ==========================================
# 6. Vẽ ROC
# ==========================================

plt.figure(figsize=(8, 6))

plt.plot(
    fpr_base,
    tpr_base,
    label=f"Baseline - LIMIT_BAL (AUC = {auc_base:.4f})"
)

plt.plot(
    fpr_logit,
    tpr_logit,
    label=f"Scorecard / Logistic (AUC = {auc_logit:.4f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {auc_rf:.4f})"
)

plt.plot(
    fpr_xgb,
    tpr_xgb,
    label=f"XGBoost (AUC = {auc_xgb:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

output_path = "reports/roc_comparison.png"

plt.savefig(output_path, dpi=300)

plt.show()

print(f"Đã lưu: {output_path}")