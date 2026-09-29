import json
import pandas as pd


# =========================
# 1. Đọc metrics
# =========================

with open("reports/metrics/baseline_metrics.json", "r", encoding="utf-8") as f:
    baseline = json.load(f)

with open("reports/metrics/scorecard_metrics.json", "r", encoding="utf-8") as f:
    scorecard = json.load(f)

with open("reports/metrics_rf.json", "r", encoding="utf-8") as f:
    rf = json.load(f)

with open("reports/metrics_xgb.json", "r", encoding="utf-8") as f:
    xgb = json.load(f)


# =========================
# 2. Tạo bảng so sánh
# =========================

rows = [
    {
        "model": "Baseline - LIMIT_BAL",
        "auc": baseline["test_metrics"]["auc"],
        "gini": baseline["test_metrics"]["gini"],
        "ks": baseline["test_metrics"]["ks"]
    },
    {
        "model": "Scorecard / Logistic",
        "auc": scorecard["test_metrics"]["auc"],
        "gini": scorecard["test_metrics"]["gini"],
        "ks": scorecard["test_metrics"]["ks"]
    },
    {
        "model": "Random Forest",
        "auc": rf["test_metrics"]["auc"],
        "gini": rf["test_metrics"]["gini"],
        "ks": rf["test_metrics"]["ks"]
    },
    {
        "model": "XGBoost",
        "auc": xgb["test_metrics"]["auc"],
        "gini": xgb["test_metrics"]["gini"],
        "ks": xgb["test_metrics"]["ks"]
    }
]


df = pd.DataFrame(rows)


# =========================
# 3. Lưu CSV
# =========================

output_path = "reports/metrics/model_comparison.csv"

df.to_csv(output_path, index=False, encoding="utf-8-sig")


# =========================
# 4. In bảng ra màn hình
# =========================

print("===== MODEL COMPARISON =====")
print(df.to_string(index=False))

print(f"\nĐã lưu: {output_path}")