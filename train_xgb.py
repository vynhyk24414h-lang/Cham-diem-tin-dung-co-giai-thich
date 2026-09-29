# --- Đảm bảo stdout/stderr hỗ trợ UTF-8 trên Windows ---
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
train_xgb.py
============
Huấn luyện mô hình XGBoost để dự báo xác suất vỡ nợ (PD) trong bài toán
chấm điểm tín dụng.

Chạy độc lập:
    python train_xgb.py

Đầu ra:
    models/xgb_model.pkl
    reports/pred_xgb.csv, reports/pred_xgb_train.csv
    reports/metrics_xgb.json
    reports/feature_importance_xgb.csv
"""

import os
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ==============================================================================
# --- [HẰNG SỐ] Cấu hình đường dẫn và siêu tham số ---
# ==============================================================================
RANDOM_STATE = 42          # Seed cố định cho tái lập kết quả
TRAIN_PATH   = "data/train.csv"
TEST_PATH    = "data/test.csv"
TARGET_COL   = "default"   # Tên cột mục tiêu (0 = không vỡ nợ, 1 = vỡ nợ)
MODEL_DIR    = "models"
REPORT_DIR   = "reports"

# Cột loại khỏi đặc trưng (ID hoặc định danh tương tự — tránh rò rỉ dữ liệu)
# Nếu dataset không có cột ID, giữ danh sách rỗng.
EXCLUDE_COLS: list[str] = []


# ==============================================================================
# --- [BƯỚC 1] Đọc dữ liệu ---
# ==============================================================================
def load_data(train_path: str, test_path: str, target_col: str,
              exclude_cols: list[str]) -> tuple:
    """
    Đọc tập train và test, tách đặc trưng X và nhãn y.

    Tham số:
        train_path   : đường dẫn file CSV train
        test_path    : đường dẫn file CSV test
        target_col   : tên cột mục tiêu
        exclude_cols : danh sách cột cần loại khỏi đặc trưng (VD: ID)

    Giá trị trả về:
        X_train, y_train, X_test, y_test, feature_cols
    """
    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    drop_cols   = [target_col] + exclude_cols
    feature_cols = [c for c in df_train.columns if c not in drop_cols]

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_test  = df_test[feature_cols]
    y_test  = df_test[target_col]

    print(f"  Train: {X_train.shape[0]:,} mẫu | Test: {X_test.shape[0]:,} mẫu")
    print(f"  Số đặc trưng: {len(feature_cols)}")
    print(f"  Tỷ lệ default (train): {y_train.mean()*100:.2f}%")
    return X_train, y_train, X_test, y_test, feature_cols


# ==============================================================================
# --- [BƯỚC 2] Tính scale_pos_weight để xử lý mất cân bằng lớp ---
# ==============================================================================
def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Tính scale_pos_weight = số mẫu lớp âm / số mẫu lớp dương.
    XGBoost dùng tham số này để cân bằng lớp trong hàm mất mát.

    Tham số:
        y_train : nhãn tập train (0/1)

    Giá trị trả về:
        scale_pos_weight (float)
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = n_neg / n_pos
    print(f"  scale_pos_weight = {n_neg:,} / {n_pos:,} = {spw:.4f}")
    return spw


# ==============================================================================
# --- [BƯỚC 3] Tinh chỉnh siêu tham số bằng GridSearchCV ---
# ==============================================================================
def tune_hyperparameters(X_train, y_train, scale_pos_weight: float) -> dict:
    """
    Tìm siêu tham số tốt nhất cho XGBoost bằng GridSearchCV với
    StratifiedKFold 5 fold, chấm điểm theo roc_auc.
    Tập test KHÔNG được dùng ở bước này.

    Tham số:
        X_train          : đặc trưng tập train
        y_train          : nhãn tập train
        scale_pos_weight : tỷ lệ cân bằng lớp

    Giá trị trả về:
        dict chứa best_params và cv_results
    """
    # Lưới tham số cần tìm
    param_grid = {
        "n_estimators"  : [100, 200],
        "max_depth"     : [3, 5, 7],
        "learning_rate" : [0.01, 0.05, 0.1],
    }
    total_fits = (len(param_grid["n_estimators"])
                  * len(param_grid["max_depth"])
                  * len(param_grid["learning_rate"])
                  * 5)
    print(f"  Số lượng fits: {total_fits}")

    base_model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",  # tắt cảnh báo mặc định
        verbosity=0,
        use_label_encoder=False,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,   # Tự refit trên toàn bộ train với tham số tốt nhất
    )

    t0 = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_params  = grid_search.best_params_
    best_score   = grid_search.best_score_
    best_std     = grid_search.cv_results_["std_test_score"][grid_search.best_index_]

    print(f"  Thời gian GridSearchCV: {elapsed:.2f}s")
    print(f"  Tham số tốt nhất : {best_params}")
    print(f"  CV AUC (trung bình ± std): {best_score:.4f} ± {best_std:.4f}")

    return {
        "best_params"  : best_params,
        "cv_auc_mean"  : best_score,
        "cv_auc_std"   : best_std,
        "best_estimator": grid_search.best_estimator_,
    }


# ==============================================================================
# --- [BƯỚC 4] Đánh giá mô hình: AUC, Gini, KS ---
# ==============================================================================
def evaluate_model(model, X, y, label: str = "") -> dict:
    """
    Tính AUC-ROC, Gini và KS cho một tập dữ liệu.

    Tham số:
        model : mô hình đã huấn luyện (có predict_proba)
        X     : đặc trưng
        y     : nhãn thực
        label : tên tập ("Train" hoặc "Test") để in ra

    Giá trị trả về:
        dict chứa auc, gini, ks
    """
    prob  = model.predict_proba(X)[:, 1]
    auc   = roc_auc_score(y, prob)
    gini  = 2 * auc - 1

    fpr, tpr, _ = roc_curve(y, prob)
    ks          = float(np.max(tpr - fpr))

    if label:
        print(f"  [{label}] AUC={auc:.4f} | Gini={gini:.4f} | KS={ks:.4f}")
    return {"auc": auc, "gini": gini, "ks": ks, "prob": prob}


# ==============================================================================
# --- [BƯỚC 5] Lưu model, kết quả dự đoán, metrics, feature importance ---
# ==============================================================================
def save_outputs(model, tune_result: dict,
                 train_metrics: dict, test_metrics: dict,
                 X_train, y_train, X_test, y_test,
                 feature_cols: list[str]) -> None:
    """
    Lưu toàn bộ đầu ra của mô hình XGBoost vào thư mục models/ và reports/.

    Tham số:
        model         : mô hình đã refit
        tune_result   : kết quả từ tune_hyperparameters()
        train_metrics : dict AUC/Gini/KS tập train
        test_metrics  : dict AUC/Gini/KS tập test
        X_train, y_train : dữ liệu train
        X_test, y_test   : dữ liệu test
        feature_cols  : danh sách tên đặc trưng

    Giá trị trả về:
        None
    """
    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    # --- Lưu model .pkl ---
    pkl_path = os.path.join(MODEL_DIR, "xgb_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    size_mb = os.path.getsize(pkl_path) / (1024 ** 2)
    print(f"  Đã lưu model → {pkl_path} ({size_mb:.2f} MB)")
    if size_mb > 90:
        print("  ⚠️ CẢNH BÁO: File > 90 MB — GitHub từ chối file trên 100 MB!")
        print("     Gợi ý: giảm n_estimators hoặc max_depth.")

    # --- Lưu dự đoán tập test ---
    df_pred_test = pd.DataFrame({
        "index"       : X_test.index,
        "y_true"      : y_test.values,
        "prob_default": test_metrics["prob"],
    })
    pred_test_path = os.path.join(REPORT_DIR, "pred_xgb.csv")
    df_pred_test.to_csv(pred_test_path, index=False)
    print(f"  Đã lưu dự đoán test  → {pred_test_path}")

    # --- Lưu dự đoán tập train ---
    df_pred_train = pd.DataFrame({
        "index"       : X_train.index,
        "y_true"      : y_train.values,
        "prob_default": train_metrics["prob"],
    })
    pred_train_path = os.path.join(REPORT_DIR, "pred_xgb_train.csv")
    df_pred_train.to_csv(pred_train_path, index=False)
    print(f"  Đã lưu dự đoán train → {pred_train_path}")

    # --- Lưu metrics JSON ---
    metrics = {
        "best_params"  : tune_result["best_params"],
        "cv_auc_mean"  : round(tune_result["cv_auc_mean"], 6),
        "cv_auc_std"   : round(tune_result["cv_auc_std"],  6),
        "train": {
            "auc" : round(train_metrics["auc"],  6),
            "gini": round(train_metrics["gini"], 6),
            "ks"  : round(train_metrics["ks"],   6),
        },
        "test": {
            "auc" : round(test_metrics["auc"],  6),
            "gini": round(test_metrics["gini"], 6),
            "ks"  : round(test_metrics["ks"],   6),
        },
        "auc_diff_train_test": round(train_metrics["auc"] - test_metrics["auc"], 6),
    }
    metrics_path = os.path.join(REPORT_DIR, "metrics_xgb.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"  Đã lưu metrics       → {metrics_path}")

    # --- Lưu feature importance ---
    importance = model.feature_importances_
    df_fi = pd.DataFrame({
        "feature"   : feature_cols,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi_path = os.path.join(REPORT_DIR, "feature_importance_xgb.csv")
    df_fi.to_csv(fi_path, index=False)
    print(f"  Đã lưu feature imp.  → {fi_path}")

    return df_fi


# ==============================================================================
# --- MAIN ---
# ==============================================================================
def main():
    """Hàm chính: đọc dữ liệu → tinh chỉnh → đánh giá → lưu kết quả."""
    t_start = time.time()

    print("=" * 60)
    print("  HUẤN LUYỆN MÔ HÌNH XGBoost — Chấm điểm tín dụng")
    print("=" * 60)

    # --- [BƯỚC 1] Đọc dữ liệu ---
    print("\n[BƯỚC 1] Đọc dữ liệu...")
    X_train, y_train, X_test, y_test, feature_cols = load_data(
        TRAIN_PATH, TEST_PATH, TARGET_COL, EXCLUDE_COLS
    )

    # --- [BƯỚC 2] Tính scale_pos_weight ---
    print("\n[BƯỚC 2] Tính scale_pos_weight...")
    spw = compute_scale_pos_weight(y_train)

    # --- [BƯỚC 3] Tinh chỉnh siêu tham số (CHỈ trên train) ---
    print("\n[BƯỚC 3] Tinh chỉnh siêu tham số bằng GridSearchCV (5-fold CV)...")
    tune_result = tune_hyperparameters(X_train, y_train, spw)
    model = tune_result["best_estimator"]

    # --- [BƯỚC 4] Đánh giá trên train và test ---
    print("\n[BƯỚC 4] Đánh giá mô hình...")
    train_metrics = evaluate_model(model, X_train, y_train, label="Train")
    test_metrics  = evaluate_model(model, X_test,  y_test,  label="Test ")

    diff_auc = train_metrics["auc"] - test_metrics["auc"]
    print(f"\n{'─'*50}")
    print(f"  {'Chỉ số':<12} {'Train':>10} {'Test':>10} {'Chênh lệch':>12}")
    print(f"  {'─'*44}")
    print(f"  {'AUC-ROC':<12} {train_metrics['auc']:>10.4f} {test_metrics['auc']:>10.4f} {diff_auc:>12.4f}")
    print(f"  {'Gini':<12} {train_metrics['gini']:>10.4f} {test_metrics['gini']:>10.4f} {train_metrics['gini']-test_metrics['gini']:>12.4f}")
    print(f"  {'KS':<12} {train_metrics['ks']:>10.4f} {test_metrics['ks']:>10.4f} {train_metrics['ks']-test_metrics['ks']:>12.4f}")
    print(f"{'─'*50}")
    if diff_auc > 0.05:
        print(f"  ⚠️  Chênh lệch AUC = {diff_auc:.4f} > 0.05 → có dấu hiệu overfitting")
    else:
        print(f"  ✅ Chênh lệch AUC = {diff_auc:.4f} ≤ 0.05 → tổng quát hóa tốt")

    # --- [BƯỚC 5] Lưu đầu ra ---
    print("\n[BƯỚC 5] Lưu model và báo cáo...")
    df_fi = save_outputs(
        model, tune_result,
        train_metrics, test_metrics,
        X_train, y_train, X_test, y_test,
        feature_cols,
    )

    # --- [BƯỚC 6] In 15 đặc trưng quan trọng nhất ---
    print("\n[BƯỚC 6] Top 15 đặc trưng quan trọng nhất:")
    print(f"  {'Hạng':<6} {'Đặc trưng':<30} {'Importance':>12}")
    print(f"  {'─'*50}")
    for i, row in df_fi.head(15).iterrows():
        print(f"  {i+1:<6} {row['feature']:<30} {row['importance']:>12.4f}")

    t_total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  Hoàn tất! Tổng thời gian: {t_total:.2f} giây")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
