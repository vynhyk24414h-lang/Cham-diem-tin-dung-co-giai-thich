"""
train_rf.py - Huấn luyện mô hình Random Forest cho đề án chấm điểm tín dụng (Mục 3.2).
Dự báo xác suất vỡ nợ (Probability of Default - PD).
"""

import sys
import io

# Đảm bảo mã hóa UTF-8 cho luồng in ra console trên Windows tránh lỗi cp1252
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import time
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, roc_curve

# --- [BƯỚC 1] Cấu hình hằng số, đường dẫn và danh sách đặc trưng ---

RANDOM_STATE = 42

# Đường dẫn tương đối tính từ thư mục gốc của repository
TRAIN_DATA_PATH = "data/train.csv"
TEST_DATA_PATH = "data/test.csv"
MODEL_SAVE_DIR = "models"
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, "rf_model.pkl")
REPORT_SAVE_DIR = "reports"
PRED_TEST_PATH = os.path.join(REPORT_SAVE_DIR, "pred_rf.csv")
PRED_TRAIN_PATH = os.path.join(REPORT_SAVE_DIR, "pred_rf_train.csv")
METRICS_JSON_PATH = os.path.join(REPORT_SAVE_DIR, "metrics_rf.json")
FEATURE_IMPORTANCE_PATH = os.path.join(REPORT_SAVE_DIR, "feature_importance_rf.csv")

TARGET_COL = "default"

# Danh sách cột đặc trưng (Feature list) - đặt ở đầu file để dễ dàng tùy chỉnh
# Lưu ý: Không bao gồm cột ID và cột mục tiêu 'default'
FEATURE_COLS = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
]


# --- [BƯỚC 2] Định nghĩa các hàm phụ trợ tính toán và đánh giá ---

def calculate_ks_statistic(y_true, y_prob):
    """
    Tính chỉ số thống kê Kolmogorov-Smirnov (KS) dựa trên đường cong ROC.

    Tham số:
        y_true (array-like): Nhãn thực tế (0 hoặc 1).
        y_prob (array-like): Xác suất dự báo lớp 1 (PD).

    Giá trị trả về:
        ks_stat (float): Giá trị thống kê KS = max(TPR - FPR).
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ks_stat = float(np.max(tpr - fpr))
    return ks_stat


def evaluate_model(y_true, y_prob):
    """
    Tính toán bộ 3 chỉ số thẩm định mô hình tín dụng: AUC-ROC, Gini và KS.

    Tham số:
        y_true (array-like): Nhãn thực tế (0 hoặc 1).
        y_prob (array-like): Xác suất dự báo lớp 1 (PD).

    Giá trị trả về:
        dict: Chứa các giá trị 'auc', 'gini', 'ks'.
    """
    auc = float(roc_auc_score(y_true, y_prob))
    gini = float(2 * auc - 1)
    ks = calculate_ks_statistic(y_true, y_prob)
    return {
        "auc": auc,
        "gini": gini,
        "ks": ks
    }


def load_dataset(train_path, test_path, feature_cols, target_col):
    """
    Đọc dữ liệu train và test từ file CSV, tách đặc trưng X và nhãn y,
    kiểm tra và loại bỏ cột ID nếu có để tránh rò rỉ dữ liệu.

    Tham số:
        train_path (str): Đường dẫn file train.csv.
        test_path (str): Đường dẫn file test.csv.
        feature_cols (list): Danh sách các cột đặc trưng dự kiến.
        target_col (str): Tên cột mục tiêu ('default').

    Giá trị trả về:
        tuple: (X_train, y_train, X_test, y_test, df_train, df_test, final_features)
    """
    print(f"[*] Đang tải dữ liệu từ {train_path} và {test_path}...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Kiểm tra cột ID hoặc id tương tự để loại bỏ nếu có trong tập dữ liệu
    potential_id_cols = [col for col in df_train.columns if col.lower() in ['id', 'customer_id', 'client_id']]
    if potential_id_cols:
        print(f"    [Cảnh báo rò rỉ dữ liệu] Đã phát hiện cột định danh: {potential_id_cols}, loại khỏi đặc trưng.")
        feature_cols = [c for c in feature_cols if c not in potential_id_cols]

    # Kiểm tra tính hợp lệ của các cột đặc trưng
    final_features = [c for c in feature_cols if c in df_train.columns and c != target_col]
    print(f"    -> Số lượng đặc trưng sử dụng: {len(final_features)}")
    print(f"    -> Kích thước tập Train: {df_train.shape} | Tỷ lệ default: {df_train[target_col].mean():.2%}")
    print(f"    -> Kích thước tập Test:  {df_test.shape}  | Tỷ lệ default: {df_test[target_col].mean():.2%}")

    X_train = df_train[final_features]
    y_train = df_train[target_col]
    X_test = df_test[final_features]
    y_test = df_test[target_col]

    return X_train, y_train, X_test, y_test, df_train, df_test, final_features


# --- [BƯỚC 3] Chương trình thực thi chính ---

def main():
    start_total_time = time.time()
    print("=" * 70)
    print(" HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST (CHẤM ĐIỂM TÍN DỤNG)")
    print("=" * 70)

    # 1. Tạo thư mục models/ và reports/ nếu chưa tồn tại
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(REPORT_SAVE_DIR, exist_ok=True)

    # 2. Đọc dữ liệu
    X_train, y_train, X_test, y_test, df_train, df_test, final_features = load_dataset(
        TRAIN_DATA_PATH, TEST_DATA_PATH, FEATURE_COLS, TARGET_COL
    )

    # 3. Cấu hình mô hình Random Forest
    # LƯU Ý VỀ MẤT CÂN BẰNG LỚP:
    # Tỷ lệ vỡ nợ (default=1) trong tập train chiếm khoảng ~22% (thiểu số).
    # Do đó, lựa chọn class_weight='balanced' là cần thiết để Random Forest
    # tự động gán trọng số nghịch đảo với tần suất lớp, giúp mô hình học tốt
    # phân phối của nhóm vỡ nợ và nâng cao khả năng phân tách (AUC/KS).
    base_rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # 4. Thiết lập không gian siêu tham số (Grid) cho GridSearchCV
    # - n_estimators: Số lượng cây trong rừng (100, 200)
    # - max_depth: Giới hạn độ sâu để kiểm soát overfitting và dung lượng file mô hình
    # - min_samples_leaf: Số mẫu tối thiểu ở nút lá, giúp làm mượt dự báo xác suất PD
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [8, 12, 16],
        'min_samples_leaf': [10, 30, 50]
    }

    # 5. Phân tầng 5-fold cross-validation chỉ trên tập train
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("\n--- [BƯỚC 4] Bắt đầu tinh chỉnh siêu tham số với GridSearchCV (5-Fold CV) ---")
    print(f"    -> Không gian tham số: {param_grid}")
    print(f"    -> Chỉ số tối ưu: roc_auc | n_jobs: -1")
    
    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True  # Sau khi chọn tham số tốt nhất, tự động refit trên toàn bộ tập train
    )

    cv_start_time = time.time()
    grid_search.fit(X_train, y_train)
    cv_elapsed = time.time() - cv_start_time
    print(f"[*] Hoàn thành GridSearchCV trong: {cv_elapsed:.2f} giây.")

    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_auc = grid_search.best_score_

    # Lấy độ lệch chuẩn AUC giữa các fold của mô hình tốt nhất
    best_index = grid_search.best_index_
    best_cv_std = float(grid_search.cv_results_['std_test_score'][best_index])

    print("\n--- [BƯỚC 5] Kết quả Cross-Validation của tham số tối ưu ---")
    print(f"    -> Tham số tốt nhất (Best Params): {best_params}")
    print(f"    -> CV AUC trung bình (Mean AUC):  {best_cv_auc:.4f} (± {best_cv_std:.4f})")

    # 6. Dự báo xác suất vỡ nợ (PD) trên tập Train và tập Test
    # Test set CHỈ được dùng 1 lần duy nhất ở đây để đánh giá khách quan
    print("\n--- [BƯỚC 6] Đánh giá mô hình trên tập Train và tập Test ---")
    prob_train = best_rf.predict_proba(X_train)[:, 1]
    prob_test = best_rf.predict_proba(X_test)[:, 1]

    # Tính toán các chỉ số AUC, Gini, KS
    metrics_train = evaluate_model(y_train, prob_train)
    metrics_test = evaluate_model(y_test, prob_test)
    auc_diff = metrics_train["auc"] - metrics_test["auc"]

    # In bảng so sánh đối sánh Train và Test
    print("\n" + "=" * 65)
    print(f"{'Chỉ số (Metric)':<20} | {'Train':<12} | {'Test':<12} | {'Chênh lệch (Train - Test)':<15}")
    print("-" * 65)
    print(f"{'AUC-ROC':<20} | {metrics_train['auc']:<12.4f} | {metrics_test['auc']:<12.4f} | {auc_diff:<15.4f}")
    print(f"{'Gini':<20} | {metrics_train['gini']:<12.4f} | {metrics_test['gini']:<12.4f} | {metrics_train['gini'] - metrics_test['gini']:<15.4f}")
    print(f"{'KS Statistic':<20} | {metrics_train['ks']:<12.4f} | {metrics_test['ks']:<12.4f} | {metrics_train['ks'] - metrics_test['ks']:<15.4f}")
    print("=" * 65)
    print(f"[*] Chênh lệch AUC Train - Test = {auc_diff:.4f}")
    if auc_diff > 0.05:
        print("    [Nhận xét Overfitting] Mức chênh lệch > 0.05, có dấu hiệu overfitting nhẹ.")
    else:
        print("    [Nhận xét Overfitting] Mức chênh lệch <= 0.05, mô hình tổng quát hóa tốt.")

    # 7. Đánh giá Feature Importance (15 đặc trưng quan trọng nhất)
    importances = best_rf.feature_importances_
    df_importance = pd.DataFrame({
        "feature": final_features,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    print("\n--- [BƯỚC 7] Top 15 đặc trưng quan trọng nhất (Feature Importance) ---")
    print(f"{'Hạng':<5} | {'Tên đặc trưng':<20} | {'Mức độ quan trọng (%)':<25}")
    print("-" * 55)
    for idx, row in df_importance.head(15).iterrows():
        print(f"{idx+1:<5} | {row['feature']:<20} | {row['importance']*100:<25.2f}%")
    print("-" * 55)

    # 8. Lưu kết quả ra file phục vụ mục 3.5 và 3.6
    print("\n--- [BƯỚC 8] Xuất file mô hình và báo cáo kết quả ---")

    # 8.1. Lưu mô hình .pkl bằng pickle
    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(best_rf, f)
    model_size_mb = os.path.getsize(MODEL_SAVE_PATH) / (1024 * 1024)
    print(f"    [+] Đã lưu mô hình tại: {MODEL_SAVE_PATH}")
    print(f"    [+] Dung lượng file mô hình: {model_size_mb:.2f} MB")
    if model_size_mb > 90:
        print(f"    [!] CẢNH BÁO: Dung lượng file ({model_size_mb:.2f} MB) vượt ngưỡng an toàn 90 MB!")
        print("        Gợi ý: Hãy giảm n_estimators hoặc max_depth, hoặc tăng min_samples_leaf để tránh bị GitHub từ chối (>100 MB).")
    else:
        print("    [+] Dung lượng an toàn (< 90 MB, phù hợp với giới hạn GitHub).")

    # 8.2. Lưu dự báo xác suất (Predictions CSV)
    # Tập Test
    df_pred_test = pd.DataFrame({
        "test_index": df_test.index,
        "y_true": y_test.values,
        "prob_default": prob_test
    })
    for id_col in [c for c in df_test.columns if c.lower() in ['id', 'customer_id', 'client_id']]:
        df_pred_test.insert(1, id_col, df_test[id_col].values)
    df_pred_test.to_csv(PRED_TEST_PATH, index=False)
    print(f"    [+] Đã lưu dự báo Test tại: {PRED_TEST_PATH}")

    # Tập Train
    df_pred_train = pd.DataFrame({
        "train_index": df_train.index,
        "y_true": y_train.values,
        "prob_default": prob_train
    })
    for id_col in [c for c in df_train.columns if c.lower() in ['id', 'customer_id', 'client_id']]:
        df_pred_train.insert(1, id_col, df_train[id_col].values)
    df_pred_train.to_csv(PRED_TRAIN_PATH, index=False)
    print(f"    [+] Đã lưu dự báo Train tại: {PRED_TRAIN_PATH}")

    # 8.3. Lưu metrics JSON
    metrics_payload = {
        "model_name": "Random Forest",
        "random_state": RANDOM_STATE,
        "best_params": best_params,
        "cv_results": {
            "n_splits": 5,
            "cv_auc_mean": float(best_cv_auc),
            "cv_auc_std": float(best_cv_std)
        },
        "train_metrics": metrics_train,
        "test_metrics": metrics_test,
        "auc_diff": float(auc_diff),
        "model_file_size_mb": float(model_size_mb)
    }
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=4, ensure_ascii=False)
    print(f"    [+] Đã lưu chỉ số đánh giá tại: {METRICS_JSON_PATH}")

    # 8.4. Lưu bảng Feature Importance CSV
    df_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    print(f"    [+] Đã lưu bảng đặc trưng quan trọng tại: {FEATURE_IMPORTANCE_PATH}")

    total_time = time.time() - start_total_time
    print(f"\n[*] Toàn bộ quy trình hoàn tất trong: {total_time:.2f} giây.")
    print("=" * 70)


if __name__ == "__main__":
    main()
