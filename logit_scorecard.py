# %%
# ==============================================================================
# BƯỚC 0: KHAI BÁO THƯ VIỆN & CẤU HÌNH THANG ĐIỂM (CFG)
# ==============================================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42

# Thiết lập các tham số cố định theo tiêu chuẩn Scorecard
CFG = dict(
    target="default",          # Tên cột mục tiêu (1 = Vỡ nợ, 0 = Tốt)
    id_col="ID",               # Cột định danh khách hàng
    base_score=600,            # Điểm cơ sở tại tỷ lệ odds cơ sở
    base_odds=20,              # Tỷ lệ Good/Bad cơ sở (20 khách tốt mới có 1 khách xấu)
    pdo=40,                    # Points to Double the Odds (+40 điểm thì odds tốt/xấu tăng gấp đôi)
    score_min=300, score_max=850, # Cận trên và cận dưới của thang điểm
    corr_max=0.70,             # Ngưỡng tương quan tối đa giữa 2 biến
    vif_max=5.0,               # Ngưỡng VIF tối đa để loại đa cộng tuyến
    protected=["SEX", "AGE"]   # Biến nhạy cảm -> LOẠI KHỎI mô hình chính (để đảm bảo công bằng)
)

# Nạp dữ liệu đã biến đổi WoE từ Nhóm 2
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
woe_train = pd.read_csv('data/woe_train.csv')
woe_test = pd.read_csv('data/woe_test.csv')

y_tr = train[CFG["target"]].to_numpy()
y_te = test[CFG["target"]].to_numpy()

# %%
# ==============================================================================
# BƯỚC 1: LỌC BIẾN & LÀM SẠCH ĐA CỘNG TUYẾN (VIF & CORRELATION)
# ==============================================================================

# 1.1 Hàm tính chỉ số VIF (Variance Inflation Factor)
def vif_table(X):
    Xc = sm.add_constant(X)
    return pd.DataFrame({
        "feature": X.columns,
        "VIF": [variance_inflation_factor(Xc.values, i + 1) for i in range(X.shape[1])]
    }).sort_values("VIF", ascending=False)

# 1.2 Hàm lọc các cặp biến có tương quan quá cao (|r| > 0.70)
def drop_by_corr(X, thr=0.70):
    corr = X.corr().abs()
    keep = list(X.columns)
    for i, a in enumerate(X.columns):
        for b in X.columns[i + 1:]:
            if a in keep and b in keep and corr.loc[a, b] > thr:
                keep.remove(b) # Bỏ biến thứ hai trong cặp tương quan cao
    return keep

# Tạo danh sách biến ứng viên ban đầu (loại bỏ ID, target, SEX, AGE)
candidates = [c for c in woe_train.columns if c not in [CFG["target"], CFG["id_col"]] + CFG["protected"]]

# BƯỚC LỌC 1: Loại biến tương quan cao
candidates = drop_by_corr(woe_train[candidates], CFG["corr_max"])

# BƯỚC LỌC 2: Loại dần biến có VIF cao nhất cho đến khi tất cả VIF <= 5.0
feats = candidates.copy()
while True:
    v = vif_table(woe_train[feats])
    if v["VIF"].max() <= CFG["vif_max"]:
        break
    # Bỏ biến có VIF lớn nhất ở đầu bảng
    feats.remove(v.iloc[0]["feature"])

print("=== DANH SÁCH BIẾN ĐẠT CHUẨN (VIF <= 5.0) ===")
print(vif_table(woe_train[feats]))

# %%
# ==============================================================================
# BƯỚC 2: HUẤN LUYỆN LOGISTIC REGRESSION & KIỂM TRẢ DẤU / P-VALUE
# ==============================================================================

# Hàm huấn luyện mô hình Logistic bằng statsmodels
def fit_logit(X, y):
    Xc = sm.add_constant(X)
    return sm.Logit(y, Xc).fit(disp=0, maxiter=200)

# Vòng lặp kiểm tra giả định hệ số
# Quy ước WoE của Nhóm 2: WoE = ln(%Good / %Bad) -> Hệ số beta BẮT BUỘC PHẢI ÂM (-1)
EXPECTED_SIGN = -1 

while True:
    res = fit_logit(woe_train[feats], y_tr)
    pvals = res.pvalues.drop("const")
    betas = res.params.drop("const")
    
    # Tìm các biến vi phạm điều kiện
    bad_pvalues = pvals[pvals > 0.05]                          # Biến không có ý nghĩa thống kê
    bad_signs = betas[np.sign(betas) != EXPECTED_SIGN]         # Biến bị sai dấu kinh tế
    
    # Nếu tất cả biến đều p <= 0.05 và đúng dấu -> DỪNG VÒNG LẶP
    if len(bad_pvalues) == 0 and len(bad_signs) == 0:
        break
        
    # Nếu có vi phạm, ưu tiên loại biến có p-value cao nhất trước
    if len(bad_pvalues) > 0:
        remove_var = bad_pvalues.idxmax()
    else:
        remove_var = bad_signs.index[0]
        
    print(f"Loại bỏ biến {remove_var} do p-value cao hoặc sai dấu hệ số.")
    feats.remove(remove_var)

print("\n=== BẢNG KẾT QUẢ MÔ HÌNH LOGISTIC HOÀN CHỈNH (MỤC 3.3 BÁO CÁO) ===")
print(res.summary())

# %%
# ==============================================================================
# BƯỚC 3: QUY ĐỔI THANG ĐIỂM (SCORECARD SCALING) & PHÂN BAND RỦI RO (A-E)
# ==============================================================================

# 3.1 Tính hằng số Factor và Offset
FACTOR = CFG["pdo"] / np.log(2)
OFFSET = CFG["base_score"] - FACTOR * np.log(CFG["base_odds"])

print(f"\nHằng số Factor: {FACTOR:.3f} | Offset: {OFFSET:.3f}")

# 3.2 Hàm đổi Xác suất vỡ nợ (PD) sang Điểm số (Score)
def score_from_pd(pd_val):
    pd_val = np.clip(pd_val, 1e-6, 1 - 1e-6) # Tránh lỗi log(0)
    # Công thức: Score = Offset - Factor * ln(PD / (1 - PD))
    s = OFFSET - FACTOR * np.log(pd_val / (1 - pd_val))
    return np.clip(s, CFG["score_min"], CFG["score_max"])

# Tính PD và Score cho tập Train và Test
p_tr = res.predict(sm.add_constant(woe_train[feats]))
s_tr = score_from_pd(p_tr)

p_te = res.predict(sm.add_constant(woe_test[feats]))
s_te = score_from_pd(p_te)

# 3.3 Phân 5 Band Rủi ro (A -> E) dựa trên phân vị (Quantiles) của tập TRAIN
BAND_LABELS = ["E", "D", "C", "B", "A"] # E: Rủi ro cao nhất, A: An toàn nhất
Q = [0, .10, .30, .70, .90, 1.0]        # Tỷ lệ chia band: 10% - 20% - 40% - 20% - 10%

# Lấy ngưỡng điểm chia band từ tập Train
edges = np.quantile(s_tr, Q)
edges[0], edges[-1] = -np.inf, np.inf # Đảm bảo bao phủ toàn bộ khoảng điểm

def to_band(score):
    return pd.cut(score, bins=edges, labels=BAND_LABELS, include_lowest=True)

# Kiểm tra tính ĐƠN ĐIỆU (Monotonicity) của Bad Rate trên tập Test
df_band_test = pd.DataFrame({"band": to_band(s_te), "y": y_te})
print("\n=== KIỂM TRA BAD RATE THEO BAND TRÊN TẬP TEST ===")
print(df_band_test.groupby("band", observed=True)["y"].agg(SoLuong="size", BadRate="mean"))

# %%
import joblib

rf_model = joblib.load("models/rf_model.pkl")

print(type(rf_model))

# %%
from sklearn.metrics import roc_auc_score

# Tách biến đầu vào và biến mục tiêu
features_rf = list(rf_model.feature_names_in_)

X_train_rf = train[features_rf]
y_train_rf = train["default"]

X_test_rf = test[features_rf]
y_test_rf = test["default"]

# Dự báo bằng mô hình Random Forest đã lưu
prob_train_rf = rf_model.predict_proba(X_train_rf)[:, 1]
prob_test_rf = rf_model.predict_proba(X_test_rf)[:, 1]

# Tính AUC
auc_train_rf = roc_auc_score(y_train_rf, prob_train_rf)
auc_test_rf = roc_auc_score(y_test_rf, prob_test_rf)

print(f"Random Forest Train AUC: {auc_train_rf:.4f}")
print(f"Random Forest Test AUC:  {auc_test_rf:.4f}")
print(f"AUC Gap: {auc_train_rf - auc_test_rf:.4f}")

# %%
import xgboost as xgb

# %%
import joblib

xgb_model = joblib.load("models/xgb_model.pkl")
print(type(xgb_model))

# %%
from sklearn.metrics import roc_auc_score
# Kiểm tra các biến đầu vào của XGBoost
features_xgb = list(xgb_model.feature_names_in_)

print("Số biến đầu vào:", len(features_xgb))
print("Các biến:", features_xgb)

# %%
# Lấy đúng dữ liệu đầu vào
X_train_xgb = train[features_xgb]
y_train_xgb = train["default"]

X_test_xgb = test[features_xgb]
y_test_xgb = test["default"]

# Dự báo bằng mô hình đã lưu
prob_train_xgb = xgb_model.predict_proba(X_train_xgb)[:, 1]
prob_test_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]

# Tính AUC
auc_train_xgb = roc_auc_score(y_train_xgb, prob_train_xgb)
auc_test_xgb = roc_auc_score(y_test_xgb, prob_test_xgb)

print(f"XGBoost Train AUC: {auc_train_xgb:.4f}")
print(f"XGBoost Test AUC:  {auc_test_xgb:.4f}")
print(f"AUC Gap: {auc_train_xgb - auc_test_xgb:.4f}")

# %%
print("Logistic Scorecard")
print("Train AUC:", roc_auc_score(y_tr, p_tr))
print("Test AUC :", roc_auc_score(y_te, p_te))
print("AUC Gap  :", roc_auc_score(y_tr, p_tr) - roc_auc_score(y_te, p_te))

# %%
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# Lấy các biến độc lập trong mô hình Logistic cuối cùng
logit_vars = [var for var in res.params.index if var != "const"]

# Tạo dữ liệu chỉ gồm 11 biến này
X_vif_final = woe_train[logit_vars].copy()

# Tính VIF
vif_final = pd.DataFrame({
    "Variable": X_vif_final.columns,
    "VIF": [
        variance_inflation_factor(X_vif_final.values, i)
        for i in range(X_vif_final.shape[1])
    ]
})

vif_final = vif_final.sort_values("VIF", ascending=False)

print(vif_final.to_string(index=False))


