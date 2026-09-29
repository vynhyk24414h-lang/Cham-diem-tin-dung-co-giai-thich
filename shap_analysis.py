# --- Đảm bảo stdout/stderr hỗ trợ UTF-8 trên Windows ---
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

"""
shap_analysis.py
================
Nhóm 4 — Giải thích mô hình (Explainable AI) bằng SHAP.

Nội dung:
    (1) SHAP toàn cục (global): biến nào ảnh hưởng nhiều nhất đến PD của XGBoost?
    (2) SHAP cục bộ (local)   : vì sao MỘT khách hàng cụ thể bị chấm rủi ro cao/thấp?
    (3) Đối chiếu SHAP (XGBoost) với hệ số Logistic Scorecard  -> Mục 3.6 báo cáo.

Chạy độc lập (từ thư mục gốc của repo):
    python shap_analysis.py

Đầu vào:
    data/test.csv, data/woe_train.csv, data/woe_test.csv
    models/xgb_model.pkl, models/logit_scorecard.pkl

Đầu ra (thư mục reports/shap/):
    shap_summary.png            biểu đồ beeswarm (global)   <- sản phẩm bàn giao chính
    shap_bar.png                biểu đồ cột mean|SHAP|
    shap_dependence_top1.png    biến quan trọng nhất: giá trị biến -> SHAP
    shap_local_high_risk.png    waterfall: khách rủi ro cao (vỡ nợ thật)
    shap_local_low_risk.png     waterfall: khách rủi ro thấp (không vỡ nợ)
    shap_local_borderline.png   waterfall: khách sát ngưỡng phê duyệt
    shap_local_cases.csv        số liệu chi tiết 3 khách hàng minh họa
    shap_importance_xgb.csv     bảng xếp hạng mean|SHAP| của mọi biến
    shap_vs_scorecard.csv       bảng đối chiếu SHAP vs Logistic (Mục 3.6)
    shap_vs_scorecard.png       biểu đồ đối chiếu SHAP vs Logistic
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                # vẽ ra file, không mở cửa sổ
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ==============================================================================
# --- [HẰNG SỐ] Cấu hình đường dẫn và tham số ---
# ==============================================================================
RANDOM_STATE   = 42
TARGET_COL     = "default"          # 1 = vỡ nợ, 0 = không vỡ nợ
ID_COL         = "ID"               # cột định danh (nếu có)
PROTECTED      = ["SEX", "AGE"]     # biến nhạy cảm (Logistic đã loại, XGBoost có thể vẫn dùng)
NOMINAL        = ["SEX", "EDUCATION", "MARRIAGE"]   # biến DANH MỤC: mã số 1,2,3.. không có thứ tự thật
                                                    # (PAY_x là biến thứ bậc nên vẫn đo chiều bình thường)

TEST_PATH      = "data/test.csv"
WOE_TRAIN_PATH = "data/woe_train.csv"
WOE_TEST_PATH  = "data/woe_test.csv"
XGB_PATH       = "models/xgb_model.pkl"
LOGIT_PATH     = "models/logit_scorecard.pkl"
OUT_DIR        = "reports/shap"

MAX_ROWS       = 6000   # nếu test lớn hơn số này thì lấy mẫu ngẫu nhiên cho nhanh
THRESHOLD      = 0.5    # ngưỡng chọn ca "sát ngưỡng" (ngưỡng chính thức do Nhóm 5 đề xuất ở Mục 4.3)
TOP_N          = 15     # số biến hiển thị trên biểu đồ


# ==============================================================================
# --- [HÀM PHỤ] Lưu hình ---
# ==============================================================================
def save_fig(filename: str) -> None:
    """Lưu hình matplotlib hiện tại vào reports/shap/ rồi đóng hình."""
    path = os.path.join(OUT_DIR, filename)
    plt.gcf().savefig(path, dpi=200, bbox_inches="tight")
    plt.close("all")
    print(f"  Đã lưu hình → {path}")


# ==============================================================================
# --- [BƯỚC 1] Đọc dữ liệu và mô hình ---
# ==============================================================================
def load_inputs():
    """
    Đọc tập test (biến thô), tập WoE, mô hình XGBoost.

    Lưu ý quan trọng:
        - XGBoost được huấn luyện trên BIẾN THÔ (data/train.csv).
        - Logistic Scorecard được huấn luyện trên BIẾN WoE (data/woe_train.csv).
        -> SHAP tính trên biến thô; hệ số Logistic tính trên biến WoE.
           Ở Bước 5 ta đưa cả hai về cùng "tên biến gốc" để so sánh.

    Giá trị trả về:
        model, features, X (biến thô của mẫu), y, idx (chỉ số dòng của mẫu),
        woe_train, woe_test
    """
    test      = pd.read_csv(TEST_PATH)
    woe_train = pd.read_csv(WOE_TRAIN_PATH)
    woe_test  = pd.read_csv(WOE_TEST_PATH)
    model     = joblib.load(XGB_PATH)

    # Danh sách biến mà XGBoost thực sự đã dùng khi huấn luyện
    features = list(model.feature_names_in_)

    # test.csv và woe_test.csv phải cùng số dòng, cùng thứ tự (cùng khách hàng)
    assert len(test) == len(woe_test), "test.csv và woe_test.csv khác số dòng!"

    # Lấy mẫu nếu tập test quá lớn (SHAP chạy nhanh, nhưng biểu đồ 30.000 điểm rất nặng)
    if len(test) > MAX_ROWS:
        idx = test.sample(MAX_ROWS, random_state=RANDOM_STATE).index
    else:
        idx = test.index

    X = test.loc[idx, features]
    y = test.loc[idx, TARGET_COL]

    print(f"  Số dòng dùng để tính SHAP: {len(X):,} (tỷ lệ vỡ nợ: {y.mean()*100:.2f}%)")
    print(f"  Số biến XGBoost sử dụng  : {len(features)}")

    # --- Cảnh báo về các biến đáng chú ý trong mô hình XGBoost ---
    if ID_COL in features:
        print(f"  ⚠️ XGBoost đang dùng cột '{ID_COL}' làm đặc trưng — cần báo Nhóm 3 loại bỏ.")
    used_protected = [c for c in PROTECTED if c in features]
    if used_protected:
        print(f"  ⚠️ XGBoost đang dùng biến nhạy cảm: {used_protected} "
              f"(Logistic đã loại) — cần nêu trong báo cáo công bằng.")
    return model, features, X, y, idx, woe_train, woe_test


# ==============================================================================
# --- [BƯỚC 2] Tính giá trị SHAP ---
# ==============================================================================
def compute_shap(model, X: pd.DataFrame):
    """
    Tính SHAP cho mô hình XGBoost bằng TreeExplainer (chính xác, nhanh với mô hình cây).

    Ý nghĩa của giá trị SHAP:
        - Mỗi khách hàng, mỗi biến có 1 giá trị SHAP.
        - SHAP > 0: biến đó ĐẨY xác suất vỡ nợ LÊN; SHAP < 0: KÉO xuống.
        - Đơn vị là log-odds của mô hình XGBoost (không phải % xác suất).
        - Công thức cộng: base_value + tổng SHAP các biến = log-odds dự báo của khách đó.

    Nếu TreeExplainer của thư viện shap lỗi (do khác phiên bản xgboost), dùng cách
    dự phòng: chính XGBoost tự tính SHAP qua pred_contribs=True (kết quả tương đương).

    Giá trị trả về:
        exp (shap.Explanation), values (ndarray n x p)
    """
    try:
        raw = shap.TreeExplainer(model)(X)
        values = np.asarray(raw.values, dtype=float)
        base   = np.asarray(raw.base_values, dtype=float)
        method = "shap.TreeExplainer"
    except Exception as e:
        print(f"  ⚠️ TreeExplainer lỗi ({type(e).__name__}), chuyển sang pred_contribs.")
        contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        values, base = contrib[:, :-1], contrib[:, -1]   # cột cuối = base value
        method = "XGBoost pred_contribs"

    base = np.broadcast_to(base, (len(X),)).astype(float)   # ép base về vector n phần tử
    exp = shap.Explanation(values=values, base_values=base,
                           data=X.values, feature_names=list(X.columns))

    # --- Kiểm tra tính đúng đắn: base + tổng SHAP phải bằng log-odds mô hình dự báo ---
    margin = model.predict(X, output_margin=True)
    err = float(np.max(np.abs(values.sum(axis=1) + base - margin)))
    print(f"  Phương pháp: {method}")
    print(f"  Kiểm tra cộng dồn (base + ΣSHAP vs log-odds mô hình): sai số lớn nhất = {err:.2e}")
    return exp, values


# ==============================================================================
# --- [BƯỚC 3] SHAP toàn cục (global) ---
# ==============================================================================
def global_shap(exp, values, X: pd.DataFrame) -> pd.DataFrame:
    """
    Vẽ và lưu các biểu đồ toàn cục:
        - Beeswarm (shap_summary.png): mỗi chấm = 1 khách hàng.
            Trục ngang: SHAP (sang phải = tăng rủi ro vỡ nợ).
            Màu: giá trị của biến (đỏ = cao, xanh = thấp).
        - Bar (shap_bar.png): trung bình |SHAP| = mức độ quan trọng tổng thể.
        - Dependence (shap_dependence_top1.png): giá trị biến quan trọng nhất -> SHAP.

    Giá trị trả về:
        DataFrame xếp hạng biến theo mean|SHAP|
    """
    imp = pd.DataFrame({
        "feature"      : X.columns,
        "mean_abs_shap": np.abs(values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["share_pct"] = 100 * imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp["rank"]      = imp.index + 1
    imp.to_csv(os.path.join(OUT_DIR, "shap_importance_xgb.csv"), index=False)

    # Beeswarm
    shap.plots.beeswarm(exp, max_display=TOP_N, show=False)
    plt.title("SHAP toàn cục — XGBoost (beeswarm)")
    save_fig("shap_summary.png")

    # Bar
    shap.plots.bar(exp, max_display=TOP_N, show=False)
    plt.title("Mức độ quan trọng trung bình |SHAP| — XGBoost")
    save_fig("shap_bar.png")

    # Dependence của biến quan trọng nhất
    top1 = imp.loc[0, "feature"]
    shap.plots.scatter(exp[:, top1], show=False)
    plt.title(f"Quan hệ giá trị {top1} và SHAP")
    save_fig("shap_dependence_top1.png")

    print(f"\n  Top 10 biến theo mean|SHAP|:")
    print(imp.head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return imp


# ==============================================================================
# --- [BƯỚC 4] SHAP cục bộ (local) ---
# ==============================================================================
def local_shap(model, exp, X: pd.DataFrame, y: pd.Series) -> None:
    """
    Giải thích 3 khách hàng điển hình bằng waterfall:
        1. Rủi ro cao  : khách vỡ nợ thật và mô hình cho PD cao nhất.
        2. Rủi ro thấp : khách không vỡ nợ và mô hình cho PD thấp nhất.
        3. Sát ngưỡng  : khách có PD gần THRESHOLD nhất (ca khó quyết định nhất).

    Waterfall đọc từ dưới lên: bắt đầu từ E[f(x)] (log-odds trung bình),
    mỗi thanh là đóng góp của 1 biến, kết thúc ở f(x) (log-odds của khách đó).
    """
    prob   = model.predict_proba(X)[:, 1]
    y_arr  = y.values
    pos    = np.where(y_arr == 1)[0]
    neg    = np.where(y_arr == 0)[0]

    cases = {
        "high_risk"  : int(pos[np.argmax(prob[pos])]),
        "low_risk"   : int(neg[np.argmin(prob[neg])]),
        "borderline" : int(np.argmin(np.abs(prob - THRESHOLD))),
    }

    rows = []
    for name, i in cases.items():
        shap.plots.waterfall(exp[i], max_display=10, show=False)
        plt.title(f"Khách hàng {name} — PD dự báo = {prob[i]:.3f}, nhãn thật = {y_arr[i]}")
        save_fig(f"shap_local_{name}.png")

        # Lưu 5 biến đóng góp lớn nhất của khách này (để viết diễn giải trong báo cáo)
        order = np.argsort(-np.abs(exp.values[i]))[:5]
        for r, j in enumerate(order, start=1):
            rows.append({
                "case": name, "row_index": X.index[i], "pd_xgb": round(float(prob[i]), 4),
                "y_true": int(y_arr[i]), "top": r,
                "feature": X.columns[j], "value": X.iloc[i, j],
                "shap": round(float(exp.values[i, j]), 4),
            })
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "shap_local_cases.csv"), index=False)
    print("  Đã lưu → reports/shap/shap_local_cases.csv")


# ==============================================================================
# --- [BƯỚC 5] Đối chiếu SHAP với hệ số Logistic Scorecard (Mục 3.6) ---
# ==============================================================================
def load_logit_params(path: str) -> pd.Series:
    """
    Đọc hệ số β của Logistic Scorecard từ file .pkl (bỏ hệ số chặn 'const').

    File logit_scorecard.pkl của nhóm là một dict gồm các khóa:
        'features', 'params' (hệ số β), 'pvalues', 'cfg', 'factor', 'offset', 'band_edges'...
    Hàm cũng tự nhận dạng thêm trường hợp file lưu trực tiếp kết quả statsmodels.
    """
    obj = joblib.load(path)
    if isinstance(obj, dict) and "params" in obj:
        params = pd.Series(obj["params"], dtype=float)
    elif hasattr(obj, "params") and isinstance(obj.params, pd.Series):
        params = obj.params
    else:
        raise RuntimeError(
            f"Không đọc được hệ số từ {path} (kiểu: {type(obj)}). "
            "Hãy gửi kết quả print(type(obj)) để chỉnh hàm này.")
    return params.drop("const", errors="ignore")


def to_raw_name(col: str, raw_cols) -> str:
    """Đổi tên cột WoE (VD 'PAY_0_woe') về tên biến gốc ('PAY_0') để ghép với SHAP."""
    if col in raw_cols:
        return col
    for suf in ("_woe", "_WOE", "_Woe"):
        if col.endswith(suf) and col[:-len(suf)] in raw_cols:
            return col[:-len(suf)]
    for pre in ("woe_", "WOE_"):
        if col.startswith(pre) and col[len(pre):] in raw_cols:
            return col[len(pre):]
    return col


def compare_with_scorecard(values, X, idx, woe_train, woe_test) -> None:
    """
    So sánh mức độ quan trọng và CHIỀU tác động giữa SHAP (XGBoost) và Logistic.

    Cách quy đổi Logistic về cùng thang với SHAP:
        Đóng góp của biến j cho khách i vào log-odds = β_j × (WoE_ij − trung bình WoE_j).
        Đây chính là SHAP của mô hình tuyến tính (khi biến độc lập), nên so sánh được
        trực tiếp với SHAP của XGBoost.

    Vì hai mô hình có thang log-odds khác nhau (XGBoost dùng scale_pos_weight),
    ta KHÔNG so độ lớn tuyệt đối mà so:
        (a) tỷ trọng % trong tổng |đóng góp| của các biến chung,
        (b) thứ hạng (Spearman) và top-5,
        (c) chiều tác động: tương quan Spearman giữa giá trị biến gốc và đóng góp.
            (cách này không phụ thuộc quy ước dấu của WoE)
            CHỈ áp dụng cho biến số/thứ bậc. Với biến DANH MỤC (NOMINAL) mã số không có
            thứ tự nên không đo "chiều"; thay vào đó lập bảng đóng góp THEO TỪNG NHÓM.
    """
    params    = load_logit_params(LOGIT_PATH)
    woe_cols  = list(params.index)
    raw_cols  = set(X.columns)
    raw_names = [to_raw_name(c, raw_cols) for c in woe_cols]

    print("  Biến trong Logistic Scorecard (cột WoE → biến gốc, hệ số β):")
    for c, r, b in zip(woe_cols, raw_names, params.values):
        flag = "" if r in raw_cols else "   ⚠️ không khớp tên biến gốc"
        print(f"    {c:<20} → {r:<14} β = {b:+.4f}{flag}")

    # Đóng góp của từng biến trong Logistic (cùng các khách hàng như mẫu SHAP)
    mu = woe_train[woe_cols].mean()
    contrib = (woe_test.loc[idx, woe_cols] - mu) * params
    contrib.columns = raw_names

    shap_df = pd.DataFrame(values, columns=X.columns, index=X.index)

    # Chỉ so sánh các biến có mặt ở CẢ HAI mô hình
    shared = [c for c in contrib.columns if c in shap_df.columns]
    if len(shared) < 3:
        print("  ⚠️ Quá ít biến chung để đối chiếu — kiểm tra lại tên biến.")
        return

    shap_abs  = shap_df[shared].abs().mean()
    logit_abs = contrib[shared].abs().mean()

    table = pd.DataFrame({
        "feature"          : shared,
        "beta_logit"       : [params.iloc[raw_names.index(f)] for f in shared],
        "shap_mean_abs"    : shap_abs.values,
        "logit_mean_abs"   : logit_abs.values,
        "shap_share_pct"   : 100 * shap_abs.values / shap_abs.sum(),
        "logit_share_pct"  : 100 * logit_abs.values / logit_abs.sum(),
    })
    table["shap_rank"]  = table["shap_share_pct"].rank(ascending=False).astype(int)
    table["logit_rank"] = table["logit_share_pct"].rank(ascending=False).astype(int)

    # Chiều tác động: tăng giá trị biến gốc thì rủi ro tăng (+) hay giảm (−)?
    def direction(x, contribution):
        rho = spearmanr(x, contribution).correlation
        return 0 if (rho is None or np.isnan(rho)) else int(np.sign(rho))

    is_nom = table["feature"].isin(NOMINAL)
    table["dir_shap"]  = [np.nan if n else direction(X[f], shap_df[f])
                          for f, n in zip(table["feature"], is_nom)]
    table["dir_logit"] = [np.nan if n else direction(X[f], contrib[f])
                          for f, n in zip(table["feature"], is_nom)]
    table["cung_chieu"] = np.where(is_nom, "Không áp dụng (biến danh mục)",
                          np.where(table["dir_shap"] == table["dir_logit"], "Có", "Không"))

    table = table.sort_values("shap_rank").reset_index(drop=True)
    table.to_csv(os.path.join(OUT_DIR, "shap_vs_scorecard.csv"), index=False)
    print("  Đã lưu → reports/shap/shap_vs_scorecard.csv")

    # --- Chỉ số tóm tắt cho báo cáo ---
    rho = spearmanr(table["shap_share_pct"], table["logit_share_pct"]).correlation
    top5_shap  = set(shap_df.abs().mean().sort_values(ascending=False).head(5).index)
    top5_logit = set(contrib.abs().mean().sort_values(ascending=False).head(5).index)
    n_same_dir = int((table["cung_chieu"] == "Có").sum())
    n_dir_total = int((~is_nom.values).sum())          # số biến số/thứ bậc được đo chiều

    print(f"\n  Số biến chung             : {len(shared)}")
    print(f"  Spearman thứ hạng (SHAP vs Logistic): {rho:.3f}")
    print(f"  Trùng top-5 biến quan trọng: {len(top5_shap & top5_logit)}/5 -> {sorted(top5_shap & top5_logit)}")
    print(f"  Cùng chiều tác động (biến số/thứ bậc): {n_same_dir}/{n_dir_total} biến")

    only_xgb = [c for c in shap_df.columns if c not in shared]
    if only_xgb:
        share_only = shap_df[only_xgb].abs().mean().sum() / shap_df.abs().mean().sum() * 100
        print(f"  Biến chỉ XGBoost dùng (Logistic đã loại do VIF/tương quan/p-value): "
              f"{len(only_xgb)} biến, chiếm {share_only:.1f}% tổng |SHAP|")

    # --- Biến danh mục: đóng góp trung bình THEO TỪNG NHÓM (thay cho "chiều tác động") ---
    cat_rows = []
    for f in [c for c in shared if c in NOMINAL]:
        g = pd.DataFrame({"nhom": X[f].values,
                          "shap": shap_df[f].values,
                          "logit": contrib[f].values}).groupby("nhom")
        for k, d in g:
            cat_rows.append({"feature": f, "nhom": k, "so_khach": len(d),
                             "ty_le_pct": round(100 * len(d) / len(X), 2),
                             "shap_tb_xgb": round(d["shap"].mean(), 4),
                             "dong_gop_tb_logit": round(d["logit"].mean(), 4)})
    if cat_rows:
        cat_df = pd.DataFrame(cat_rows)
        cat_df.to_csv(os.path.join(OUT_DIR, "shap_vs_scorecard_categorical.csv"), index=False)
        print("\n  Biến danh mục — đóng góp trung bình theo nhóm (log-odds):")
        print(cat_df.to_string(index=False))
        print("  Đã lưu → reports/shap/shap_vs_scorecard_categorical.csv")

    # --- Biểu đồ so sánh tỷ trọng ---
    top = table.head(TOP_N).iloc[::-1]
    pos = np.arange(len(top))
    plt.figure(figsize=(9, 0.45 * len(top) + 1.5))
    plt.barh(pos + 0.2, top["shap_share_pct"],  height=0.4, label="SHAP (XGBoost)")
    plt.barh(pos - 0.2, top["logit_share_pct"], height=0.4, label="Logistic Scorecard")
    plt.yticks(pos, top["feature"])
    plt.xlabel("Tỷ trọng trong tổng |đóng góp| của các biến chung (%)")
    plt.title("Đối chiếu mức độ quan trọng: SHAP (XGBoost) vs Logistic Scorecard")
    plt.legend()
    save_fig("shap_vs_scorecard.png")


# ==============================================================================
# --- MAIN ---
# ==============================================================================
def main():
    """Hàm chính: đọc → tính SHAP → global → local → đối chiếu Logistic."""
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  PHÂN TÍCH SHAP — Nhóm 4: Giải thích mô hình")
    print("=" * 60)

    print("\n[BƯỚC 1] Đọc dữ liệu và mô hình...")
    model, features, X, y, idx, woe_train, woe_test = load_inputs()

    print("\n[BƯỚC 2] Tính giá trị SHAP...")
    exp, values = compute_shap(model, X)

    print("\n[BƯỚC 3] SHAP toàn cục (global)...")
    global_shap(exp, values, X)

    print("\n[BƯỚC 4] SHAP cục bộ (local)...")
    local_shap(model, exp, X, y)

    print("\n[BƯỚC 5] Đối chiếu với Logistic Scorecard (Mục 3.6)...")
    compare_with_scorecard(values, X, idx, woe_train, woe_test)

    print("\n" + "=" * 60)
    print("  Hoàn tất! Kết quả nằm trong reports/shap/")
    print("=" * 60)


if __name__ == "__main__":
    main()