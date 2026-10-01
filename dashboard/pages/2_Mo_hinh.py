"""
dashboard/pages/2_Mo_hinh.py
=================================================================================
Trang "Mô hình & Chấm điểm khách hàng" 
 
Trang này chỉ ĐỌC LẠI những gì cả nhóm đã tính sẵn (model, metrics, predictions),
không train lại bất cứ mô hình nào — giữ đúng tinh thần "kết quả đã có, dashboard
chỉ kể lại một cách dễ hiểu".
 
-------------------------------------------------------------------------------
CÁC FILE TRANG NÀY SẼ DÙNG :
 
  data/
    train.csv, test.csv          -> giá trị GỐC của khách hàng (chưa đổi WoE)
    woe_train.csv, woe_test.csv  -> giá trị đã đổi sang WoE, CÙNG THỨ TỰ HÀNG với train/test.csv
 
  models/
    logit_scorecard.pkl          -> artifact scorecard (features, hệ số, factor, offset, band...)
    rf_model.pkl, xgb_model.pkl  -> model sklearn/xgboost đã .fit() xong
 
  reports/metrics/
    model_comparison.csv         -> bảng AUC/Gini/KS tổng hợp 3 mô hình (STT 14)
    baseline_metrics.json        -> mô hình đối chứng (STT 13) — hiển thị tham khảo
    scorecard_metrics.json       -> chỉ số riêng của scorecard (nếu có thêm chi tiết)
 
  reports/tables/
    pred_rf.csv, pred_rf_train.csv     -> dự đoán Random Forest trên test / train
    pred_xgb.csv, pred_xgb_train.csv   -> dự đoán XGBoost trên test / train
    roc_comparison.png                 -> hình ROC tĩnh nhóm đã xuất trước (đối chiếu nhanh)
 
=================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
import plotly.graph_objects as go
 
st.set_page_config(page_title="Mô hình & Chấm điểm", page_icon="🤖", layout="wide")

 # ---------------------------------------------------------------------------
# TÙY CHỈNH GIAO DIỆN
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Thu gọn khoảng trắng hai bên và phía trên */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }

    /* Tăng độ rõ của chữ nội dung */
    .stApp {
        color: #263238;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        font-size: 1.3rem;
        line-height: 1.6;
        color: #263238;
    }

    /* Tiêu đề chính */
    h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #17324D !important;
        margin-bottom: 0.8rem !important;
    }

    h2 {
        font-size: 1.5rem !important;
        font-weight: 650 !important;
        color: #17324D !important;
    }

    h3 {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #244B64 !important;
    }

    /* Khoảng cách giữa các thành phần */
    div[data-testid="stVerticalBlock"] {
        gap: 0.8rem;
    }

    /* Viền nhẹ cho các khung thông báo */
    div[data-testid="stAlert"] {
        border-radius: 8px;
    }

    /* Chữ trong bảng dễ đọc hơn */
    [data-testid="stDataFrame"] {
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)
# ---------------------------------------------------------------------------
# CẤU HÌNH CHUNG — gom hết mọi thứ hay phải sửa vào đây, đỡ phải lục cả file
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
MODELS_DIR = Path("models")
METRICS_DIR = Path("reports") / "metrics"
REPORTS_DIR = Path("reports")
TABLES_DIR = REPORTS_DIR / "tables"
 
TARGET_COL = "default"        # SỬA nếu Clean.py đặt tên cột nhãn khác
 

    # Khai báo tên cột thực tế trong các file dự đoán
PRED_COLS = {
    "rf":  {"y_true": "y_true", "y_proba": "prob_default"},
    "xgb": {"y_true": "y_true", "y_proba": "prob_default"},
}

 
# Một màu cố định cho một mô hình — dùng xuyên suốt mọi bảng/biểu đồ trong trang,
# để người xem không phải "học lại" ký hiệu mỗi khi đổi tab
PALETTE = {
    "Logistic Scorecard": "#C9A227",
    "Random Forest": "#2E5266",
    "XGBoost": "#8C3B3B",
}
MODEL_ORDER = ["Logistic Scorecard", "Random Forest", "XGBoost"]
BAND_COLOR = {"A": "#2E7D32", "B": "#66BB6A", "C": "#FBC02D", "D": "#F57C00", "E": "#C62828"}
 
 
# ---------------------------------------------------------------------------
# TIỆN ÍCH NHỎ: tự dò tên cột nhãn / xác suất trong file predictions
# (vì mỗi bạn code một kiểu, tên cột có thể không giống nhau 100%)
# ---------------------------------------------------------------------------
def _detect_col(df: pd.DataFrame, candidates: list[str]) -> str:
    """Trả về tên cột đầu tiên trong candidates mà df thực sự có (không phân biệt hoa/thường)."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise KeyError(
        f"Không tìm thấy cột nào khớp trong {list(df.columns)}. "
        f"Đã thử các tên: {candidates}. Hãy khai báo cứng trong PRED_COLS ở đầu file nhé."
    )
 
 
def _read_pred(model_key: str, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Đọc file pred_*.csv, trả về (y_true, y_proba). model_key = 'rf' hoặc 'xgb'."""
    df = pd.read_csv(path)
    if model_key in PRED_COLS:
        y_col = PRED_COLS[model_key]["y_true"]
        p_col = PRED_COLS[model_key]["y_proba"]
    else:
        y_col = _detect_col(df, ["y_true", "y", "target", TARGET_COL, "label"])
        p_col = _detect_col(df, ["y_proba", "y_pred_proba", "proba", "prob", "pred_proba", "probability", "score_pd"])
    return df[y_col].to_numpy(), df[p_col].to_numpy()
 
 
# ---------------------------------------------------------------------------
# LOAD MODEL & DỮ LIỆU — dùng cache để chỉ đọc file 1 lần, dashboard chạy mượt hơn
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    """Nạp 3 model đã train sẵn. cache_resource vì đây là object model, không phải bảng dữ liệu."""
    logit = joblib.load(MODELS_DIR / "logit_scorecard.pkl")
    rf = joblib.load(MODELS_DIR / "rf_model.pkl")
    xgb = joblib.load(MODELS_DIR / "xgb_model.pkl")
    return logit, rf, xgb
 
 
@st.cache_data

def load_comparison_table() -> pd.DataFrame:
    """Bảng AUC/Gini/KS tổng hợp cho 3 mô hình chính."""

    df = pd.read_csv(METRICS_DIR / "model_comparison.csv")

    rename = {}
    for want in ["model", "AUC", "Gini", "KS"]:
        col = _detect_col(df, [want, want.lower(), want.upper()])
        rename[col] = want

    df = df.rename(columns=rename)

    # Chuẩn hóa tên Logistic Scorecard theo MODEL_ORDER
    df["model"] = df["model"].replace({
        "Scorecard / Logistic": "Logistic Scorecard"
    })

    # Chỉ giữ 3 mô hình cần so sánh
    df = df[df["model"].isin(MODEL_ORDER)].copy()

    # Sắp xếp theo thứ tự mô hình đã quy định
    df["model"] = pd.Categorical(
        df["model"],
        categories=MODEL_ORDER,
        ordered=True
    )

    return df.sort_values("model").reset_index(drop=True)
 
 
@st.cache_data
def load_baseline_metrics() -> dict | None:
    """Mô hình đối chứng (STT13) — chỉ để tham khảo, cho thấy 3 mô hình chính hơn baseline bao nhiêu."""
    path = METRICS_DIR / "baseline_metrics.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
 
 
@st.cache_data
def compute_train_test_auc(_logit_artifact: dict) -> pd.DataFrame:
    """
    Tự tính AUC train/test cho cả 3 mô hình, KHÔNG train lại — chỉ predict lại từ
    dữ liệu và model đã lưu sẵn. Đây là dữ liệu cho Tab "Train vs Test".
    """
    rows = []
 
    # --- Logistic Scorecard: tính PD từ artifact (hệ số) + woe_train/woe_test đã có sẵn ---
    woe_tr = pd.read_csv(DATA_DIR / "woe_train.csv")
    woe_te = pd.read_csv(DATA_DIR / "woe_test.csv")
    feats = _logit_artifact["features"]
    beta = _logit_artifact["params"]
    b0 = beta["const"]
 
    def _logit_predict(woe_df):
        logit_val = b0 + sum(beta[f] * woe_df[f] for f in feats)
        return 1 / (1 + np.exp(-logit_val))
 
    p_tr = _logit_predict(woe_tr)
    p_te = _logit_predict(woe_te)
    rows.append(dict(
        model="Logistic Scorecard",
        AUC_train=roc_auc_score(woe_tr[TARGET_COL], p_tr),
        AUC_test=roc_auc_score(woe_te[TARGET_COL], p_te),
    ))
 
    # --- Random Forest & XGBoost: đọc thẳng từ file dự đoán đã lưu sẵn ---
    for key, name, test_file, train_file in [
        ("rf", "Random Forest", "pred_rf.csv", "pred_rf_train.csv"),
        ("xgb", "XGBoost", "pred_xgb.csv", "pred_xgb_train.csv"),
    ]:
        y_tr, p_tr_m = _read_pred(key, REPORTS_DIR / train_file)
        y_te, p_te_m = _read_pred(key, REPORTS_DIR / test_file)
        rows.append(dict(
            model=name,
            AUC_train=roc_auc_score(y_tr, p_tr_m),
            AUC_test=roc_auc_score(y_te, p_te_m),
        ))
 
    out = pd.DataFrame(rows)
    out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
    return out.sort_values("model").reset_index(drop=True)
 
 
@st.cache_data
def compute_roc_curves() -> dict:
    """Tính điểm ROC (fpr, tpr) cho cả 3 mô hình trên tập TEST — dùng cho Tab ROC."""
    curves = {}
 
    logit_artifact, _, _ = load_models()
    woe_te = pd.read_csv(DATA_DIR / "woe_test.csv")
    feats = logit_artifact["features"]
    beta = logit_artifact["params"]
    b0 = beta["const"]
    logit_val = b0 + sum(beta[f] * woe_te[f] for f in feats)
    p_logit = 1 / (1 + np.exp(-logit_val))
    fpr, tpr, _ = roc_curve(woe_te[TARGET_COL], p_logit)
    curves["Logistic Scorecard"] = (fpr, tpr)
 
    for key, name, test_file in [("rf", "Random Forest", "pred_rf.csv"), ("xgb", "XGBoost", "pred_xgb.csv")]:
        y_te, p_te = _read_pred(key, REPORTS_DIR / test_file)
        fpr, tpr, _ = roc_curve(y_te, p_te)
        curves[name] = (fpr, tpr)
 
    return curves
 
 
@st.cache_data
def build_woe_mapping() -> dict:
    """
    Dựng bảng tra "giá trị gốc -> nhóm WoE" trực tiếp từ data/train.csv + data/woe_train.csv
    (hai file này CÙNG THỨ TỰ HÀNG nên ghép được theo vị trí). Việc này để phục vụ Tab
    "Chấm điểm khách hàng": khi người dùng gõ một giá trị gốc (vd LIMIT_BAL = 100000),
    mình cần biết giá trị đó rơi vào nhóm WoE nào thì mới tính điểm logistic được.
 
    Trả về: {tên_biến: [(giá_trị_thấp_nhất_của_nhóm, giá_trị_cao_nhất_của_nhóm, woe), ...]}
    """
    raw = pd.read_csv(DATA_DIR / "train.csv")
    woe = pd.read_csv(DATA_DIR / "woe_train.csv")
 
    mapping: dict[str, list] = {}
    common_cols = [c for c in woe.columns if c in raw.columns and c not in [TARGET_COL, "ID"]]
    for col in common_cols:
        tmp = pd.DataFrame({"raw": raw[col].values, "woe": woe[col].values})
        bins = []
        for woe_val, g in tmp.groupby("woe"):
            bins.append((g["raw"].min(), g["raw"].max(), float(woe_val)))
        mapping[col] = sorted(bins, key=lambda b: b[0])
    return mapping
 
 
def raw_to_woe(var: str, value: float, mapping: dict) -> float:
    """Tra WoE của 1 giá trị gốc theo nhóm đã học từ train. Giá trị nằm ngoài mọi nhóm đã thấy
    (vd khách nhập số quá lớn/quá nhỏ so với dữ liệu train) sẽ được gán vào nhóm gần nhất,
    thay vì trả về 0 — như vậy vẫn ra một điểm hợp lý thay vì một điểm "vô nghĩa"."""
    bins = mapping.get(var, [])
    if not bins:
        return 0.0
    for lo, hi, woe in bins:
        if lo <= value <= hi:
            return woe
    if value < bins[0][0]:
        return bins[0][2]
    return bins[-1][2]
 
 
# ---------------------------------------------------------------------------
# NẠP TẤT CẢ DỮ LIỆU CẦN THIẾT, BÁO LỖI THÂN THIỆN NẾU THIẾU FILE
# ---------------------------------------------------------------------------
try:
    logit_artifact, rf_model, xgb_model = load_models()
    metrics_df = load_comparison_table()
    baseline_metrics = load_baseline_metrics()
    train_test_df = compute_train_test_auc(logit_artifact)
    roc_curves = compute_roc_curves()
    woe_mapping = build_woe_mapping()
    DATA_READY = True
except FileNotFoundError as e:
    DATA_READY = False
    st.warning(
        "Ơ, hình như trang này đang thiếu một vài file kết quả để hiển thị 🙈\n\n"
        f"Chi tiết lỗi: `{e}`\n\n"
        "Bạn kiểm tra giúp xem các bạn phụ trách model đã chạy xong và lưu đúng đường dẫn "
        "trong `models/` và `reports/` chưa nhé — chỉ cần đủ file là trang sẽ tự lên ngay!"
    )
    st.stop()
except KeyError as e:
    DATA_READY = False
    st.warning(
        "Trang đọc được file rồi, nhưng chưa đoán đúng tên cột bên trong 🙂\n\n"
        f"Chi tiết: `{e}`\n\n"
        "Mở file `pred_rf.csv`/`pred_xgb.csv` lên xem tên cột thật là gì, rồi khai báo "
        "vào biến PRED_COLS ở đầu file này giúp mình nhé."
    )
    st.stop()
 
 
# ============================================================================
# HEADER + VÀI CON SỐ NỔI BẬT NGAY ĐẦU TRANG
# ============================================================================

# ---------------------------------------------------------------------------
# TỔNG QUAN DASHBOARD
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Nền tổng thể: xanh xám rất nhạt */
    [data-testid="stAppViewContainer"] {
        background: #F8FAFC;
    }

    /* Giới hạn chiều rộng để nội dung không bị trải quá rộng */
    .block-container {
        max-width: 1600px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Tiêu đề chính */
    h1 {
        color: #17324D !important;
        font-size: 2rem !important;
        font-weight: 750 !important;
        letter-spacing: -0.5px;
        margin-bottom: 0.5rem !important;
    }

    h2 {
        color: #17324D !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
    }

    h3 {
        color: #244B64 !important;
        font-size: 1.15rem !important;
        font-weight: 650 !important;
    }

    /* Chữ nội dung */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: #334155;
        line-height: 1.65;
    }

    /* Bốn thẻ chỉ số: nền trắng trên nền xám */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE5EF;
        border-left: 4px solid #315B78;
        border-radius: 10px;
        padding: 20px 18px;
        min-height: 125px;
        box-shadow: 0 3px 10px rgba(23, 50, 77, 0.05);
    }

    div[data-testid="stMetricLabel"] p {
        color: #64748B !important;
        font-size: 0.9rem !important;
        font-weight: 550 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #17324D !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 0.85rem !important;
    }

    /* Tab: làm rõ tab đang được chọn */
    button[data-baseweb="tab"] {
        color: #64748B;
        font-size: 0.95rem;
        font-weight: 600;
        padding: 12px 18px;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #17324D !important;
        border-bottom-color: #168C83 !important;
    }

    /* Khung mở rộng và các vùng nội dung */
    div[data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #DCE5EF;
        border-radius: 9px;
    }

    /* Bảng */
    div[data-testid="stDataFrame"] {
        background: #FFFFFF;
        border: 1px solid #DCE5EF;
        border-radius: 9px;
        overflow: hidden;
    }

    /* Đường phân cách nhẹ nhàng */
    hr {
        border-color: #DCE5EF;
    }
</style>
""", unsafe_allow_html=True)


st.title("Mô hình & Chấm điểm khách hàng")

st.markdown(
    "Đánh giá và so sánh ba mô hình chấm điểm tín dụng "
    "trên cùng tập dữ liệu thử nghiệm: **Logistic Scorecard**, "
    "**Random Forest** và **XGBoost**."
)

st.caption(
    "TỔNG QUAN HIỆU NĂNG  ·  Gini, AUC, KS và chênh lệch Train–Test"
)

best_row = metrics_df.loc[metrics_df["Gini"].idxmax()]
gap_series = (train_test_df["AUC_train"] - train_test_df["AUC_test"]).abs()
worst_gap_row = train_test_df.loc[gap_series.idxmax()]

k1, k2, k3, k4 = st.columns(4, gap="medium")

k1.metric(
    "Mô hình có Gini cao nhất",
    best_row["model"],
    f'{best_row["Gini"]:.3f}',
)

k2.metric(
    "AUC",
    f'{best_row["AUC"]:.3f}',
)

k3.metric(
    "KS",
    f'{best_row["KS"]:.3f}',
)

k4.metric(
    "Chênh lệch AUC Train–Test",
    worst_gap_row["model"],
    f'{gap_series.loc[worst_gap_row.name]:.3f}',
    delta_color="inverse",
)

if baseline_metrics:
    with st.expander("So sánh với mô hình đối chứng (Baseline)"):
        st.json(baseline_metrics)
        st.caption(
            "Baseline là mốc tham chiếu để đánh giá mức cải thiện "
            "của các mô hình chấm điểm tín dụng."
        )

st.divider()

tab_compare, tab_roc, tab_overfit, tab_score = st.tabs(
    [
        "So sánh mô hình",
        "Đường cong ROC",
        "Train vs Test",
        "Chấm điểm khách hàng",
    ]
)
 
# ============================================================================
# TAB 1 — SO SÁNH MÔ HÌNH
# ============================================================================


with tab_compare:
    st.subheader("So sánh hiệu năng mô hình")
    st.caption(
        "Đánh giá khả năng phân biệt khách hàng theo AUC, Gini và KS. "
        "Giá trị cao hơn thể hiện khả năng phân biệt tốt hơn trên các chỉ số này."
    )

    # Chuẩn bị dữ liệu so sánh
    compare_df = metrics_df[["model", "AUC", "Gini", "KS"]].copy()
    compare_df = compare_df.reset_index(drop=True)
    compare_df = compare_df.rename(columns={
        "model": "Mô hình",
        "AUC": "AUC",
        "Gini": "Gini",
        "KS": "KS",
    })

    # Làm nổi bật giá trị cao nhất ở từng chỉ số
    def _highlight_best(s):
        is_best = s == s.max()
        return [
            (
                "background-color: #E8F1F8; "
                "color: #17324D; font-weight: 600"
            ) if v else ""
            for v in is_best
        ]

    styled_df = (
        compare_df.style
        .apply(_highlight_best, subset=["AUC", "Gini", "KS"])
        .format({
            "AUC": "{:.3f}",
            "Gini": "{:.3f}",
            "KS": "{:.3f}",
        })
        .set_properties(**{
            "text-align": "center",
            "padding": "10px 14px",
        })
        .set_properties(
            subset=["Mô hình"],
            **{"text-align": "left", "font-weight": "500"},
        )
                .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#E2EAF3"),
                    ("color", "#194A7B"),
                    ("font-weight", "700"),
                    ("font-size", "16px"),
                    ("padding", "12px 14px"),
                    ("text-align", "center"),
                    ("border", "1px solid #C7D4E2"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("color", "#356967"),
                    ("font-size", "15px"),
                    ("padding", "10px 14px"),
                    ("border", "1px solid #D8E1EA"),
                ],
            },
        ])
    )

    st.dataframe(
    styled_df,
    use_container_width=True,
    hide_index=True,
)
    
    with st.expander("Giải thích các chỉ số"):
        st.markdown(
            "- **AUC:** Đo khả năng phân biệt giữa hai nhóm khách hàng.\n"
            "- **Gini:** Chỉ số được tính từ AUC theo công thức.\n"
            "  Gini = 2 × AUC − 1.\n"
            "- **KS:** Đo mức độ phân tách giữa phân phối điểm của hai nhóm."
        )

    st.caption(
        f"Mô hình có Gini cao nhất trong kết quả hiện tại: "
        f"**{best_row['model']}**. "
        "Việc lựa chọn mô hình cũng cần xem xét khả năng giải thích "
        "và chênh lệch hiệu năng giữa tập huấn luyện và kiểm thử."
    )

    # Biểu đồ: màu sắc nhất quán theo từng chỉ số
    metric_colors = {
        "AUC": "#244B64",
        "Gini": "#168C83",
        "KS": "#C39A3A",
    }

    fig_bar = go.Figure()

    for metric in ["AUC", "Gini", "KS"]:
        fig_bar.add_bar(
            name=metric,
            x=metrics_df["model"],
            y=metrics_df[metric],
            marker_color=metric_colors[metric],
            marker_line_width=0,
            text=[f"{v:.3f}" for v in metrics_df[metric]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "<b>%{x}</b><br>"
                + metric
                + ": %{y:.4f}<extra></extra>"
            ),
        )

    fig_bar.update_layout(
        barmode="group",
        height=420,
        margin=dict(l=20, r=20, t=35, b=30),
        paper_bgcolor="#F2F5FA",
        plot_bgcolor="#F8FAFC",
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="#334155",
        ),
        legend=dict(
            title_text="Chỉ số",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        xaxis=dict(
            title=None,
            showgrid=False,
            showline=True,
            linecolor="#CBD5E1",
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Giá trị chỉ số",
            range=[0, 1.12],
            dtick=0.2,
            gridcolor="#E8EDF3",
            zeroline=False,
        ),
        bargap=0.25,
        bargroupgap=0.08,
        hoverlabel=dict(
            bgcolor="white",
            font_color="#17324D",
        ),
    )

    st.plotly_chart(
        fig_bar,
        use_container_width=True,
        config={"displayModeBar": False},
    )
 
# ============================================================================
# TAB 2 — ROC CURVE
# ============================================================================


with tab_roc:
    st.subheader("Phân tích khả năng phân biệt rủi ro")
    st.caption(
        "ROC thể hiện khả năng phân biệt khách hàng có và không vỡ nợ. "
        "AUC càng cao, khả năng phân biệt tổng thể càng tốt."
    )

    # ---------------------------------------------------------
    # 1. Thẻ AUC: dùng số liệu thực tế của dự án
    # ---------------------------------------------------------
    auc_cols = st.columns(3)
    auc_descriptions = {
        "Logistic Scorecard": "Mô hình scorecard",
        "Random Forest": "Mô hình rừng ngẫu nhiên",
        "XGBoost": "Mô hình tăng cường",
    }

    for col, model in zip(auc_cols, MODEL_ORDER):
        row = metrics_df.loc[metrics_df["model"] == model, "AUC"]

        with col:
            if not row.empty:
                auc_value = float(row.iloc[0])
                st.metric(
                    label=model,
                    value=f"{auc_value:.3f}",
                    help=auc_descriptions.get(model, ""),
                )

    st.markdown("---")

    # ---------------------------------------------------------
    # 2. Biểu đồ ROC: chỉ hiển thị một biểu đồ
    # ---------------------------------------------------------
    fig_roc = go.Figure()

    line_styles = {
        "Logistic Scorecard": "solid",
        "Random Forest": "dash",
        "XGBoost": "solid",
    }

    for model in MODEL_ORDER:
        if model not in roc_curves:
            continue

        fpr, tpr = roc_curves[model]

        auc_row = metrics_df.loc[
            metrics_df["model"] == model, "AUC"
        ]
        if auc_row.empty:
            continue

        auc_value = float(auc_row.iloc[0])

        fig_roc.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{model} · AUC {auc_value:.3f}",
                line=dict(
                    color=PALETTE[model],
                    width=3 if model == best_row["model"] else 2.2,
                    dash=line_styles[model],
                ),
                hovertemplate=(
                    f"<b>{model}</b><br>"
                    "FPR: %{x:.3f}<br>"
                    "TPR: %{y:.3f}<extra></extra>"
                ),
            )
        )

    # Đường tham chiếu: dự đoán ngẫu nhiên
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Ngẫu nhiên · AUC 0.500",
            line=dict(
                color="#A0AEC0",
                width=1.5,
                dash="dot",
            ),
            hoverinfo="skip",
        )
    )

    fig_roc.update_layout(
        height=520,
        margin=dict(l=35, r=25, t=65, b=45),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="#24364B",
        ),
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
        ),
        xaxis=dict(
            title="Tỷ lệ dự đoán dương tính giả (FPR)",
            range=[0, 1],
            dtick=0.2,
            tickformat=".1f",
            gridcolor="#E8EDF3",
            linecolor="#CBD5E1",
            zeroline=False,
            constrain="domain",
        ),
        yaxis=dict(
            title="Tỷ lệ dự đoán dương tính đúng (TPR)",
            range=[0, 1],
            dtick=0.2,
            tickformat=".1f",
            gridcolor="#E8EDF3",
            linecolor="#CBD5E1",
            zeroline=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#CBD5E1",
            font_color="#24364B",
        ),
    )

    st.plotly_chart(
        fig_roc,
        use_container_width=True,
        config={"displayModeBar": False},
    )

    st.caption(
        "Đường chéo nét đứt là mốc dự đoán ngẫu nhiên. "
        "Các đường ROC được tính từ kết quả dự đoán trên tập test."
    )
    
 
# ============================================================================
# TAB 3 — TRAIN VS TEST (KIỂM TRA OVERFITTING)
# ============================================================================
with tab_overfit:
    st.subheader("Mô hình có \"học vẹt\" tập train không? 🧐")
 
    ot = train_test_df.copy()
    ot["gap"] = ot["AUC_train"] - ot["AUC_test"]
 
    fig_ot = go.Figure()
    fig_ot.add_bar(name="AUC Train", x=ot["model"], y=ot["AUC_train"], marker_color="#B0BEC5")
    fig_ot.add_bar(name="AUC Test", x=ot["model"], y=ot["AUC_test"],
                    marker_color=[PALETTE[m] for m in ot["model"]])
    fig_ot.update_layout(barmode="group", height=380, yaxis_range=[0.5, 1.0])
    st.plotly_chart(fig_ot, use_container_width=True)
 
    def _flag_gap(v):
        return "background-color:#FDECEA; color:#C62828; font-weight:600" if v > 0.03 else ""
 
    st.dataframe(
        ot.set_index("model")[["AUC_train", "AUC_test", "gap"]]
        .style.map(_flag_gap, subset=["gap"]).format("{:.4f}"),
        use_container_width=True,
    )
    st.caption(
        "Dòng nào chênh lệch Train–Test vượt 0.03 sẽ được tô đỏ nhẹ — không phải \"báo động\", "
        "chỉ là gợi ý để nhóm mình xem lại (giảm độ sâu cây, tăng regularization...). "
        "Ngưỡng 0.03 là quy ước riêng của nhóm, không phải chuẩn bắt buộc."
    )
 
 
# ============================================================================
# TAB 4 — CHẤM ĐIỂM KHÁCH HÀNG
# ============================================================================
with tab_score:
    st.subheader("Thử chấm điểm cho một khách hàng 💳")
    st.caption(
        "Bạn cứ thoải mái nhập một hồ sơ bất kỳ — trang chỉ dùng model đã học sẵn để "
        "*dự đoán*, không có bước huấn luyện nào chạy lại khi bạn bấm nút đâu, yên tâm nhé."
    )
 
    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**👤 Thông tin cơ bản**")
            limit_bal = st.number_input("Hạn mức tín dụng (LIMIT_BAL)", min_value=0, value=100_000, step=10_000)
            age = st.number_input("Tuổi", min_value=18, max_value=100, value=30)
            sex = st.selectbox("Giới tính", ["Male", "Female"])
            education = st.selectbox("Học vấn", ["Graduate school", "University", "High school", "Others"])
            marriage = st.selectbox("Hôn nhân", ["Married", "Single", "Others"])
        with c2:
            st.markdown("**🕒 Lịch sử thanh toán (PAY_x)**")
            pay_opts = list(range(-2, 9))
            pay_0 = st.selectbox("PAY_0 (tháng gần nhất)", pay_opts, index=pay_opts.index(0))
            pay_2 = st.selectbox("PAY_2", pay_opts, index=pay_opts.index(0))
            pay_3 = st.selectbox("PAY_3", pay_opts, index=pay_opts.index(0))
            pay_4 = st.selectbox("PAY_4", pay_opts, index=pay_opts.index(0))
            pay_5 = st.selectbox("PAY_5", pay_opts, index=pay_opts.index(0))
            pay_6 = st.selectbox("PAY_6", pay_opts, index=pay_opts.index(0))
        with c3:
            st.markdown("**🧾 Hoá đơn & thanh toán gần nhất**")
            bill_amt1 = st.number_input("BILL_AMT1", min_value=0, value=50_000, step=5_000)
            pay_amt1 = st.number_input("PAY_AMT1", min_value=0, value=5_000, step=1_000)
            bill_amt2 = st.number_input("BILL_AMT2", min_value=0, value=48_000, step=5_000)
            pay_amt2 = st.number_input("PAY_AMT2", min_value=0, value=5_000, step=1_000)
 
        threshold = st.slider(
            "Ngưỡng duyệt thử nghiệm (PD ≤ ngưỡng thì duyệt)", 0.0, 1.0, 0.30, 0.01,
            help="Chỉ để bạn thử nghiệm cho vui — ngưỡng chính thức của nhóm nằm ở phần "
                 "phân tích chi phí bên trang Giải thích & Fairness.",
        )
        submitted = st.form_submit_button("🔍 Chấm điểm ngay", use_container_width=True)
 
    if submitted:
        raw_input = dict(
            LIMIT_BAL=limit_bal, AGE=age, SEX=1 if sex == "Male" else 2,
            EDUCATION={"Graduate school": 1, "University": 2, "High school": 3, "Others": 4}[education],
            MARRIAGE={"Married": 1, "Single": 2, "Others": 3}[marriage],
            PAY_0=pay_0, PAY_2=pay_2, PAY_3=pay_3, PAY_4=pay_4, PAY_5=pay_5, PAY_6=pay_6,
            BILL_AMT1=bill_amt1, PAY_AMT1=pay_amt1, BILL_AMT2=bill_amt2, PAY_AMT2=pay_amt2,
        )
 
        # --- Nhánh Logistic Scorecard: đổi raw -> WoE theo nhóm học từ train, rồi tính PD/điểm ---
        feats = logit_artifact["features"]
        beta = logit_artifact["params"]
        b0 = beta["const"]
        factor, offset = logit_artifact["factor"], logit_artifact["offset"]
 
        logit_sum = b0
        for var in feats:
            if var in raw_input:
                logit_sum += beta[var] * raw_to_woe(var, raw_input[var], woe_mapping)
        pd_logit = 1 / (1 + np.exp(-logit_sum))
        score_logit = float(np.clip(offset - factor * np.log(pd_logit / (1 - pd_logit)), 300, 850))
 
        edges, labels = logit_artifact["band_edges"], logit_artifact["band_labels"]
        band_logit = pd.cut([score_logit], bins=edges, labels=labels, include_lowest=True)[0]
 
        # --- Nhánh RF / XGBoost: dùng thẳng biến gốc, đúng thứ tự cột lúc train ---
        rf_feats = list(rf_model.feature_names_in_)
        xgb_feats = list(xgb_model.feature_names_in_)
        X_rf = pd.DataFrame([{f: raw_input.get(f, 0) for f in rf_feats}])
        X_xgb = pd.DataFrame([{f: raw_input.get(f, 0) for f in xgb_feats}])
        pd_rf = rf_model.predict_proba(X_rf)[0, 1]
        pd_xgb = xgb_model.predict_proba(X_xgb)[0, 1]
 
        
        
        st.divider()
        st.subheader("Kết quả đánh giá tín dụng")
        st.markdown(
            """
            <div style="
                color: #243B53;
                font-size: 17px;
                font-weight: 600;
                line-height: 1.6;
                margin: 4px 0 18px 0;
            ">
                Ba mô hình cùng đánh giá một hồ sơ.
                Trạng thái được xác định theo ngưỡng PD thử nghiệm bạn đã chọn.
            </div>
            """,
            unsafe_allow_html=True,
        )

        results = [
            ("Logistic Scorecard", pd_logit, score_logit, band_logit),
            ("Random Forest", pd_rf, None, None),
            ("XGBoost", pd_xgb, None, None),
        ]

        result_cols = st.columns(3, gap="medium")

        for col, (name, pd_val, score_val, band_val) in zip(
            result_cols, results
        ):
            with col:
                with st.container(border=True):
                    # Tiêu đề đồng nhất giữa ba thẻ
                    st.markdown(f"**{name}**")
                    st.divider()

                    # Chỉ số chính: cùng vị trí trên cả ba thẻ
                    st.metric(
                        label="Xác suất vỡ nợ (PD)",
                        value=f"{pd_val:.2%}",
                    )

                    st.divider()

                    # Dành cùng không gian cho thông tin Scorecard
                    if score_val is not None:
                        st.metric(
                            label="Điểm tín dụng",
                            value=f"{int(score_val)}",
                        )
                        st.metric(
                            label="Nhóm rủi ro",
                            value=f"Nhóm {band_val}",
                        )
                    else:
                        st.metric(
                            label="Điểm tín dụng",
                            value="—",
                            help="Mô hình này không tạo điểm Scorecard.",
                        )
                        st.metric(
                            label="Nhóm rủi ro",
                            value="—",
                            help="Phân nhóm A–E hiện chỉ áp dụng cho Scorecard.",
                        )

                    st.divider()

                    # Trạng thái nổi bật, dùng màu và biểu tượng rõ ràng
                    if pd_val <= threshold:
                        st.markdown(
                            """
                            <div style="
                                background:#E8F5E9;
                                border:1px solid #81C784;
                                border-radius:8px;
                                padding:12px 8px;
                                text-align:center;
                                color:#1B5E20;
                                font-weight:800;
                                font-size:15px;
                            ">
                                ✓ TRONG NGƯỠNG
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            """
                            <div style="
                                background:#FDECEC;
                                border:1px solid #E57373;
                                border-radius:8px;
                                padding:12px 8px;
                                text-align:center;
                                color:#B71C1C;
                                font-weight:800;
                                font-size:15px;
                            ">
                                ⚠ VƯỢT NGƯỠNG RỦI RO
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.caption(f"Ngưỡng PD: {threshold:.0%}")

        st.markdown(
            """
            <div style="
                border-left: 4px solid #D99A22;
                background-color: #FFF9ED;
                padding: 12px 16px;
                margin-top: 18px;
                border-radius: 0 6px 6px 0;
                color: #59451F;
                font-size: 15px;
                line-height: 1.6;
            ">
                <span style="
                    font-weight: 800;
                    color: #8A5A00;
                    font-size: 15px;
                ">⚠ LƯU Ý</span>
                <span style="padding-left: 8px;">
                    Trạng thái chỉ phản ánh việc PD có nằm trong ngưỡng thử nghiệm
                    hay không, không phải quyết định cấp tín dụng chính thức.
                    Điểm tín dụng và nhóm rủi ro A–E chỉ áp dụng cho Logistic Scorecard.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        