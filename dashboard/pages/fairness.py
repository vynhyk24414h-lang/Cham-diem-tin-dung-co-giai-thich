from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Fairness", page_icon="⚖️", layout="wide")

# ============================================================
# ĐƯỜNG DẪN & HẰNG SỐ
# ============================================================
ROOT = Path(__file__).resolve().parents[2]
TEST_PATH = ROOT / "data" / "test.csv"
PRED_PATH = ROOT / "reports" / "pred_xgb.csv"

DEFAULT_THRESHOLD = 0.5      # PD < ngưỡng thì Duyệt
AGE_CUT = 25                 # <= 25: trẻ tuổi
DI_LIMIT = 0.8               # Quy tắc 80% (Four-Fifths Rule)

C_BLUE, C_ORANGE = "#1f77b4", "#ff7f0e"
C_OK, C_BAD = "#2E7D32", "#C62828"
FONT = dict(family="Arial", color="black")

st.markdown("""
<style>
div[data-testid="stMetricValue"] {
    font-size: 45px !important;
    font-weight: 900 !important;
    color: #FF4B4B;
}
div[data-testid="stMetricLabel"] {
    font-size: 20px !important;
    font-weight: bold !important;
    color: #31333F;
}
h2, h3 { font-weight: 800 !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DỮ LIỆU & TÍNH TOÁN (đúng theo code gốc)
# ============================================================
@st.cache_data
def load_data():
    test = pd.read_csv(TEST_PATH)
    pred = pd.read_csv(PRED_PATH)
    df = test[["SEX", "AGE"]].copy()
    df["prob_default"] = pred["prob_default"].values
    df["AGE_group"] = df["AGE"].apply(
        lambda x: f"Trẻ tuổi (<={AGE_CUT})" if x <= AGE_CUT else f"Trưởng thành (>{AGE_CUT})")
    df["SEX_group"] = df["SEX"].map({1: "Nam", 2: "Nữ"})
    return df


def group_metrics(df, col, thr):
    """Số hồ sơ + tỷ lệ duyệt theo nhóm; DI = tỷ lệ nhóm thấp / nhóm cao."""
    d = df.assign(Approved=(df["prob_default"] < thr).astype(int))
    m = d.groupby(col)["Approved"].agg(["count", "mean"])
    m.columns = ["Số hồ sơ", "Tỷ lệ duyệt"]
    weak = m["Tỷ lệ duyệt"].idxmin()
    strong = m["Tỷ lệ duyệt"].idxmax()
    di = m.loc[weak, "Tỷ lệ duyệt"] / m.loc[strong, "Tỷ lệ duyệt"] if m.loc[strong, "Tỷ lệ duyệt"] > 0 else np.nan
    return m, di, weak, strong


def verdict(di):
    return "ĐẠT (Không thiên lệch)" if di >= DI_LIMIT else "KHÔNG ĐẠT (Có thiên lệch)"


def base_layout(title, height):
    return dict(height=height, title=dict(text=f"<b>{title}</b>", font=dict(size=22)),
                font=FONT, paper_bgcolor="white", plot_bgcolor="white",
                margin=dict(l=20, r=40, t=80, b=60),
                xaxis=dict(automargin=True, title=dict(standoff=15)),
                yaxis=dict(automargin=True, title=dict(standoff=15)))


def show(fig):
    st.plotly_chart(fig, use_container_width=True, theme=None)


def rate_bar(m, title):
    colors = [C_BLUE, C_ORANGE] + [C_BLUE] * 5
    fig = go.Figure(go.Bar(
        x=m.index.tolist(), y=(m["Tỷ lệ duyệt"] * 100).round(2),
        marker_color=colors[:len(m)],
        text=[f"<b>{v:.2f}%</b>" for v in m["Tỷ lệ duyệt"] * 100],
        textposition="inside", insidetextanchor="middle", textfont=dict(size=22, color="white"),
        customdata=m["Số hồ sơ"], hovertemplate="%{x}<br>Tỷ lệ duyệt: %{y:.2f}%<br>Số hồ sơ: %{customdata:,}<extra></extra>",
    ))
    fig.update_layout(**base_layout(title, 450), showlegend=False)
    fig.update_xaxes(tickfont=dict(size=18), showline=True, linecolor="#333")
    fig.update_yaxes(title="<b>Tỷ lệ duyệt (%)</b>", range=[0, 100], showline=True, linecolor="#333",
                     gridcolor="#E5E5E5")
    return fig


# ============================================================
# TIÊU ĐỀ + NẠP DỮ LIỆU
# ============================================================
st.title("⚖️ Fairness: Công bằng mô hình")
st.markdown("---")

if not (TEST_PATH.exists() and PRED_PATH.exists()):
    st.error("Thiếu `data/test.csv` hoặc `reports/pred_xgb.csv` (cột `prob_default`).")
    st.stop()

df = load_data()

thr = st.slider("Ngưỡng phê duyệt (PD < ngưỡng thì Duyệt)", 0.05, 0.95, DEFAULT_THRESHOLD, 0.01)

sex_m, di_sex, sex_weak, sex_strong = group_metrics(df, "SEX_group", thr)
age_m, di_age, age_weak, age_strong = group_metrics(df, "AGE_group", thr)

# ============================================================
# A. TỔNG QUAN
# ============================================================
st.markdown("## **A. Tổng quan kiểm định công bằng**")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("👥 Số hồ sơ kiểm thử", f"{len(df):,}")
with c2:
    st.metric("✅ Tỷ lệ duyệt chung", f"{(df['prob_default'] < thr).mean():.2%}")
with c3:
    st.metric("⚖️ DI — Giới tính", f"{di_sex:.2%}")
with c4:
    st.metric("🎂 DI — Độ tuổi", f"{di_age:.2%}")

st.write("")
st.markdown(f"""
<div style='font-size: 18px; line-height: 1.7; background-color: #D6E6FA; padding: 20px; border-radius: 10px;'>
<b>Phương pháp:</b> Dùng xác suất vỡ nợ (PD) dự báo của XGBoost trên tập test. Hồ sơ được <b>Duyệt</b> khi
PD &lt; <b>{thr:.2f}</b>. Chỉ số <b>Disparate Impact (DI)</b> = tỷ lệ duyệt của nhóm yếu thế ÷ tỷ lệ duyệt của nhóm còn lại.
Theo <b>Quy tắc 80% (Four-Fifths Rule)</b>, DI ≥ 80% được coi là không thiên lệch.
Hai thuộc tính nhạy cảm được kiểm tra: <b>Giới tính (SEX)</b> và <b>Độ tuổi (AGE)</b>
(trẻ tuổi ≤ {AGE_CUT}).
</div>
""", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)


# ============================================================
# B / C. TỪNG THUỘC TÍNH
# ============================================================
def section(letter, title, m, di, weak, strong, chart_title):
    st.markdown(f"## **{letter}. {title}**")
    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.markdown("<br><b style='font-size:18px;'>Tỷ lệ duyệt theo nhóm:</b>", unsafe_allow_html=True)
        st.dataframe(m.style.format({"Số hồ sơ": "{:,}", "Tỷ lệ duyệt": "{:.2%}"}),
                     use_container_width=True)
        st.metric("Chỉ số DI", f"{di:.2%}")
        if di >= DI_LIMIT:
            st.success(f"✅ **{verdict(di)}**")
        else:
            st.error(f"❌ **{verdict(di)}**")
        st.caption(f"Nhóm yếu thế: **{weak}** · Nhóm đối chiếu: **{strong}**")
    with col2:
        show(rate_bar(m, chart_title))


section("B", "Kiểm định thiên lệch theo Giới tính (SEX)", sex_m, di_sex, sex_weak, sex_strong,
        "Tỷ lệ duyệt theo Giới tính")
st.markdown("<br><br>", unsafe_allow_html=True)
section("C", "Kiểm định thiên lệch theo Độ tuổi (AGE)", age_m, di_age, age_weak, age_strong,
        "Tỷ lệ duyệt theo Độ tuổi")
st.markdown("<br><br>", unsafe_allow_html=True)

# ============================================================
# D. TỔNG HỢP + ĐỘ NHẠY THEO NGƯỠNG
# ============================================================
st.markdown("## **D. Tổng hợp Quy tắc 80%**")

summary = pd.DataFrame({
    "Thuộc tính": ["Giới tính (SEX)", "Độ tuổi (AGE)"],
    "Nhóm yếu thế": [sex_weak, age_weak],
    "Tỷ lệ duyệt yếu thế": [sex_m.loc[sex_weak, "Tỷ lệ duyệt"], age_m.loc[age_weak, "Tỷ lệ duyệt"]],
    "Tỷ lệ duyệt đối chiếu": [sex_m.loc[sex_strong, "Tỷ lệ duyệt"], age_m.loc[age_strong, "Tỷ lệ duyệt"]],
    "DI": [di_sex, di_age],
    "Đánh giá Quy tắc 80%": [verdict(di_sex), verdict(di_age)],
})
st.dataframe(summary.style.format({"Tỷ lệ duyệt yếu thế": "{:.2%}", "Tỷ lệ duyệt đối chiếu": "{:.2%}",
                                   "DI": "{:.2%}"}),
             hide_index=True, use_container_width=True)

st.info("💡 **Ghi chú:** DI thấp hơn 80% cho thấy nhóm yếu thế có tỷ lệ được duyệt thấp đáng kể so với nhóm còn lại. "
        "Đây là kiểm định thống kê dựa trên kết quả đầu ra của mô hình, chưa kết luận về nguyên nhân; "
        "cần kết hợp phân tích SHAP (đóng góp của SEX, AGE) để diễn giải.")