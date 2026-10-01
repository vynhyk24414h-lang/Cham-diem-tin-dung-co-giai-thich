import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="Tổng quan", page_icon="🏠", layout="wide")

# --- CUSTOM CSS ĐỂ LÀM SỐ TO VÀ IN ĐẬM (Tương thích Dark/Light Mode) ---
st.markdown("""
<style>
/* Chỉnh số liệu KPI to và đậm */
div[data-testid="stMetricValue"] {
    font-size: 45px !important;
    font-weight: 900 !important;
    color: #FF4B4B; /* Màu đỏ nổi bật */
}
div[data-testid="stMetricLabel"] {
    font-size: 20px !important;
    font-weight: bold !important;
}
/* Chỉnh tiêu đề in đậm */
h2, h3 {
    font-weight: 800 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🏠 Trang 1 — Tổng quan & Dữ liệu")
st.markdown("---")

# A. Tổng quan dữ liệu
st.markdown("## **A. Tổng quan dữ liệu**")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="👥 Tổng số khách hàng", value="30,000")
with col2:
    st.metric(label="⚠️ Tỷ lệ vỡ nợ (Default)", value="22.12%")
with col3:
    st.metric(label="🌟 Số biến WoE", value="16")
with col4:
    st.metric(label="🤖 Số mô hình test", value="3")

st.write("") # Tạo khoảng trắng
st.info("""
**Chi tiết:** Bộ dữ liệu gồm 30.000 khách hàng. Biến mục tiêu là khả năng khách hàng không thanh toán đúng hạn vào tháng tiếp theo. 
Sau quá trình làm sạch, chuẩn hóa và xử lý WoE/IV, tập dữ liệu được chia thành **Train (24.000)** và **Test (6.000)**. 
16 biến có khả năng phân biệt cao nhất (IV ≥ 0.02) được lựa chọn làm đầu vào cho mô hình chấm điểm tín dụng.
""")

st.markdown("<br><br>", unsafe_allow_html=True)

# B. Phân bố Default
st.markdown("## **B. Phân bố biến mục tiêu (Default)**")
col_b1, col_b2 = st.columns([1, 2.5])

df_default = pd.DataFrame({
    "Trạng thái": ["0 (Không vỡ nợ)", "1 (Vỡ nợ)"],
    "Số lượng": [23364, 6636],
    "Tỷ lệ (%)": [77.88, 22.12]
})

with col_b1:
    st.markdown("<br><b style='font-size:18px;'>Sự mất cân bằng dữ liệu:</b>", unsafe_allow_html=True)
    st.dataframe(df_default.style.format({"Tỷ lệ (%)": "{:.2f}%"}), hide_index=True, use_container_width=True)

with col_b2:
    fig_def = px.bar(
        df_default, 
        x="Trạng thái", 
        y="Số lượng", 
        text="Số lượng",
        hover_data=["Tỷ lệ (%)"],
        color="Trạng thái",
        color_discrete_sequence=["#1f77b4", "#ff7f0e"] 
    )
    fig_def.update_traces(texttemplate='<b>%{text}</b>', textposition='inside', textfont_size=24, insidetextanchor="middle")
    fig_def.update_layout(
        showlegend=False, 
        height=500,
        title=dict(text="<b>Phân bố khách hàng Vỡ nợ vs Không vỡ nợ</b>", font=dict(size=24)),
        xaxis=dict(title="<b>Trạng thái</b>", tickfont=dict(size=18, weight="bold")),
        yaxis=dict(title="<b>Số lượng khách hàng</b>", tickfont=dict(size=16)),
        font=dict(family="Arial") 
    )
    st.plotly_chart(fig_def, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# C. WoE / IV
st.markdown("## **C. Khả năng phân biệt của các biến (WoE / IV)**")

# Dữ liệu IV ĐỒNG BỘ 100% VỚI BẢNG 4 TRONG PDF (Lấy 16 biến IV >= 0.02)
iv_data = {
    "Variable": [
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", 
        "LIMIT_BAL", "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", 
        "PAY_AMT6", "PAY_AMT4", "PAY_AMT5", "EDUCATION", "BILL_AMT6", "AGE"
    ],
    "IV": [
        0.888519, 0.551451, 0.418478, 0.364325, 0.332939, 0.292779, 
        0.196834, 0.166456, 0.149679, 0.126682, 
        0.093162, 0.085107, 0.081023, 0.036380, 0.022571, 0.021667
    ],
    "Strength": [
        "Rất mạnh", "Rất mạnh", "Mạnh", "Mạnh", "Mạnh", "Trung bình", 
        "Trung bình", "Trung bình", "Trung bình", "Trung bình", 
        "Yếu", "Yếu", "Yếu", "Yếu", "Yếu", "Yếu"
    ] # Phân loại theo quy tắc chung của IV
}
df_iv = pd.DataFrame(iv_data)

col_c1, col_c2 = st.columns([1, 1.5])

with col_c1:
    st.markdown("<b style='font-size:18px;'>Bảng giá trị Information Value (IV)</b>", unsafe_allow_html=True)
    st.dataframe(df_iv, height=600, use_container_width=True)

with col_c2:
    fig_iv = px.bar(
        df_iv.sort_values("IV", ascending=True), 
        x="IV", 
        y="Variable", 
        orientation='h',
        text="IV",
        hover_data=["Strength"],
        color="IV",
        color_continuous_scale="Viridis"
    )
    fig_iv.update_traces(texttemplate='<b>%{text:.4f}</b>', textposition='outside', textfont_size=16)
    fig_iv.update_layout(
        height=650,
        coloraxis_showscale=False, 
        title=dict(text="<b>Khả năng phân biệt (IV) của các biến</b>", font=dict(size=24)),
        xaxis=dict(title="<b>Giá trị IV</b>", tickfont=dict(size=14)),
        yaxis=dict(title="<b>Tên biến</b>", tickfont=dict(size=12, weight="bold")),
        font=dict(family="Arial") 
    )
    st.plotly_chart(fig_iv, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# D. Correlation WoE
st.markdown("## **D. Ma trận tương quan (Correlation) sau chuyển đổi WoE**")
st.info("💡 **Ghi chú quan trọng:** Không có cặp biến nào có độ tương quan |correlation| ≥ 0.7 sau khi chuyển đổi WoE, đảm bảo mô hình không bị hiện tượng đa cộng tuyến nghiêm trọng.")

# Tải dữ liệu thật để tính ma trận tương quan ĐỒNG BỘ 100% VỚI HÌNH 2 TRONG PDF
woe_path = os.path.join("data", "woe_train.csv")
if os.path.exists(woe_path):
    df_woe = pd.read_csv(woe_path)
    # Lấy 16 biến (bỏ cột default) theo đúng thứ tự của IV
    features = [col for col in df_iv["Variable"].tolist() if col in df_woe.columns]
    corr_matrix = df_woe[features].corr()
else:
    # Fallback nếu không thấy file (Không xảy ra vì file đã tồn tại)
    st.error("Không tìm thấy file data/woe_train.csv")
    corr_matrix = pd.DataFrame(np.eye(16), columns=df_iv["Variable"], index=df_iv["Variable"])

fig_corr = px.imshow(
    corr_matrix,
    labels=dict(color="Correlation"),
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    text_auto=".2f",
    color_continuous_scale="Viridis", # Trùng màu heatmap với PDF
    zmin=-1, zmax=1
)
fig_corr.update_traces(textfont=dict(size=14, family="Arial", weight="bold"))
fig_corr.update_layout(
    height=800, # Tăng height để chứa đủ 16 biến
    title=dict(text="<b>Correlation Heatmap (WoE Transformed Variables)</b>", font=dict(size=24)),
    xaxis=dict(tickfont=dict(size=12, weight="bold")),
    yaxis=dict(tickfont=dict(size=12, weight="bold")),
    font=dict(family="Arial")
)
st.plotly_chart(fig_corr, use_container_width=True)
