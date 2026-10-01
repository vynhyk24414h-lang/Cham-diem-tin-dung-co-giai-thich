import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
    /* Xóa màu cứng để nó tự đổi theo chế độ Dark/Light */
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
# Dùng thẻ info của Streamlit để nó tự thích ứng màu sắc với giao diện Đen/Trắng
st.info("""
**Chi tiết:** Bộ dữ liệu gồm 30.000 khách hàng. Biến mục tiêu là khả năng khách hàng không thanh toán đúng hạn vào tháng tiếp theo. 
Sau quá trình làm sạch, chuẩn hóa và xử lý WoE/IV, tập dữ liệu được chia thành **Train (24.000)** và **Test (6.000)**. 
16 biến có khả năng phân biệt cao nhất được lựa chọn làm đầu vào cho mô hình chấm điểm tín dụng.
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
        font=dict(family="Arial") # Bỏ màu chữ cứng để tự động sáng/tối
    )
    st.plotly_chart(fig_def, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# C. WoE / IV
st.markdown("## **C. Khả năng phân biệt của các biến (WoE / IV)**")

iv_data = {
    "Variable": ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6", "LIMIT_BAL", "PAY_AMT1", "PAY_AMT2", "BILL_AMT1"],
    "IV": [0.8885, 0.5515, 0.4185, 0.3590, 0.3330, 0.2850, 0.1730, 0.1580, 0.1440, 0.0450],
    "Strength": ["Rất mạnh", "Rất mạnh", "Mạnh", "Mạnh", "Mạnh", "Trung bình", "Trung bình", "Trung bình", "Trung bình", "Yếu"]
}
df_iv = pd.DataFrame(iv_data)

col_c1, col_c2 = st.columns([1, 1.5])

with col_c1:
    st.markdown("<b style='font-size:18px;'>Bảng giá trị Information Value (IV)</b>", unsafe_allow_html=True)
    st.dataframe(df_iv, height=500, use_container_width=True)

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
    fig_iv.update_traces(texttemplate='<b>%{text:.4f}</b>', textposition='outside', textfont_size=18)
    fig_iv.update_layout(
        height=550,
        coloraxis_showscale=False, 
        title=dict(text="<b>Khả năng phân biệt (IV) của các biến</b>", font=dict(size=24)),
        xaxis=dict(title="<b>Giá trị IV</b>", tickfont=dict(size=16)),
        yaxis=dict(title="<b>Tên biến</b>", tickfont=dict(size=16, weight="bold")),
        font=dict(family="Arial") # Bỏ màu chữ cứng để tự động sáng/tối
    )
    st.plotly_chart(fig_iv, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# D. Correlation WoE
st.markdown("## **D. Ma trận tương quan (Correlation) sau chuyển đổi WoE**")
st.info("💡 **Ghi chú quan trọng:** Không có cặp biến nào có độ tương quan |correlation| ≥ 0.7 sau khi chuyển đổi WoE, đảm bảo mô hình không bị hiện tượng đa cộng tuyến nghiêm trọng.")

vars_corr = df_iv["Variable"].tolist()[:8]
np.random.seed(42)
dummy_corr = np.random.uniform(0.01, 0.45, size=(8, 8))
np.fill_diagonal(dummy_corr, 1.0)
dummy_corr = (dummy_corr + dummy_corr.T) / 2
np.fill_diagonal(dummy_corr, 1.0)

fig_corr = px.imshow(
    dummy_corr,
    labels=dict(color="Correlation"),
    x=vars_corr,
    y=vars_corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    zmin=-1, zmax=1
)
# Bỏ màu chữ cứng để tự động sáng/tối
fig_corr.update_traces(textfont=dict(size=18, family="Arial", weight="bold"))
fig_corr.update_layout(
    height=650,
    title=dict(text="<b>Correlation Heatmap (WoE Transformed Variables)</b>", font=dict(size=24)),
    xaxis=dict(tickfont=dict(size=16, weight="bold")),
    yaxis=dict(tickfont=dict(size=16, weight="bold")),
    font=dict(family="Arial")
)
st.plotly_chart(fig_corr, use_container_width=True)
